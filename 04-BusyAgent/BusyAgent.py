from openai import OpenAI
import subprocess
import uuid
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


load_dotenv()


def build_system_prompt(
    custom_prompt: str | None = None,
    selected_tools: list[str] | None = None,
    tool_snippets: dict[str, str] | None = None,
    prompt_guidelines: list[str] | None = None,
    append_system_prompt: str | None = None,
    cwd: str | None = None,
    context_files: list[dict[str, str]] | None = None,
) -> str:
    resolved_cwd = cwd or str(Path.cwd())
    prompt_cwd = resolved_cwd.replace("\\", "/")
    current_date = date.today().isoformat()

    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""
    context_items = context_files or []

    if custom_prompt:
        prompt = custom_prompt + append_section
        if context_items:
            prompt += "\n\n# Project Context\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for item in context_items:
                file_path = item.get("path", "unknown")
                content = item.get("content", "")
                prompt += f"## {file_path}\n\n{content}\n\n"

        prompt += f"\nCurrent date: {current_date}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    tools = selected_tools or ["run_shell_command"]
    snippets = tool_snippets or {}
    visible_tools = [name for name in tools if snippets.get(name)]
    if visible_tools:
        tools_list = "\n".join(f"- {name}: {snippets[name]}" for name in visible_tools)
    else:
        tools_list = "(none)"

    guideline_set: set[str] = set()
    guideline_list: list[str] = []

    def add_guideline(guideline: str) -> None:
        normalized = guideline.strip()
        if not normalized or normalized in guideline_set:
            return
        guideline_set.add(normalized)
        guideline_list.append(normalized)

    add_guideline(
        "Use run_shell_command only when needed; prefer direct answers when no command is required"
    )
    add_guideline(
        "When calling tools, pass plain text command input without JSON wrappers"
    )
    add_guideline("Be concise in responses")
    add_guideline("Show command intent clearly before using the shell tool")

    for guideline in prompt_guidelines or []:
        add_guideline(guideline)

    guidelines = "\n".join(f"- {g}" for g in guideline_list)

    prompt = (
        "You are an expert coding assistant operating inside BusyAgent, a tool-calling harness. "
        "You help users by reasoning, answering directly, and using tools when needed.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "Tool-calling policy:\n"
        "- You have exactly one tool: run_shell_command.\n"
        "- The shell is busybox ash.\n"
        "- If a shell command is needed, call the tool with plain text input containing only the command.\n"
        "- You may call the tool multiple times until the task is complete.\n"
        "- If no tool is needed, answer normally.\n\n"
        f"Guidelines:\n{guidelines}"
    )

    if append_section:
        prompt += append_section

    if context_items:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for item in context_items:
            file_path = item.get("path", "unknown")
            content = item.get("content", "")
            prompt += f"## {file_path}\n\n{content}\n\n"

    prompt += f"\nCurrent date: {current_date}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


SYSTEM_PROMPT = build_system_prompt(
    selected_tools=["run_shell_command"],
    tool_snippets={
        "run_shell_command": "Execute a shell command in busybox ash and return exit code + output."
    },
)

proc = subprocess.Popen(
    "busybox64u sh",
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)


def run_cmd(command: str):
    if proc.poll() is not None:
        raise RuntimeError("Process has exited")
    token = f"__CMD_DONE_{uuid.uuid4().hex}__"
    pattern = re.compile(rf"^{re.escape(token)} (\d+)\n?$")
    assert proc.stdin
    assert proc.stdout
    proc.stdin.write(command.rstrip("\n") + "\n")
    proc.stdin.write(f'printf "{token} %s\\n" "$?"\n')
    proc.stdin.flush()
    output = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("bash terminated before command completion")
        m = pattern.match(line)
        if m:
            exit_code = int(m.group(1))
            return exit_code, "".join(output)
        output.append(line)


def stream_one_request(client: OpenAI, tools, input_items):
    text_started = False
    pending_input = {}
    pending_name = {}
    pending_call_id = {}
    tool_calls = []
    response_items = []
    response = client.responses.create(
        model="gpt-5.4-mini",
        instructions=SYSTEM_PROMPT,
        tools=tools,
        input=input_items,
        stream=True,
    )

    for event in response:

        data = event.model_dump()

        etype = data.get("type")

        if etype == "response.output_text.delta":
            delta = data.get("delta") or ""
            if delta:
                if not text_started:
                    print("assistant> ", end="", flush=True)
                    text_started = True
                print(delta, end="", flush=True)
            continue

        if etype == "response.output_item.added":
            item = data.get("item", {})
            item_type = item.get("type")
            if item_type == "custom_tool_call":
                idx = data.get("output_index")
                pending_name[idx] = item.get("name")
                pending_call_id[idx] = item.get("call_id")
                payload = item.get("input")
                pending_input[idx] = payload or ""
            continue

        if etype == "response.custom_tool_call_input.delta":
            idx = data.get("output_index")
            delta = data.get("delta") or ""
            pending_input[idx] = (pending_input.get(idx) or "") + delta
            continue

        if etype == "response.custom_tool_call_input.done":
            idx = data.get("output_index")
            item = data.get("item", {})
            name = item.get("name") or pending_name.get(idx)
            call_id = item.get("call_id") or pending_call_id.get(idx)
            payload = item.get("input")
            if payload is None:
                payload = pending_input.get(idx) or ""
            tool_calls.append(
                {
                    "type": "custom_tool_call",
                    "name": name,
                    "call_id": call_id,
                    "input": payload,
                }
            )
            continue

        if etype == "response.completed":
            resp = data.get("response", {})
            response_items = resp.get("output", [])
            continue

    if text_started:
        print()

    if not response_items:
        raise RuntimeError("Stream completed without response output items")
    if not tool_calls:
        for item in response_items:
            raw = item
            if hasattr(item, "model_dump"):
                raw = item.model_dump()
            if not isinstance(raw, dict):
                continue
            item_type = raw.get("type")
            if item_type != "custom_tool_call":
                continue
            payload = raw.get("input")
            tool_calls.append(
                {
                    "type": item_type,
                    "name": raw.get("name"),
                    "call_id": raw.get("call_id"),
                    "input": payload or "",
                }
            )
    # aihubmix does not support tool namespace
    for response_item in response_items:
        if "namespace" in response_item:
            del response_item["namespace"]

    return response_items, tool_calls


client = OpenAI()

tools = [
    {
        "type": "custom",
        "name": "run_shell_command",
        "description": "Run a shell command in busybox ash. Input must be plain text command.",
    }
]

print("BusyAgent started. Type 'exit' or 'quit' to stop.")

history_items: list[Any] = []

try:
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        history_items.append({"role": "user", "content": user_text})

        while True:
            response_items, tool_calls = stream_one_request(
                client, tools, history_items
            )

            history_items.extend(response_items)

            if not tool_calls:
                break

            for call in tool_calls:
                if call.get("name") != "run_shell_command":
                    continue

                command = (call.get("input") or "").strip()
                if not command:
                    err_output = "[exit_code: 2]\n" "invalid_tool_input: empty command"
                    history_items.append(
                        {
                            "type": "custom_tool_call_output",
                            "call_id": call.get("call_id"),
                            "output": err_output,
                        }
                    )
                    print("tool> empty command")
                    continue

                print(f"tool> $ {command}")
                code, out = run_cmd(command)
                result_text = f"[exit_code: {code}]\n{out}"
                print(f"<tool_resp> {result_text.strip()}\n</tool_resp>")
                history_items.append(
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call.get("call_id"),
                        "output": result_text,
                    }
                )
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    finally:
        if proc.poll() is None:
            proc.terminate()
    sys.exit(0)
