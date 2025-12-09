import logging
import re

import dotenv
import tenacity
from openai import OpenAI
from openai.types import *
from openai.types.chat import *

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

tmpl_inc = re.compile(r"\<:inc\s+(.*?)\s*:\>")


def parse_file(file: str):
    with open(file, "r", encoding="utf-8") as f:
        tmpl = f.read()

    def fn_inc(m):
        with open(m.group(1), "r", encoding="utf-8") as f:
            return f.read()

    tmpl = tmpl_inc.sub(fn_inc, tmpl)
    return tmpl


client = OpenAI(timeout=None)


@tenacity.retry(
    wait=tenacity.wait_random(min=1, max=2),
    before_sleep=tenacity.before_sleep_log(logger, logging.ERROR),
)
def ask_gpt(chat_hist, model: ChatModel = "gpt-4o", silent=False) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=chat_hist,
        stream=True,
    )
    resp_blocks = []
    for block in resp:
        if cont := block.choices[0].delta.content:
            if not silent:
                print(cont, end="", flush=True)
            resp_blocks.append(cont)
    return "".join(resp_blocks)
