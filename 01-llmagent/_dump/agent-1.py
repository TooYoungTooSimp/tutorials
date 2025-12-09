import yaml
from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    OpenAIServerModel,
    ToolCallingAgent,
    tool,
)
from gpt_common import *

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

"""

def main():
    client = OpenAI()
    db = FAISS.load_local(
        "faiss_db",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    tools_def = [
        {
            "type": "function",
            "name": "query_db",
            "description": "accept keyword and return related information",
            "parameters": {
                "type": "object",
                "properties": {
                    "kw": {
                        "type": "string",
                        "description": "The keyword to query the database",
                    }
                },
                "required": ["kw"],
            },
        }
    ]

    def real_ask(question: str):
        with open("keywords.txt", "r", encoding="utf-8") as f:
            keywords = f.read().strip()
        from openai.types.responses.response_input_param import Message

        input_msgs: list[Message] = [
            {
                "type": "message",
                "role": "system",
                "content": f"你是一名中石化新入职大学生，正在复习领导的讲座。讲座材料以听写转录的形式给出，注意同音字近音字的存在。必须调用函数（可多次）来进行关键词RAG检索，返回相关听写转录。参考关键词：{keywords}",
            },
            {
                "type": "message",
                "role": "user",
                "content": question,
            },
        ]
        resp = client.responses.create(
            model="gpt-4.1",
            tools=tools_def,
            input=input_msgs,
            tool_choice="required",
        )
        print(resp.output)
        for toolcall in resp.output:
            if toolcall.type != "function_call":
                continue
            if toolcall.name == "query_db":
                kw = json.loads(toolcall.arguments)["kw"]
                print(f"Keyword: {kw}")
                docs = []
                for skw in kw.split():
                    if not (skw := skw.strip()):
                        continue
                    print(f"Searching for keyword: {skw}")
                    docs.extend(db.similarity_search(skw, k=3))
                rag_result = "\n".join([doc.page_content for doc in docs])
                input_msgs.append(toolcall)
                input_msgs.append(
                    {
                        "type": "function_call_output",
                        "call_id": toolcall.call_id,
                        "output": str(rag_result),
                    }
                )
                print(f"{kw=} appended.")
        if input_msgs[-1]["type"] == "function_call_output":
            resp = client.responses.create(
                model="gpt-4.1",
                input=input_msgs,
                # tools=tools_def,
                stream=True,
            )
            for chunk in resp:
                if chunk.type == "response.output_text.delta":
                    print(chunk.delta, end="", flush=True)
            print()
        else:
            print(resp.output_text)

    while True:
        real_ask(input("=> "))

"""

db = FAISS.load_local(
    "faiss_db",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

with open("keywords.txt", "r", encoding="utf-8") as f:
    keywords = f.read().strip()
content = f"你是一名中石化新入职大学生，正在复习领导的讲座。讲座材料以听写转录的形式给出，注意同音字近音字的存在。可多次调用函数来进行关键词RAG检索，返回相关听写转录。"


@tool
def get_keywords() -> str:
    """
    return the keywords for RAG search
    """
    return " ".join(keywords.split())


@tool
def query_db(kw: str) -> str:
    """
    accept keyword and return related information

    Args:
        kw (str): The keyword to query the database
    """
    docs = []
    for skw in kw.split():
        if not (skw := skw.strip()):
            continue
        print(f"Searching for keyword: {skw}")
        docs.extend(db.similarity_search(skw, k=3))
    rag_result = "\n".join([doc.page_content for doc in docs])
    return rag_result


agent = CodeAgent(
    model=OpenAIServerModel("gpt-5"),
    tools=[get_keywords, query_db],  # add your tools here (don't remove final_answer)
    verbosity_level=1,
)

# from Gradio_UI import GradioUI

# GradioUI(agent).launch()

agent.run(f"{content}\n\n请帮我复习一下领导的讲座")
