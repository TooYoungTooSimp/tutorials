import json
from openai import OpenAI
from dotenv import load_dotenv
from glob import glob
import re
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from _types import *


load_dotenv()


def read_once(pth):
    with open(pth, "r", encoding="utf-8") as f:
        return f.read()


def write_once(pth, content):
    with open(pth, "w", encoding="utf-8") as f:
        f.write(content)


all_texts = [x for x in glob("*分.txt")]
mappings = json.loads(read_once("mappings.json"))
all_texts = "\n\n".join(
    [f"# {mappings[x.replace(".txt",".m4a")]}\n{read_once(x)}" for x in all_texts]
)
outline = read_once("outline.txt")

outline_titles = """
I. 引言：启程与定位
II. 思想铸魂：筑牢忠诚干净担当的根基
III. 认知油田：走进技术与管理的核心
IV. 融入职场：解锁角色转变的密码
V. 总结与展望：扬帆起航向未来
"""
outline_titles = [x.strip() for x in outline_titles.splitlines() if x.strip()]

current_summary = ""

for title in outline_titles:
    hist: list[Message] = [
        {
            "type": "message",
            "role": "system",
            "content": "你是一名中石化新入职大学生，正在复习两周来的培训讲座并写总结。讲座材料以听写转录的形式给出，注意同音字近音字的存在。",
        },
        {
            "type": "message",
            "role": "user",
            "content": (
                f"你要根据以下大纲:\n{outline}\n\n和以下讲座材料:\n{all_texts}\n\n写一篇总结。"
                f"现在，你仅需要撰写“{title}”这一部分"
                + (
                    f"以下是已经写过的内容，无需重复：\n{current_summary}\n"
                    if current_summary
                    else ""
                )
            ),
        },
    ]

    client = OpenAI()
    resp = client.responses.create(
        model="gpt-4.1",
        input=hist,
        stream=True,
    )
    try:
        for chunk in resp:
            if chunk.type == "response.output_text.delta":
                print(chunk.delta, end="", flush=True)
            if chunk.type == "response.output_text.done":
                current_summary += chunk.text
    except Exception as e:
        print(f"Error during response streaming: {e}")

write_once("summary.txt", current_summary)
