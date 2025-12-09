import json
from openai import OpenAI
from dotenv import load_dotenv
from glob import glob
import re
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def get_available_texts():
    audio = glob("*.m4a")

    texts = [a.replace("m4a", "txt") for a in audio]
    texts = [t for t in texts if os.path.exists(t)]
    return texts


def preprocess_texts():
    repeat_killer = re.compile(r"(.+)\1{9,}", re.DOTALL)

    texts = get_available_texts()

    for idx, txt in enumerate(texts):
        with open(txt, "r", encoding="utf-8") as f:
            content = f.read()
        content = repeat_killer.sub("", content)
        texts[idx] = content
    splitter = CharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=2048,
        separator="",
    )
    from tqdm.auto import tqdm

    db = None
    for txt in tqdm(texts):
        pieces = splitter.split_text(txt)
        if not db:
            db = FAISS.from_texts(pieces, OpenAIEmbeddings())
        else:
            db.add_texts(pieces)
    db.save_local("faiss_db")
    print(db)


def extract_keywords():
    repeat_killer = re.compile(r"(.+)\1{9,}", re.DOTALL)

    texts = get_available_texts()

    for idx, txt in enumerate(texts):
        with open(txt, "r", encoding="utf-8") as f:
            content = f.read()
        content = repeat_killer.sub("", content)
        texts[idx] = content
    splitter = CharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=2048,
        separator="",
    )
    texts = splitter.split_text("\n".join(texts))
    from multiprocessing.pool import ThreadPool

    client = OpenAI()

    def extract_kws(text: str):
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": ""
                    "你是一名中石化新入职大学生，正在听领导的讲座并写感想。"
                    "讲座材料以听写转录的形式给出，注意同音字近音字的存在，"
                    "直接提取这段讲话中的五个关键词，不需要解释，每个关键词用空格隔开",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        print(resp.choices[0].message.content)
        return resp.choices[0].message.content

    with ThreadPool(8) as pool:
        kws = pool.map(extract_kws, texts)
    kws = [kw for k in kws for kw in k.split()]
    kws = list(set(kws))
    with open("keywords.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(kws))


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


if __name__ == "__main__":
    main()
    # preprocess_texts()
    # extract_keywords()
