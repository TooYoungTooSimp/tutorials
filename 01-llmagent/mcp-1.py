from gpt_common import *
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

mcp = FastMCP()
db = FAISS.load_local(
    "faiss_db",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)


@mcp.tool()
def get_keywords() -> str:
    """
    return the keywords for RAG search
    """
    import json

    with open("keywords.txt", "r", encoding="utf-8") as f:
        return json.dumps(f.read().strip().split(), ensure_ascii=False)


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
