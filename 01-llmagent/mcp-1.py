from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from os import environ

load_dotenv()
EMB_MODEL_NAME = environ.get("EMB_MODEL_NAME", "text-embedding-3")

mcp = FastMCP()
db = FAISS.load_local(
    "faiss_db",
    OpenAIEmbeddings(model=EMB_MODEL_NAME),
    allow_dangerous_deserialization=True,
)


@mcp.tool()
def query_paperdb(kw: str) -> str:
    """
    Accept keywords and return related paper titles, multiple keywords should be separated by '|'.
    This tool should be called before output any realworld-related information.

    Args:
        kw (str): The keyword to query the database
    """
    docs = []
    for skw in kw.split("|"):
        if not (skw := skw.strip()):
            continue
        print(f"Searching for keyword: {skw}")
        docs.extend(db.similarity_search(skw, k=10))
    rag_result = "\n".join([doc.page_content for doc in docs])
    return rag_result


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
