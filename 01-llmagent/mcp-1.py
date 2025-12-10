from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
mcp = FastMCP()
db = FAISS.load_local(
    "faiss_db",
    OpenAIEmbeddings(),
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
