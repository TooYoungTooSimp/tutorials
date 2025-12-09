from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from gpt_common import *

db = FAISS.load_local(
    "faiss_db",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)


def query_db(kw: str) -> str:
    docs = []
    for skw in kw.split():
        if not (skw := skw.strip()):
            continue
        print(f"Searching for keyword: {skw}")
        docs.extend(db.similarity_search(skw, k=1))
    rag_result = "\n".join([doc.page_content for doc in docs])
    return rag_result


with open("keywords.txt", "r", encoding="utf-8") as f:
    keywords = f.read().strip()
sys_prompt = f"你是一名中石化新入职大学生，正在复习领导的讲座。讲座材料以听写转录的形式给出，注意同音字近音字的存在。必须调用函数（可多次）来进行关键词RAG检索，返回相关听写转录。参考关键词：{keywords}"

from langchain.tools import Tool

query_db_tool = Tool(
    name="query_db",
    func=query_db,
    description="accept keyword and return related information",
)

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

chat = ChatOpenAI(model="gpt-5", verbose=True)
chat_with_tools = chat.bind_tools([query_db_tool])


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([query_db_tool]))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

agent = builder.compile()

messages = [
    SystemMessage(content=sys_prompt),
    HumanMessage(content="请帮我复习一下领导的讲座，挑几个关键词就行"),
]

response = agent.invoke({"messages": messages})

print(response["messages"][-1].content)

for idx, part in enumerate(response["messages"]):
    print(f"Part {idx}: \n{part.content}\n\n")
