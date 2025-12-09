from smolagents import CodeAgent, OpenAIServerModel, ToolCollection

from gpt_common import *

content = f"你是一名中石化新入职大学生，正在复习领导的讲座。讲座材料以听写转录的形式给出，注意同音字近音字的存在。可多次调用函数来进行关键词RAG检索，返回相关听写转录。"

with ToolCollection.from_mcp(
    {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"},
    trust_remote_code=True,
) as tool_collection:
    agent = CodeAgent(
        model=OpenAIServerModel("gpt-5"),
        tools=[*tool_collection.tools],
        verbosity_level=1,
    )

    # from Gradio_UI import GradioUI
    # GradioUI(agent).launch()

    agent.run(f"{content}\n\n请帮我复习一下领导的讲座")
