# 01-LLMAgent · Agentic RAG 学习项目 （Under construction）

这是一个以学习为目的、从零实现的 LLM Agent 工程，用来回馈社区。项目聚焦 Agentic RAG（检索增强生成 + 工具调用），面向初学者与希望快速对比多种 Agent 实现路径的开发者。

本目录内的所有实际操作都通过 `1-intro.ipynb` 完成。你无需单独运行脚本，Notebook 已整合数据准备、三种 Agent 模式的示例与可选的简单 UI。

## 实现模式概览
- 直连工具（函数式）
  - 思路：Agent 直接调用本地函数工具（例如“获取关键词”“向量检索”），将检索结果拼接进上下文后生成答案。
  - 优点：最直观、依赖少、易调试；很适合入门与本地实验。
  - 适用：单体项目、快速原型、教学示例。
- 图/工作流（状态机驱动）
  - 思路：将“对话→判断→调用工具→汇总”等步骤编排成图（状态节点 + 条件边），Agent 在图上推进。
  - 优点：结构清晰、可视化编排、便于扩展与维护。
  - 适用：复杂流程、可观察、可插拔的生产级编排思维训练。
- MCP 远程工具（协议化）
  - 思路：工具通过 MCP（Model Context Protocol）以服务形式暴露，Agent 以统一接口远程调用。
  - 优点：强解耦、跨进程/跨语言、便于团队协作与多工具生态。
  - 适用：多语言工具混用、集中式工具维护、分布式部署场景。

## RAG 工作机制简述
- 文本预处理：将长文本分为带重叠的片段，便于嵌入与召回。
- 向量检索：根据问题或关键词在向量库中做相似度搜索（默认 FAISS，可替换）。
- 上下文拼接：将命中的片段作为上下文提供给大模型组织最终回答。
- 关键词辅助：先由模型/规则抽取关键词，再做召回，提升准确与可控性。

## 先决条件
- Python 3.10+（推荐 3.10/3.11）
- 一个兼容 OpenAI 的模型服务（或自建/代理），在项目根放置 `.env`
- 依赖安装（PowerShell）
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install -U pip
  pip install smolagents langchain langgraph langchain-openai langchain-community openai python-dotenv tenacity tqdm gradio
  pip install faiss-cpu  # Windows 若安装受限，优先使用 Conda 或改用其他向量库（如 Chroma）
  pip install mcp fastmcp # 启用 MCP 模式所需
  ```

## 环境变量示例（.env）
```ini
OPENAI_API_KEY=sk-xxx
# 可选：如使用自定义/代理服务
# OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

## 如何使用 1-intro.ipynb
1. 打开 Notebook：在 VS Code 中选择 `.venv` 作为 Python 解释器，打开 `1-intro.ipynb`。
2. 数据准备：
   - 按单元格提示导入或检测你的转录文本。
   - 一键完成“切分-嵌入-入库”（构建 `faiss_db/`）。
   - 一键完成“关键词抽取”（生成 `keywords.txt`）。
3. 选择 Agent 模式：
   - 在 Notebook 中选择“直连工具 / 图工作流 / MCP 远程工具”任一模式。
   - 各模式的示例单元格已经封装好对应的调用逻辑。
4. 提问与对比：
   - 在对应单元格输入问题，观察工具调用与 RAG 检索效果。
   - 可在三种模式间切换，比较行为差异和优缺点。
5. 可选 UI：
   - Notebook 提供一个基于 Gradio 的简易交互界面单元格，方便非技术同学体验。

## MCP 模式补充说明
- Notebook 提供“一键启动/停止 MCP 工具服务”的示例单元格（默认 `http://127.0.0.1:8000/mcp`）。
- Agent 通过统一接口加载远程工具，与直连工具用法保持一致。
- 推荐场景：大型团队、跨语言工具、远程/容器化运行。

## 常见问题
- Windows 下 FAISS 安装
  - 首选 `faiss-cpu` 预编译包或 Conda；不行时可在 Notebook 中换用其他向量库（如 Chroma）。
- 模型名与兼容性
  - 示例以 OpenAI 兼容接口为目标，若你使用代理/网关，请在 `.env` 设置 `OPENAI_BASE_URL` 并在 Notebook 中选择可用的模型名称/路由。
- API 速率与费用
  - 关键词抽取与嵌入步骤会产生 API 调用，请按需批量/并行与速率控制，注意费用与配额。

## 项目声明
- 学习与回馈：本项目旨在用最小可行代码复现 Agentic RAG 的核心路径，帮助学习与二次开发。
- 欢迎贡献：欢迎使用反馈、Issue 与 PR，共同完善示例与文档。
