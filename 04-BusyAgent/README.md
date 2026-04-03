# We proudly present to you, BusyAgent

**大道至简。一个有且仅有一个 Tool 的 Agent。**

在这个充斥着各种臃肿框架、动辄需要 1GB 内存才能跑起来的“小龙虾”生态里，我们常常迷失在无尽的“工具箱”定义中：读文件要一个 Tool，写文件要一个 Tool，执行代码又要一个 Tool…… 

为什么不回归本质？既然现在最小的 LLM 都能写出像样的 Bash 命令，为什么我们不直接给它一个终端？

BusyAgent 应运而生。它放弃了繁杂的 API 封装，为大模型提供了一个最纯粹、最通用的单点接口：**Busybox**。

---

## 💡 核心设计哲学 (The Philosophy)

* **万物皆 Shell (The "One Tool" Agent)**
    我们删除了所有零碎的工具箱。读取文件？用 `cat`。写入文件？用重定向和 `EOF`。修改文件？用 `head`、`tail` 配合重定向。所有的操作，全部统一为**执行命令**。
* **跨平台利器：Busybox**
    为什么不直接用 bash？因为 Windows 默认没有 bash。而 Busybox 提供了一个跨平台的轻量级 CLI 工具集合与内置 shell (`ash`)，完美抹平了操作系统的鸿沟。
* **Stateful（有状态）的交互环境**
    与 OpenClaw/pi 框架中那种每次调用都“阅后即焚”的 Stateless（无状态）bash 工具不同，BusyAgent 是 **Stateful** 的。模型可以在 shell 中随意设置环境变量、更改工作目录 (`cd`)、串联一系列复杂的操作，并将其状态完美保留到下一步。

## 🛡️ 安全与扩展性 (Security & Sandboxing)

虽然 BusyAgent 目前直接在宿主机的 Busybox 进程中运行，但它的架构天生适配沙盒：
* **容器化:** 可以极其容易地将唯一的 Tool 桥接到 `docker cli`，在一个封闭的容器环境中安全地执行任何操作。
* **极致轻量沙盒:** 甚至可以直接对接 `crun`（借助 Linux namespaces 和 seccomp）构建硬隔离环境，在保证安全的同时榨干每一滴性能。