# GPT-Researcher 全局架构与学习路线规划

你好！欢迎来到 Agent 开发的世界。`gpt-researcher` 是一个非常优秀的开源项目，它不仅仅是一个简单的套壳脚本，而是完整展示了**大模型（LLM）+ 检索增强生成（RAG）+ 多智能体（Multi-Agent）**这三大前沿技术的深度融合。

为了让你能够“抽丝剥茧”地掌握它，我将作为你的代码导师，用通俗易懂的语言、形象的类比，带你从底层原理（Why）到代码实现（How）逐步攻克这个项目。

---

## 1. 鸟瞰全局：GPT-Researcher 是如何运作的？

我们可以把 GPT-Researcher 想象成一个**“全自动化的新闻调查编辑部”**。

当用户提出一个复杂问题（例如：“总结2024年固态电池的最新突破及商业化前景”）时，如果仅仅依赖单个大模型，它可能会胡编乱造（幻觉），或者给出过时的信息。GPT-Researcher 的解决思路是**模拟人类团队的真实研究流**。

### 核心流转节点（Workflow Node）
1. **主编接单（Task Breakdown）**：接收用户问题，评估需要哪些维度的信息，并**拆解**成多个具体的搜索子查询（Sub-queries）。
2. **情报员外勤（Information Gathering）**：带着子查询，并行调用各种搜索引擎（Tavily, Google 等）去互联网上“进货”，并爬取网页全文。
3. **HR筛简历（Context Filtering & Rerank）**：面对海量的网页文本，过滤掉广告和无关废话，只把最核心的段落“萃取”出来。
4. **编辑部会审（Multi-Agent Orchestration）**：由不同的 Agent（主编、作家、审核员）共同协作，围绕共享的“研究笔记”进行撰写、交叉验证和修改，最终输出一份结构化的万字深度报告。

---

## 2. 结构化学习路线规划 (8大核心模块)

为了让你获得**专家级 (Expert-level)** 的理解，我将之前的路线进一步细化，拆解为 **8 个颗粒度更细的核心模块**。每个模块我们都会深入到具体的 `class` 和 `def`，并附带**可独立运行的最小示例 (MRE)**。

### 📍 模块一：单兵研究员与生命周期 (The Core Researcher Engine)
*   **深层解剖**：剖析 `gpt_researcher/agent.py` 中的 `GPTResearcher` 类。
*   **核心议题**：一个单兵 Agent 是如何管理配置、初始化工具、并贯穿“规划-搜索-合成”整个生命周期的。

### 📍 模块二：任务破冰与意图拆解 (Query Processing & Task Breakdown)
*   **深层解剖**：剖析 `actions/query_processing.py`。
*   **核心议题**：如何通过 Function Calling 技术，把一句含糊的自然语言转化为 LLM 可执行的结构化 `["子查询1", "子查询2"]` JSON 数组。

### 📍 模块三：多路情报网与工具自愈 (Web Scraping & Tool Routing)
*   **深层解剖**：剖析 `retrievers/` 和 `scraper/` 目录。
*   **核心议题**：Agent 如何动态选择合适的搜索 API（Tavily vs DuckDuckGo）；面对防爬虫、404 错误时，如何实现异常重试与自我纠错 (Self-Correction)。

### 📍 模块四：沙里淘金的深度 RAG (Advanced RAG & Reranking)
*   **深层解剖**：剖析 `context/` 与 `document/`。
*   **核心议题**：如何把杂乱的网页文本切块 (Chunking) 并向量化。重点讲解**重排 (Rerank)** 机制：为什么捞出 20 条网页数据，最后只喂给大模型 5 条？

### 📍 模块五：短时记忆与上下文防爆 (Memory & Context Control)
*   **深层解剖**：剖析 `memory/` 与上下文截断机制。
*   **核心议题**：当爬取的网页资料超过了 LLM 的 128K Token 上限怎么办？学习它如何通过滑动窗口、摘要压缩 (Summarization) 来守住 Token 预算。

### 📍 模块六：万字长文组装车间 (Report Generation & Assembly)
*   **深层解剖**：剖析 `actions/report_generation.py`。
*   **核心议题**：有了高质量的“上下文”后，如何通过合理的 Prompt Engineering，把碎片化的事实拼接成具有 Markdown 完美格式的深度研报。

### 📍 模块七：LangGraph 多智能体拓扑与状态流 (Multi-Agent Topology & State)
*   **深层解剖**：剖析 `multi_agents/main.py` 及图结构定义。
*   **核心议题**：从顺序流到中控流——解析 LangGraph 的节点 (Nodes) 与边 (Edges)；多 Agent 之间如何通过同一个 `ResearchState` 共享研究笔记，避免各说各话。

### 📍 模块八：角色扮演与交叉审校机制 (Role Playing & Collaboration)
*   **深层解剖**：剖析 `multi_agents/agents/` (Editor, Writer, Reviewer)。
*   **核心议题**：如何为不同的 Agent 撰写人设 System Prompt？审核员 (Reviewer) 是如何向作家 (Writer) 提出修改意见并打回重写的？

---

## 下一步行动 (Next Steps)

这份细化的 8 大模块路线图，将带你毫无死角地打通从底层 RAG 到顶层 Multi-Agent 架构的所有硬核技术点。

**请您再次过目，如果这 8 个模块的拆解粒度符合您的期望，请回复“继续”，我将立即为您呈上【模块一：单兵研究员与生命周期】的详细剖析文档和实操代码！**
