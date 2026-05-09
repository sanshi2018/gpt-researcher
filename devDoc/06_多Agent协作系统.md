# Phase 6: 多 Agent 协作系统

> 本章目标：深入理解 `multi_agents/` 下基于 LangGraph 的多 Agent 工作流，包括主编排器、子流程、状态传递、Human-in-the-Loop。

---

## 6.1 多Agent架构总览

```
┌───────────── ChiefEditorAgent (主编排器) ─────────────┐
│                                                        │
│  ┌────── LangGraph 主工作流 (ResearchState) ─────────┐ │
│  │                                                    │ │
│  │  [browser]  ──→  [planner]  ──→  [human]           │ │
│  │                                     │              │ │
│  │                    ┌────← revise ←──┤              │ │
│  │                    │                │ accept        │ │
│  │                    ↓                ↓              │ │
│  │              [planner]        [researcher]         │ │
│  │                                     │              │ │
│  │                                     ↓              │ │
│  │                               [writer]             │ │
│  │                                     │              │ │
│  │                                     ↓              │ │
│  │                              [publisher] ──→ END   │ │
│  └────────────────────────────────────────────────────┘ │
│                                                        │
│  ┌────── 子工作流 (DraftState) × N 并行 ─────────────┐ │
│  │  [researcher] ──→ [reviewer] ←──→ [reviser]        │ │
│  │                       │                             │ │
│  │                  accept (review=None)               │ │
│  │                       ↓                             │ │
│  │                      END                            │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 6.2 ChiefEditorAgent — 主编排器

**源码位置**：`multi_agents/agents/orchestrator.py`

### 初始化

```python
class ChiefEditorAgent:
    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None):
        self.task = task           # 研究任务配置
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}
        self.task_id = int(time.time())   # 时间戳作为任务ID
        self.output_dir = f"./outputs/run_{self.task_id}_{task['query'][:40]}"
```

### Agent 初始化

```python
def _initialize_agents(self):
    return {
        "writer":    WriterAgent(self.websocket, self.stream_output, self.headers),
        "editor":    EditorAgent(self.websocket, self.stream_output, self.tone, self.headers),
        "research":  ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
        "publisher": PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers),
        "human":     HumanAgent(self.websocket, self.stream_output, self.headers),
    }
```

### LangGraph 工作流定义

```python
def _create_workflow(self, agents):
    workflow = StateGraph(ResearchState)    # 使用 ResearchState TypedDict
    
    # 添加节点
    workflow.add_node("browser", agents["research"].run_initial_research)
    workflow.add_node("planner", agents["editor"].plan_research)
    workflow.add_node("researcher", agents["editor"].run_parallel_research)
    workflow.add_node("writer", agents["writer"].run)
    workflow.add_node("publisher", agents["publisher"].run)
    workflow.add_node("human", agents["human"].review_plan)
    
    # 添加边
    workflow.add_edge('browser', 'planner')
    workflow.add_edge('planner', 'human')
    workflow.add_edge('researcher', 'writer')
    workflow.add_edge('writer', 'publisher')
    workflow.set_entry_point("browser")
    workflow.add_edge('publisher', END)
    
    # 条件边: Human-in-the-Loop
    workflow.add_conditional_edges(
        'human',
        lambda review: "accept" if review['human_feedback'] is None else "revise",
        {"accept": "researcher", "revise": "planner"}
    )
    
    return workflow
```

### 运行研究任务

```python
async def run_research_task(self, task_id=None):
    research_team = self.init_research_team()
    chain = research_team.compile()     # 编译 LangGraph 图
    
    config = {
        "configurable": {
            "thread_id": task_id,
            "thread_ts": datetime.datetime.utcnow()
        }
    }
    
    result = await chain.ainvoke({"task": self.task}, config=config)
    return result
```

---

## 6.3 ResearchAgent — 研究执行

**源码位置**：`multi_agents/agents/researcher.py`

### 核心设计：每个 Agent 内部复用 GPTResearcher

```python
class ResearchAgent:
    async def research(self, query, research_report="research_report", 
                       parent_query="", verbose=True, source="web", tone=None):
        # ⭐ 每次研究都创建一个新的 GPTResearcher 实例
        researcher = GPTResearcher(
            query=query,
            report_type=research_report,
            parent_query=parent_query,
            verbose=verbose,
            report_source=source,
            tone=tone,
            websocket=self.websocket,
            headers=self.headers
        )
        
        # 执行完整的研究+报告流程
        await researcher.conduct_research()
        report = await researcher.write_report()
        return report
```

### 三种研究方法

```python
# 1. 初始研究 — 用于 browser 节点
async def run_initial_research(self, research_state: dict):
    task = research_state.get("task")
    query = task.get("query")
    return {"task": task, "initial_research": await self.research(query=query, ...)}

# 2. 子主题深度研究 — 用于 researcher 节点
async def run_subtopic_research(self, parent_query, subtopic, ...):
    report = await self.research(
        parent_query=parent_query, 
        query=subtopic,
        research_report="subtopic_report",  # 子主题报告类型
        ...
    )
    return {subtopic: report}

# 3. 深度研究 — 用于 DraftState 子流程
async def run_depth_research(self, draft_state: dict):
    topic = draft_state.get("topic")
    parent_query = draft_state["task"]["query"]
    research_draft = await self.run_subtopic_research(
        parent_query=parent_query, subtopic=topic, ...
    )
    return {"draft": research_draft}
```

---

## 6.4 EditorAgent — 研究规划与并行执行

**源码位置**：`multi_agents/agents/editor.py`

### plan_research() — 章节规划

```python
async def plan_research(self, research_state):
    initial_research = research_state.get("initial_research")
    task = research_state.get("task")
    human_feedback = research_state.get("human_feedback")
    max_sections = task.get("max_sections")
    
    prompt = [
        {"role": "system", "content": "You are a research editor..."},
        {"role": "user", "content": f"""
            Research summary: '{initial_research}'
            {human_feedback_instruction}
            Generate an outline of sections headers (max {max_sections}).
            Return JSON: {{'title': string, 'date': string, 'sections': [...]}}
        """},
    ]
    
    plan = await call_model(prompt=prompt, model=task.get("model"), response_format="json")
    
    return {
        "title": plan.get("title"),
        "date": plan.get("date"),
        "sections": plan.get("sections"),
    }
```

### run_parallel_research() — 并行子研究

```python
async def run_parallel_research(self, research_state):
    """为每个章节创建独立的子工作流，并行执行"""
    
    # 1. 创建子工作流（每个章节独立的 researcher → reviewer → reviser 循环）
    workflow = self._create_workflow()    # DraftState 子流程
    chain = workflow.compile()
    
    # 2. 并行执行所有章节的研究
    sections = research_state.get("sections")
    final_drafts = [
        chain.ainvoke(
            {"task": research_state["task"], "topic": section, "title": title},
            config={"tags": ["gpt-researcher"]}
        )
        for section in sections
    ]
    
    # 3. 使用 asyncio.gather 并行执行
    research_results = [
        result["draft"] 
        for result in await asyncio.gather(*final_drafts)
    ]
    
    return {"research_data": research_results}
```

### 子工作流定义（DraftState）

```python
def _create_workflow(self):
    """每个章节的研究→审查→修订循环"""
    agents = self._initialize_agents()
    workflow = StateGraph(DraftState)
    
    # 节点
    workflow.add_node("researcher", agents["research"].run_depth_research)
    workflow.add_node("reviewer", agents["reviewer"].run)
    workflow.add_node("reviser", agents["reviser"].run)
    
    # 边
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("reviser", "reviewer")
    
    # 条件边: 审查通过则结束，否则继续修订
    workflow.add_conditional_edges(
        "reviewer",
        lambda draft: "accept" if draft["review"] is None else "revise",
        {"accept": END, "revise": "reviser"},
    )
    
    return workflow
```

---

## 6.5 Human-in-the-Loop

### HumanAgent

```python
class HumanAgent:
    async def review_plan(self, research_state: dict):
        """人工审查研究计划"""
        task = research_state.get("task")
        
        if task.get("include_human_feedback"):
            # 通过 WebSocket 等待人工反馈
            human_feedback = await self._get_human_feedback(research_state)
            return {"human_feedback": human_feedback}
        
        return {"human_feedback": None}  # None = 自动接受
```

### 条件路由逻辑

```python
# human_feedback 为 None → accept → 进入 researcher 节点
# human_feedback 为具体内容 → revise → 回到 planner 重新规划
workflow.add_conditional_edges(
    'human',
    lambda review: "accept" if review['human_feedback'] is None else "revise",
    {"accept": "researcher", "revise": "planner"}
)
```

---

## 6.6 Reviewer 与 Reviser — 质量保证循环

### ReviewerAgent

```python
class ReviewerAgent:
    async def run(self, draft_state: dict):
        """审查研究草稿"""
        task = draft_state.get("task")
        draft = draft_state.get("draft")
        
        # 审查次数限制
        if self._review_count >= task.get("max_reviews", 2):
            return {"review": None}  # None = 审查通过
        
        # LLM 审查
        review = await self._review_draft(draft, task)
        
        if review_passes:
            return {"review": None}      # 通过
        else:
            return {"review": review}    # 返回修订建议
```

### ReviserAgent

```python
class ReviserAgent:
    async def run(self, draft_state: dict):
        """根据审查意见修订草稿"""
        draft = draft_state.get("draft")
        review = draft_state.get("review")
        
        revised_draft = await self._revise(draft, review)
        
        return {
            "draft": revised_draft,
            "revision_notes": review,
        }
```

### 质量保证循环流程

```
researcher 生成草稿
  → reviewer 审查
    → review = None? → 结束（质量合格）
    → review = "建议..." → reviser 修订
      → reviewer 再次审查
        → ... （循环直到通过或达到最大修订次数）
```

---

## 6.7 状态流转全景

```
初始状态:
  {"task": {...用户配置...}}

browser 节点后:
  {"task": {...}, "initial_research": "初步研究摘要..."}

planner 节点后:
  {"task": {...}, "initial_research": "...", 
   "title": "报告标题", "sections": ["章节1", "章节2", ...]}

human 节点后:
  {..., "human_feedback": None}  // 或具体反馈

researcher 节点后 (并行):
  {..., "research_data": [{"章节1": "报告内容1"}, {"章节2": "报告内容2"}]}

writer 节点后:
  {..., "report": "# 完整报告\n## 章节1\n..."}

publisher 节点后:
  {..., "report": "...", "sources": [...]}
  → END
```

---

## 6.8 与单Agent模式的对比

| 维度 | 单Agent (GPTResearcher) | 多Agent (ChiefEditorAgent) |
|------|------------------------|---------------------------|
| **编排方式** | Python 方法链 | LangGraph 状态图 |
| **状态管理** | 实例变量 (self.context) | TypedDict (ResearchState/DraftState) |
| **并行方式** | asyncio.gather on 子查询 | asyncio.gather on 章节研究 |
| **质量保证** | 无 | reviewer → reviser 循环 |
| **人工介入** | 不支持 | Human-in-the-Loop |
| **报告类型** | 单一报告 | 多章节协作报告 |
| **复用** | 直接使用 | 每个Agent封装一个GPTResearcher |

---

## 6.9 LangGraph 核心概念映射

| LangGraph 概念 | GPT-Researcher 实现 |
|----------------|---------------------|
| `StateGraph` | `StateGraph(ResearchState)` |
| `add_node` | 每个Agent的方法作为节点 |
| `add_edge` | 固定顺序的节点连接 |
| `add_conditional_edges` | human/reviewer 的分支逻辑 |
| `set_entry_point` | "browser" 节点 |
| `END` | publisher 完成后终止 |
| `compile()` | 编译为可执行链 |
| `ainvoke()` | 异步执行工作流 |
| `config` | thread_id 用于任务隔离 |

---

## 📌 本章要点回顾

- [x] 多Agent系统由 `ChiefEditorAgent` 通过 LangGraph `StateGraph` 编排
- [x] 主工作流 6 个节点：browser → planner → human → researcher → writer → publisher
- [x] 每个章节有独立的子工作流：researcher → reviewer → reviser 循环
- [x] Human-in-the-Loop 通过条件边实现：feedback=None 则 accept，否则 revise
- [x] `ResearchAgent` 内部复用 `GPTResearcher`，每次创建新实例
- [x] 章节研究通过 `asyncio.gather` 实现并行执行
- [x] Reviewer/Reviser 实现质量保证循环，有最大修订次数限制

---

## 🔬 动手实验

### 实验 1：绘制 LangGraph 工作流图
```python
from multi_agents.agents import ChiefEditorAgent

task = {"query": "AI in healthcare", "max_sections": 3, "model": "gpt-4o", ...}
chief = ChiefEditorAgent(task)
workflow = chief.init_research_team()
chain = workflow.compile()

# 可视化
chain.get_graph().print_ascii()
```

### 实验 2：跟踪状态变化
在每个节点方法中添加日志，观察 `ResearchState` 的字段变化。

### 实验 3：测试 Human-in-the-Loop
设置 `task["include_human_feedback"] = True`，通过 WebSocket 连接测试人工反馈流程。
