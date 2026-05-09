# 模块七：LangGraph 多智能体拓扑与状态流 (Multi-Agent Topology)

在前面 6 个模块中，我们都在研究 **单兵作战 (Single Agent)**：一个全能的 GPT-Researcher，从搜索、阅读、压缩、一直干到写长文。

但如果你是一家报社的老板，你绝不会让一个记者包揽所有工作。你会设置**主编、主笔、审稿人、排版员**。
这正是 `gpt-researcher/multi_agents/` 目录的精髓所在。它借助了 LangChain 生态中最强大的工具 **LangGraph**，将代码结构从“线性流水线”升级为了一个包含循环、条件判断和人类介入的“复杂状态机（有向图）”。

---

## 1. 架构总览：一家虚拟报社

打开 `multi_agents/agents/orchestrator.py`，你会看到整个“报社”的最高指挥官：`ChiefEditorAgent`。
他负责把报社的各种角色组装成一张网（Graph）：

```python
# orchestrator.py 节选
def _create_workflow(self, agents):
    workflow = StateGraph(ResearchState)

    # 1. 注册所有的“员工” (Nodes)
    workflow.add_node("browser", agents["research"].run_initial_research) # 资料搜集员
    workflow.add_node("planner", agents["editor"].plan_research)          # 策划编辑 (写大纲)
    workflow.add_node("human", agents["human"].review_plan)               # 人类总监 (你！)
    workflow.add_node("researcher", agents["editor"].run_parallel_research) # 调查小组
    workflow.add_node("writer", agents["writer"].run)                     # 主笔 (统稿)
    workflow.add_node("publisher", agents["publisher"].run)               # 排版发行

    # 2. 规定他们的工作流转顺序 (Edges)
    workflow.add_edge('browser', 'planner')
    workflow.add_edge('planner', 'human')

    # 3. 核心机制：条件分支 (Conditional Edges)
    # 如果人类总监没意见 (accept)，就把大纲交给调查小组去写；如果有修改意见 (revise)，打回给策划重新出大纲！
    workflow.add_conditional_edges(
        'human',
        lambda review: "accept" if review['human_feedback'] is None else "revise",
        {"accept": "researcher", "revise": "planner"}
    )
    # ...
```

这不仅解决了 LLM 幻觉的问题，还首次引入了 **HIL (Human-in-the-Loop, 人类在环)** 机制。机器干脏活累活，你只需要在最关键的“大纲制定”阶段点个头或者提句意见即可。

---

## 2. 惊艳的架构嵌套 (Nested Sub-Graphs)

你以为上面的图就是全部了？错。
GPT-Researcher 在 `multi_agents/agents/editor.py` 中，使用了一种极其硬核的高级特性：**嵌套子图 (Sub-Graph)**。

当大纲（比如有 3 个章节）通过人类审核，交给 `researcher` 节点执行时，主编 (`EditorAgent`) 并没有自己去写，而是**针对每个章节，拉起了 3 个独立的并发子流水线**！

让我们看看子流水线是怎么流转的（在 `_create_workflow` 方法中）：

```python
# editor.py 节选：这是处理单个章节的子流水线
workflow = StateGraph(DraftState)

workflow.add_node("researcher", agents["research"].run_depth_research) # 写初稿
workflow.add_node("reviewer", agents["reviewer"].run)                  # 挑刺审稿
workflow.add_node("reviser", agents["reviser"].run)                    # 苦逼修改

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_edge("reviser", "reviewer")

# 恶魔循环：如果审稿人提出了修改意见，就打回给 Reviser；修改完后，再交给 Reviewer 审... 直到 Reviewer 挑不出毛病为止。
workflow.add_conditional_edges(
    "reviewer",
    lambda draft: "accept" if draft["review"] is None else "revise",
    {"accept": END, "revise": "reviser"},
)
```

### 多智能体拓扑结构图 (Mermaid)

为了直观感受，我为你画了这幅架构图，清晰展现了父图与并发子图的嵌套关系：

```mermaid
graph TD
    %% 主流程 (Chief Editor)
    Start[发派任务] --> Browser[Browser: 初期搜集]
    Browser --> Planner[Planner: 拟定大纲]
    Planner --> Human{Human: 人类审核大纲}
    Human -- "打回修改" --> Planner
    
    Human -- "审核通过" --> ParallelGroup
    
    subgraph ParallelGroup [Editor: 并发子章节撰写 (Sub-Graphs)]
        direction TB
        subgraph Section 1
            R1[Researcher: 初稿] --> Rev1[Reviewer: 审核]
            Rev1 -- "不通过" --> Revi1[Reviser: 修改]
            Revi1 --> Rev1
            Rev1 -- "通过" --> End1((完稿))
        end
        subgraph Section N
            Rn[Researcher: 初稿] --> Revn[Reviewer: 审核]
            Revn -- "不通过" --> Revin[Reviser: 修改]
            Revin --> Revn
            Revn -- "通过" --> Endn((完稿))
        end
    end
    
    ParallelGroup --> Writer[Writer: 合并统稿与排版]
    Writer --> Publisher[Publisher: 输出PDF/MD]
    Publisher --> Finish[结束]
```

---

## 3. 状态流机制 (State Management)

在传统的编程中，函数之间传递的是参数。但在 LangGraph 中，Agent 之间抛来抛去的是一个“共享的记事本”，叫做 **State (状态)**。

打开 `multi_agents/memory/research.py`，你能看到最高层级的状态机定义：
```python
from typing import TypedDict, List, Dict

class ResearchState(TypedDict):
    task: dict               # 原始任务配置
    initial_research: str    # 破冰搜集回来的资料
    sections: List[str]      # 拟定好的大纲列表
    research_data: List[dict]# 各个小节写完的草稿
    title: str               # 报告标题
    human_feedback: str      # 人类给的修改意见
```
每个 Agent 拿到这个 `ResearchState`，只读取自己需要的部分，然后把自己干完的活儿更新进去（比如 Planner 把写好的大纲填进 `sections`），再把本子抛给下一个人。这彻底解耦了模块之间的依赖，也是目前硅谷构建 Agentic Workflow (智能体工作流) 最推荐的模式。

---

## 下一步 (Next Steps)

现在，图结构（流水线）搭好了，角色（打工人）也分配完了。
但是，**各个角色内部到底是靠什么运作的呢？** 审稿人（Reviewer）怎么知道该怎么挑刺？修改人（Reviser）怎么知道该怎么改？

这就是我们要解密的最后一个谜题：**【模块八：角色扮演与交叉审校机制 (Role Playing & Collaboration)】**。我们将深入那些隐藏在 `.py` 文件深处的绝密 Prompt。

**当您消化完这个复杂的有向图拓扑后，请回复“继续输出模块八”，我们来完成整个源码解剖之旅的最终章！**
