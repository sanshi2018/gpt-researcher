# Phase 5: LLM 调用与 Function Call

> 本章目标：深入理解 LLM 提供商抽象层、LLM 调用链、Prompt 工程系统，以及类 Function Call 的 Agent 选择和 MCP 工具调用。

---

## 5.1 LLM 提供商架构

```
┌─────────────── LLM 调用架构 ──────────────────┐
│                                                 │
│  create_chat_completion()  ←── 统一调用接口      │
│       ↓                                         │
│  GenericLLMProvider.from_provider()              │
│       ↓ (工厂方法)                               │
│  ┌──────────────────────────────────────┐        │
│  │ OpenAI │ Anthropic │ Google │ Ollama │        │
│  │ Azure  │ Groq      │ XAI   │ Bedrock│        │
│  │ Cohere │ Fireworks │ DeepSeek │ ...  │        │
│  └──────────────────────────────────────┘        │
│       ↓                                         │
│  LangChain Chat Model (ainvoke / astream)       │
│       ↓                                         │
│  Response / Streaming Response                   │
└─────────────────────────────────────────────────┘
```

---

## 5.2 GenericLLMProvider — LLM 工厂

**源码位置**：`gpt_researcher/llm_provider/generic/base.py`

### 支持的 LLM 提供商（30+）

```python
_SUPPORTED_PROVIDERS = {
    "openai", "anthropic", "azure_openai", "cohere",
    "google_vertexai", "google_genai", "fireworks",
    "ollama", "together", "mistralai", "huggingface",
    "groq", "bedrock", "dashscope", "xai", "deepseek",
    "litellm", "gigachat", "openrouter", "vllm_openai",
    "aimlapi", "netmind", "forge", "avian", "minimax",
}
```

### 工厂方法

```python
class GenericLLMProvider:
    def __init__(self, llm, chat_log=None, verbose=True):
        self.llm = llm  # LangChain Chat Model 实例
    
    @classmethod
    def from_provider(cls, provider, chat_log=None, verbose=True, **kwargs):
        """根据提供商名称创建 LLM 实例"""
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(**kwargs)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(base_url=os.environ["OLLAMA_BASE_URL"], **kwargs)
        elif provider == "deepseek":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_base='https://api.deepseek.com',
                openai_api_key=os.environ["DEEPSEEK_API_KEY"],
                **kwargs
            )
        # ... 更多提供商
        
        return cls(llm, chat_log, verbose=verbose)
```

### 聊天响应处理

```python
async def get_chat_response(self, messages, stream, websocket=None, **kwargs):
    if not stream:
        # 非流式: 使用 ainvoke
        output = await self.llm.ainvoke(messages, **kwargs)
        return output.content
    else:
        # 流式: 使用 astream
        return await self.stream_response(messages, websocket, **kwargs)

async def stream_response(self, messages, websocket=None, **kwargs):
    """流式响应 — 按段落推送"""
    paragraph = ""
    response = ""
    
    async for chunk in self.llm.astream(messages, **kwargs):
        content = chunk.content
        if not content:
            continue
        response += content
        paragraph += content
        if "\n" in paragraph:
            await self._send_output(paragraph, websocket)
            paragraph = ""
    
    if paragraph:
        await self._send_output(paragraph, websocket)
    
    return response
```

### 自动依赖安装

```python
def _check_pkg(pkg: str):
    """检查并自动安装缺失的 LangChain 包"""
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        print(f"Installing {pkg_kebab}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg_kebab])
```

---

## 5.3 create_chat_completion() — 统一调用入口

**源码位置**：`gpt_researcher/utils/llm.py`（被 actions 层广泛调用）

这是整个系统最核心的 LLM 调用函数，所有 LLM 交互都通过它：

```python
async def create_chat_completion(
    model,                     # 模型名 (如 "gpt-4o")
    messages,                  # 消息列表
    llm_provider=None,         # 提供商名 (如 "openai")
    temperature=0.35,          # 温度
    max_tokens=None,           # 最大token
    stream=False,              # 是否流式
    websocket=None,            # WebSocket
    llm_kwargs=None,           # 额外参数
    cost_callback=None,        # 费用回调
    reasoning_effort=None,     # 推理力度 (用于 o3/o4 模型)
    **kwargs
):
    # 1. 处理特殊模型的温度设置
    if model in NO_SUPPORT_TEMPERATURE_MODELS:
        temperature = None  # o1/o3/gpt-5 不支持自定义温度
    
    # 2. 处理推理力度
    if model in SUPPORT_REASONING_EFFORT_MODELS and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    
    # 3. 创建 LLM 提供商实例
    provider = GenericLLMProvider.from_provider(
        llm_provider, model=model, temperature=temperature, 
        max_tokens=max_tokens, **kwargs
    )
    
    # 4. 调用并获取响应
    response = await provider.get_chat_response(messages, stream, websocket)
    
    # 5. 费用追踪
    if cost_callback:
        cost = estimate_cost(model, messages, response)
        cost_callback(cost)
    
    return response
```

---

## 5.4 三级 LLM 调用场景

### Fast LLM（快速，低成本）
```python
# 用于摘要等简单任务
await create_chat_completion(
    model=cfg.fast_llm_model,        # gpt-4o-mini
    llm_provider=cfg.fast_llm_provider,
    temperature=0.25,
    ...
)
```

### Smart LLM（智能，高质量）
```python
# 用于报告生成、Agent选择
await create_chat_completion(
    model=cfg.smart_llm_model,       # gpt-4o
    llm_provider=cfg.smart_llm_provider,
    stream=True,                      # 通常流式输出
    websocket=websocket,              # 推送到前端
    ...
)
```

### Strategic LLM（战略级，用于推理）
```python
# 用于子查询规划
await create_chat_completion(
    model=cfg.strategic_llm_model,   # o4-mini
    llm_provider=cfg.strategic_llm_provider,
    reasoning_effort=ReasoningEfforts.Medium.value,  # 推理力度控制
    max_tokens=None,                  # 推理模型通常不限制
    ...
)
```

### 推理力度 (Reasoning Effort)

```python
class ReasoningEfforts(Enum):
    High = "high"      # 深度推理（Deep Research 规划）
    Medium = "medium"  # 中等推理（子查询生成）
    Low = "low"        # 轻量推理
```

仅支持 o3-mini、o3、o4-mini 等推理模型。

---

## 5.5 Agent 自动选择 — 类 Function Call 模式

**源码位置**：`gpt_researcher/actions/agent_creator.py`

### choose_agent() — 自动匹配专业 Agent

```python
async def choose_agent(query, cfg, parent_query=None, cost_callback=None, ...):
    """根据查询自动选择最合适的 Agent 角色"""
    
    response = await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": prompt_family.auto_agent_instructions()},
            {"role": "user", "content": f"task: {query}"},
        ],
        temperature=0.15,  # 低温度确保稳定输出
        ...
    )
    
    # 解析 JSON 响应
    agent_dict = json.loads(response)
    return agent_dict["server"], agent_dict["agent_role_prompt"]
```

### auto_agent_instructions() Prompt

```python
# 输入: 用户查询
# 输出: {"server": "Agent名称", "agent_role_prompt": "角色提示词"}

# 示例:
# 查询: "should I invest in apple stocks?"
# 返回: {
#     "server": "💰 Finance Agent",
#     "agent_role_prompt": "You are a seasoned finance analyst..."
# }

# 查询: "what are the most interesting sites in Tel Aviv?"
# 返回: {
#     "server": "🌍 Travel Agent",
#     "agent_role_prompt": "You are a world-travelled AI tour guide..."
# }
```

### 容错处理链

```python
try:
    # 尝试1: 直接 JSON 解析
    agent_dict = json.loads(response)
except:
    try:
        # 尝试2: json_repair 修复
        agent_dict = json_repair.loads(response)
    except:
        # 尝试3: 正则提取
        json_string = extract_json_with_regex(response)
        if json_string:
            json_data = json.loads(json_string)
        else:
            # 最终回退: 使用默认Agent
            return "Default Agent", "You are an AI critical thinker..."
```

---

## 5.6 MCP 工具选择 — 真正的 Function Call

**源码位置**：`gpt_researcher/mcp/tool_selector.py`

MCP 工具选择是最接近传统 Function Call 的部分：

### 两阶段工具调用

```
阶段1: 工具选择 (Tool Selection)
┌───────────────────────────────┐
│ 输入: 查询 + 可用工具列表      │
│ LLM 分析: 哪些工具最相关      │
│ 输出: selected_tools (top 3)  │
└───────────────────────────────┘
         ↓
阶段2: 工具执行 (Tool Execution)
┌───────────────────────────────┐
│ 输入: 查询 + 选中工具          │
│ MCP Client: 调用工具           │
│ 输出: 研究结果                 │
└───────────────────────────────┘
```

### 工具选择 Prompt

```python
# prompts.py — generate_mcp_tool_selection_prompt()
def generate_mcp_tool_selection_prompt(query, tools_info, max_tools=3):
    return f"""You are a research assistant helping to select the most relevant tools.

RESEARCH QUERY: "{query}"
AVAILABLE TOOLS: {json.dumps(tools_info, indent=2)}

SELECTION CRITERIA:
- Choose tools that can provide information related to the query
- Prioritize tools that can search, retrieve, or access relevant content
- Consider tools that complement each other

Return JSON:
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "Why this tool is relevant"
    }}
  ],
  "selection_reasoning": "Overall strategy"
}}
"""
```

### 工具执行 Prompt

```python
# prompts.py — generate_mcp_research_prompt()
def generate_mcp_research_prompt(query, selected_tools):
    return f"""You are a research assistant with access to specialized tools.

RESEARCH QUERY: "{query}"
AVAILABLE TOOLS: {tool_names}

INSTRUCTIONS:
1. Use the available tools to gather relevant information
2. Call multiple tools if needed
3. If a tool call fails, try alternative approaches
4. Synthesize information from multiple sources
"""
```

---

## 5.7 报告生成 — LLM 核心调用

**源码位置**：`gpt_researcher/actions/report_generation.py`

### generate_report() — 最终报告生成

```python
async def generate_report(
    query, context, agent_role_prompt, report_type, tone,
    report_source, websocket, cfg, available_images=None, ...
):
    # 1. 选择对应报告类型的 Prompt
    generate_prompt = get_prompt_by_report_type(report_type, prompt_family)
    
    # 2. 构建内容
    if report_type == "subtopic_report":
        content = generate_prompt(query, existing_headers, relevant_written_contents, 
                                   main_topic, context, ...)
    elif custom_prompt:
        content = f"{custom_prompt}\n\nContext: {context}"
    else:
        content = generate_prompt(query, context, report_source, ...)
    
    # 3. 可选: 附加预生成图片
    if available_images:
        content += f"""
AVAILABLE IMAGES:
{images_info}
Place each image on its own line after the relevant section header.
"""
    
    # 4. LLM 调用（流式输出）
    report = await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": agent_role_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.35,
        stream=True,             # 流式推送到前端
        websocket=websocket,
        max_tokens=cfg.smart_token_limit,
        ...
    )
    
    return report
```

### Prompt 选择策略

```python
def get_prompt_by_report_type(report_type, prompt_family):
    """根据报告类型选择对应的 Prompt 生成器"""
    match report_type:
        case "research_report":   return prompt_family.generate_report_prompt
        case "resource_report":   return prompt_family.generate_resource_report_prompt
        case "outline_report":    return prompt_family.generate_outline_report_prompt
        case "custom_report":     return prompt_family.generate_custom_report_prompt
        case "subtopic_report":   return prompt_family.generate_subtopic_report_prompt
        case "deep_research":     return prompt_family.generate_deep_research_prompt
```

---

## 5.8 PromptFamily — 提示词工程体系

**源码位置**：`gpt_researcher/prompts.py`

### Prompt 体系结构

```python
class PromptFamily:
    """基础提示词族"""
    def __init__(self, config: Config):
        self.cfg = config
    
    # ═══ 报告类 Prompts ═══
    @staticmethod
    def generate_report_prompt(question, context, report_source, ...)
    @staticmethod
    def generate_resource_report_prompt(...)
    @staticmethod
    def generate_outline_report_prompt(...)
    @staticmethod
    def generate_deep_research_prompt(...)
    @staticmethod
    def generate_subtopic_report_prompt(...)
    @staticmethod
    def generate_custom_report_prompt(...)
    
    # ═══ 研究类 Prompts ═══
    @staticmethod
    def generate_search_queries_prompt(question, parent_query, report_type, ...)
    @staticmethod
    def generate_summary_prompt(query, data)
    @staticmethod
    def auto_agent_instructions()
    @staticmethod
    def curate_sources(query, sources, max_results)
    
    # ═══ MCP 类 Prompts ═══
    @staticmethod
    def generate_mcp_tool_selection_prompt(query, tools_info, max_tools)
    @staticmethod
    def generate_mcp_research_prompt(query, selected_tools)
    
    # ═══ 图片类 Prompts ═══
    @staticmethod
    def generate_image_analysis_prompt(query, sections, max_images)
    
    # ═══ 格式化方法 ═══
    @staticmethod
    def pretty_print_docs(docs, top_n)
    @staticmethod
    def join_local_web_documents(docs_context, web_context)

# ═══ 模型特化子类 ═══
class GranitePromptFamily(PromptFamily):
    """IBM Granite 模型特化"""

class Granite3PromptFamily(PromptFamily):
    """Granite 3.x 特化 — 使用特殊标记格式"""
    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"
```

### 核心报告 Prompt 结构

```python
# generate_report_prompt() 的关键指令：
"""
Information: "{context}"
---
Using the above information, answer the following query: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative,
in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.

Please follow all of the following guidelines:
- You MUST determine your own concrete and valid opinion based on the given information.
- You MUST write the report with markdown syntax and {report_format} format.
- You MUST prioritize the relevance, reliability, and significance of the sources.
- Use in-text citation references in {report_format} format.
- {reference_prompt}  # 引用格式要求
- {tone_prompt}        # 语气要求
You MUST write the report in the following language: {language}.
"""
```

---

## 5.9 费用追踪机制

```python
class GPTResearcher:
    def add_costs(self, cost: float):
        """记录API费用，按步骤分类"""
        self.research_costs += cost          # 总费用
        step = self._current_step            # 当前步骤
        self.step_costs[step] = self.step_costs.get(step, 0.0) + cost
    
    def get_step_costs(self) -> dict:
        """获取分步骤费用明细"""
        return dict(self.step_costs)
        # 示例: {"agent_selection": 0.01, "research": 0.15, "report_writing": 0.08}
```

---

## 📌 本章要点回顾

- [x] `GenericLLMProvider` 通过工厂模式支持 30+ LLM 提供商
- [x] `create_chat_completion()` 是统一 LLM 调用入口
- [x] 三级 LLM（fast/smart/strategic）对应不同任务复杂度
- [x] Agent 选择（`choose_agent`）是类 Function Call 的 JSON 结构化输出
- [x] MCP 工具选择是真正的 Function Call 模式（两阶段选择+执行）
- [x] `PromptFamily` 支持继承覆写，可按模型定制提示词
- [x] 费用追踪贯穿整个流程，按步骤分类记录
