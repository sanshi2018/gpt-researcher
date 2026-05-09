# Phase 3: RAG 检索增强生成

> 本章目标：深入理解 GPT-Researcher 的 RAG 实现，包括多检索器架构、MCP 集成、检索策略优化。

---

## 3.1 RAG 架构总览

GPT-Researcher 的 RAG 不是传统的"检索→拼接→生成"，而是一个 **多阶段、多检索器** 的复杂管道：

```
┌──────────────── RAG 管道 ────────────────┐
│                                           │
│  Stage 1: 查询规划                        │
│  ┌─────────────────────────────────────┐  │
│  │ plan_research()                     │  │
│  │ ├── 初始搜索获取上下文              │  │
│  │ └── LLM 生成 N 个子查询             │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Stage 2: 并行检索                        │
│  ┌─────────────────────────────────────┐  │
│  │ _process_sub_query() × N           │  │
│  │ ├── 多 Retriever 并行搜索           │  │
│  │ ├── URL 去重 (visited_urls)         │  │
│  │ ├── 网页抓取 (BrowserManager)       │  │
│  │ └── MCP 检索 (可选)                 │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Stage 3: 上下文压缩                      │
│  ┌─────────────────────────────────────┐  │
│  │ ContextCompressor                   │  │
│  │ ├── RecursiveCharacterTextSplitter  │  │
│  │ ├── EmbeddingsFilter               │  │
│  │ └── ContextualCompressionRetriever  │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Stage 4: 来源筛选 (可选)                 │
│  ┌─────────────────────────────────────┐  │
│  │ SourceCurator                       │  │
│  │ └── LLM 评估来源质量               │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Stage 5: 报告生成                        │
│  ┌─────────────────────────────────────┐  │
│  │ generate_report()                   │  │
│  │ └── 压缩后的上下文 → LLM → 报告     │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
```

---

## 3.2 检索器工厂 — 16 种检索器

**源码位置**：`gpt_researcher/actions/retriever.py`

### 工厂模式实现

```python
def get_retriever(retriever: str):
    """通过名称获取检索器类"""
    match retriever:
        case "tavily":     return TavilySearch
        case "google":     return GoogleSearch
        case "bing":       return BingSearch
        case "duckduckgo": return Duckduckgo
        case "arxiv":      return ArxivSearch
        case "mcp":        return MCPRetriever
        case "serper":     return SerperSearch
        case "serpapi":     return SerpApiSearch
        case "exa":        return ExaSearch
        case "semantic_scholar": return SemanticScholarSearch
        case "pubmed_central":   return PubMedCentralSearch
        case "custom":     return CustomRetriever
        case "xquik":      return XquikSearch
        case _:            return None
```

### 检索器分类

| 类别 | 检索器 | 特点 |
|------|--------|------|
| **通用搜索** | tavily, google, bing, duckduckgo, serper, serpapi | 返回URL，需二次抓取 |
| **学术搜索** | arxiv, semantic_scholar, pubmed_central | 可能包含全文 |
| **聚合搜索** | searchapi, searx, exa | 支持多源聚合 |
| **社交搜索** | xquik | X/Twitter 搜索 |
| **协议检索** | mcp | MCP (Model Context Protocol) |
| **自定义** | custom | 用户自定义 |

### 多检索器配置

```bash
# 单一检索器
RETRIEVER=tavily

# 多检索器（逗号分隔）
RETRIEVER=tavily,google,mcp

# 支持从 headers 动态设置
headers = {"retrievers": "tavily,arxiv"}
```

### get_retrievers() — 检索器解析链

```python
def get_retrievers(headers, cfg):
    """解析优先级: headers > config > default(tavily)"""
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    elif cfg.retrievers:
        retrievers = cfg.retrievers if isinstance(cfg.retrievers, list) else cfg.retrievers.split(",")
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    else:
        retrievers = [get_default_retriever().__name__]  # Tavily
    
    return [get_retriever(r) or get_default_retriever() for r in retrievers]
```

---

## 3.3 查询规划 — 子查询生成

**源码位置**：`gpt_researcher/actions/query_processing.py`

### plan_research_outline()

研究的第一步是将用户查询分解为多个子查询：

```python
async def plan_research_outline(query, search_results, agent_role_prompt, cfg, ...):
    # 特殊处理: MCP-only 模式跳过子查询生成
    if mcp_only:
        return [query]  # 直接返回原始查询
    
    # 生成子查询
    sub_queries = await generate_sub_queries(
        query, parent_query, report_type, search_results, cfg, ...
    )
    return sub_queries
```

### generate_sub_queries()

```python
async def generate_sub_queries(query, parent_query, report_type, context, cfg, ...):
    """使用 Strategic LLM 生成子查询"""
    gen_queries_prompt = prompt_family.generate_search_queries_prompt(
        query, parent_query, report_type,
        max_iterations=cfg.max_iterations or 3,
        context=context,  # 初始搜索结果作为上下文
    )
    
    response = await create_chat_completion(
        model=cfg.strategic_llm_model,        # 使用 Strategic LLM
        messages=[{"role": "user", "content": gen_queries_prompt}],
        llm_provider=cfg.strategic_llm_provider,
        reasoning_effort=ReasoningEfforts.Medium.value,  # 中等推理力度
        ...
    )
    
    return json_repair.loads(response)  # 返回查询列表
```

### 子查询生成提示词

```python
# prompts.py 中的核心 Prompt
def generate_search_queries_prompt(question, parent_query, report_type, max_iterations, context):
    return f"""Write {max_iterations} google search queries to search online that form 
    an objective opinion from the following task: "{task}"
    
    {context_prompt}  # 包含初始搜索结果
    
    You must respond with a list of strings in the following format: 
    [{dynamic_example}].
    """
```

---

## 3.4 ResearchConductor — 研究编排器

**源码位置**：`gpt_researcher/skills/researcher.py`

### conduct_research() 主流程

```python
async def conduct_research(self):
    """根据 report_source 选择不同的研究路径"""
    
    if self.researcher.source_urls:
        # 路径A: 用户提供了指定URL
        research_data = await self._get_context_by_urls(self.researcher.source_urls)
        
    elif self.researcher.report_source == ReportSource.Web.value:
        # 路径B: 网络搜索（最常用）
        research_data = await self._get_context_by_web_search(
            self.researcher.query, [], self.researcher.query_domains
        )
        
    elif self.researcher.report_source == ReportSource.Local.value:
        # 路径C: 本地文档
        document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
        research_data = await self._get_context_by_web_search(
            self.researcher.query, document_data, ...
        )
        
    elif self.researcher.report_source == ReportSource.Hybrid.value:
        # 路径D: 混合搜索 = 本地文档 + 网络
        docs_context = await self._get_context_by_web_search(query, document_data)
        web_context = await self._get_context_by_web_search(query, [])
        research_data = prompt_family.join_local_web_documents(docs_context, web_context)
    
    elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
        # 路径E: 直接从向量库检索
        research_data = await self._get_context_by_vectorstore(query, filter)
    
    # 可选: 来源筛选
    if self.researcher.cfg.curate_sources:
        research_data = await self.researcher.source_curator.curate_sources(research_data)
    
    return research_data
```

### _get_context_by_web_search() — 网络搜索核心

```python
async def _get_context_by_web_search(self, query, scraped_data=None, query_domains=None):
    # 1. MCP 预执行（根据策略）
    if mcp_retrievers and mcp_strategy == "fast":
        mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
        self._mcp_results_cache = mcp_context  # 缓存，避免重复调用
    
    # 2. 生成子查询
    sub_queries = await self.plan_research(query, query_domains)
    if self.researcher.report_type != "subtopic_report":
        sub_queries.append(query)  # 原始查询也加入
    
    # 3. 并行处理所有子查询 ⚡
    context = await asyncio.gather(*[
        self._process_sub_query(sub_query, scraped_data, query_domains)
        for sub_query in sub_queries
    ])
    
    # 4. 过滤空结果并合并
    context = [c for c in context if c]
    return " ".join(context) if context else []
```

### _process_sub_query() — 单个子查询处理

```python
async def _process_sub_query(self, sub_query, scraped_data=[], query_domains=[]):
    """每个子查询的完整处理流程"""
    
    # 1. MCP 上下文获取（根据策略选择缓存或实时查询）
    if mcp_strategy == "fast" and self._mcp_results_cache:
        mcp_context = self._mcp_results_cache.copy()    # 复用缓存
    elif mcp_strategy == "deep":
        mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
    
    # 2. 网络搜索 + 抓取
    if not scraped_data:
        scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
    
    # 3. 上下文压缩（Embedding 相似度过滤）
    if scraped_data:
        web_context = await self.researcher.context_manager.get_similar_content_by_query(
            sub_query, scraped_data
        )
    
    # 4. 智能合并 MCP + 网络上下文
    combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
    
    return combined_context
```

### _scrape_data_by_urls() — 搜索+抓取

```python
async def _scrape_data_by_urls(self, sub_query, query_domains=None):
    # 1. 多检索器并行搜索
    new_search_urls, prefetched_content = await self._search_relevant_source_urls(
        sub_query, query_domains
    )
    
    # 2. 抓取需要二次获取的URL
    scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
    
    # 3. 合并预取内容（如PubMed已有全文的）
    scraped_content.extend(prefetched_content)
    
    # 4. 可选：入向量库
    if self.researcher.vector_store:
        self.researcher.vector_store.load(scraped_content)
    
    return scraped_content
```

---

## 3.5 MCP 集成 — Function Call 风格检索

**源码位置**：`gpt_researcher/mcp/`

### MCP 概念

MCP (Model Context Protocol) 允许 GPT-Researcher 连接外部数据源（GitHub、数据库等），类似 Function Call 的检索方式。

### MCP 策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `fast` | 仅对原始查询运行一次MCP，结果缓存复用 | 默认，性能优先 |
| `deep` | 对每个子查询都运行MCP | 需要深度MCP数据 |
| `disabled` | 完全禁用MCP | 仅使用Web检索 |

### MCP 两阶段执行

```
Stage 1: 工具选择
  ├── 获取所有可用 MCP 工具
  ├── LLM 分析查询匹配度
  └── 选择最佳工具 (max_tools=3)

Stage 2: 研究执行
  ├── 使用选中工具调用 MCP 服务器
  ├── 收集研究结果
  └── 格式化为统一上下文
```

### MCP 结果与 Web 结果合并

```python
def _combine_mcp_and_web_context(self, mcp_context, web_context, sub_query):
    combined_parts = []
    
    # Web 上下文优先
    if web_context and web_context.strip():
        combined_parts.append(web_context.strip())
    
    # MCP 上下文附加（带来源标注）
    if mcp_context:
        for item in mcp_context:
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            formatted = f"{content}\n\n*Source: {title} ({url})*"
            mcp_formatted.append(formatted)
        
        combined_parts.append("\n\n---\n\n".join(mcp_formatted))
    
    return "\n\n".join(combined_parts)
```

---

## 3.6 检索结果预取优化

项目对某些检索器（如 PubMed Central）做了智能优化，避免不必要的二次抓取：

```python
# researcher.py — _search_relevant_source_urls()
for result in search_results:
    url = result.get("href") or result.get("url")
    raw_content = result.get("raw_content") or result.get("body")
    
    if url and raw_content and len(raw_content) > 100:
        # 检索器已提供全文 → 直接使用，跳过抓取
        prefetched_content.append({
            "url": url,
            "raw_content": raw_content,
        })
    elif url:
        # 只有URL → 需要二次抓取
        new_search_urls.append(url)
```

---

## 3.7 URL 去重机制

```python
# visited_urls 是一个 set，贯穿整个研究过程
async def _get_new_urls(self, url_set_input):
    new_urls = []
    for url in url_set_input:
        if url not in self.researcher.visited_urls:
            self.researcher.visited_urls.add(url)
            new_urls.append(url)
    return new_urls
```

---

## 📌 本章要点回顾

- [x] RAG 管道包含 5 个阶段：查询规划 → 并行检索 → 上下文压缩 → 来源筛选 → 报告生成
- [x] 支持 16 种检索器，通过工厂模式统一管理
- [x] 子查询由 Strategic LLM 生成，原始查询也会被加入研究
- [x] `asyncio.gather()` 实现子查询并行处理
- [x] MCP 检索支持 fast/deep/disabled 三种策略
- [x] 智能预取优化避免已有全文的来源被重复抓取
- [x] visited_urls 全局去重，在 Deep Research 中跨 Researcher 共享
