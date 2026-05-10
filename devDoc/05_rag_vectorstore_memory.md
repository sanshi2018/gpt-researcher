# 05. RAG 全链路：Embedding 工厂、向量存储、上下文压缩

## 模块概述

03、04 篇分别讲了"问什么"（query 改写）和"找什么"（retriever / scraper）。这一篇关心"**怎么从一堆原始网页里精确抽出能塞进 prompt 的几千字**"——即整套 RAG 的**召回-过滤-压缩**流程。

具体由三件东西组成：

| 组件 | 文件 | 职责 |
|---|---|---|
| `Memory.Embeddings` 工厂 | `memory/embeddings.py` | 用 `match/case` 统一封装 **20 个 embedding provider**（OpenAI / Cohere / Voyage / Bedrock / Ollama / Nomic / 自建 OpenAI 兼容 / …） |
| `VectorStoreWrapper` | `vector_store/vector_store.py` | 把 LangChain 任意 VectorStore（Chroma / FAISS / Pinecone / Qdrant / …）加一层"GPT-Researcher Document → LangChain Document → 切块 → 入库"的薄封装 |
| **三胞胎 Compressor** | `context/compression.py` + `context/retriever.py` | `ContextCompressor`（抓回的网页）/ `VectorstoreCompressor`（用户传的向量库）/ `WrittenContentCompressor`（已写章节去重） |

它们的核心都依赖 LangChain 的两个原语：
- `RecursiveCharacterTextSplitter`（递归分块，1000 字 / 100 字 overlap）
- `EmbeddingsFilter`（embed → cosine 相似度阈值过滤）

并通过两个环境变量留出了"成本-质量"调节杆：
- `SIMILARITY_THRESHOLD`（默认 0.35，越高越严）
- `COMPRESSION_THRESHOLD`（默认 8000 字符，小于此走快路径，跳过 embed）

---

## 架构 / 流程图

### RAG 全链路（一个子查询的视角）

```mermaid
flowchart TB
    SQ[sub_query]

    subgraph Retrieve["检索（→ 04 篇）"]
        R[Retrievers] --> URLs
        URLs --> Scrap[Scrapers]
        Scrap --> Pages["pages = [{url, raw_content, title, image_urls}]"]
    end

    SQ --> Retrieve
    Pages --> CM[ContextManager.<br/>get_similar_content_by_query]
    SQ --> CM

    subgraph Compress["压缩"]
        CM --> CC[ContextCompressor.<br/>async_get_context]
        CC --> Fast{total_chars<br/>< 8000<br/>且 n ≤ 5?}
        Fast -- yes 快路径 --> Direct[Document(page_content=raw_content,<br/>metadata=full)]
        Fast -- no  慢路径 --> Build["构建 ContextualCompressionRetriever<br/>= TextSplitter + EmbeddingsFilter + SearchAPIRetriever"]
        Build --> Embed[每段 embed → 与 query 算 cosine<br/>留 score >= SIMILARITY_THRESHOLD]
        Direct --> Pretty[prompt_family.pretty_print_docs]
        Embed --> Pretty
    end

    Pretty --> Ctx["compressed_context (str)"]
    Ctx --> RG[ReportGenerator (→ 02 篇)]

    subgraph Optional["可选：用户传 LangChain VectorStore"]
        VS[(VectorStore)] --> VSW[VectorStoreWrapper]
        VSW --> VC[VectorstoreCompressor]
        VC -- pretty_print_docs --> Ctx
    end
```

### Embeddings 工厂分发

```
Config.embedding (e.g. "openai:text-embedding-3-small")
   │
   ▼
Config.parse_embedding → (provider="openai", model="text-embedding-3-small")
   │
   ▼
Memory(provider, model, **embedding_kwargs)
   │
   ▼ match provider:
      ┌─ openai          → langchain_openai.OpenAIEmbeddings
      ├─ azure_openai    → langchain_openai.AzureOpenAIEmbeddings
      ├─ cohere          → langchain_cohere.CohereEmbeddings
      ├─ google_vertexai → langchain_google_vertexai.VertexAIEmbeddings
      ├─ google_genai    → langchain_google_genai.GoogleGenerativeAIEmbeddings
      ├─ fireworks       → langchain_fireworks.FireworksEmbeddings
      ├─ gigachat        → langchain_gigachat.GigaChatEmbeddings
      ├─ ollama          → langchain_ollama.OllamaEmbeddings (env OLLAMA_BASE_URL)
      ├─ together        → langchain_together.TogetherEmbeddings
      ├─ netmind         → langchain_netmind.NetmindEmbeddings
      ├─ mistralai       → langchain_mistralai.MistralAIEmbeddings
      ├─ huggingface     → langchain_huggingface.HuggingFaceEmbeddings (本地)
      ├─ nomic           → langchain_nomic.NomicEmbeddings
      ├─ voyageai        → langchain_voyageai.VoyageAIEmbeddings
      ├─ dashscope       → langchain_community DashScopeEmbeddings
      ├─ bedrock         → langchain_aws.embeddings.BedrockEmbeddings
      ├─ aimlapi         → OpenAIEmbeddings(base_url=AIMLAPI_BASE_URL)
      ├─ openrouter      → OpenAIEmbeddings(base_url=https://openrouter.ai/api/v1)
      ├─ minimax         → OpenAIEmbeddings(base_url=https://api.minimax.io/v1)
      └─ custom          → OpenAIEmbeddings(base_url=OPENAI_BASE_URL || lmstudio default)
```

### "三胞胎" Compressor 角色对比

```
┌───────────────────────┬──────────────────────┬──────────────────────────┐
│   ContextCompressor   │ VectorstoreCompressor │  WrittenContentCompressor │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 输入：刚抓的网页 list  │ 输入：用户已经构建好  │ 输入：已写章节 list        │
│      [{url,raw_content,│      的 LangChain    │      [{section_title,    │
│        title, ...}]    │      VectorStore     │        written_content}] │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 内部 retriever：       │ 直接走 vector_store. │ SectionRetriever         │
│ SearchAPIRetriever    │ asimilarity_search   │                          │
│ (→ Document chunks)    │                      │                          │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 切块 + EmbeddingsFilter│ 不切块，靠 vs 自带ANN │ 切块 + EmbeddingsFilter   │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 有 fast path（<8000字）│ 无 fast path         │ 无 fast path             │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 输出：拼好的 str       │ 输出：拼好的 str     │ 输出：list[str]          │
├───────────────────────┼──────────────────────┼──────────────────────────┤
│ 给：每个子查询的 web  │ 给：用户自带知识库   │ 给：detailed_report 子主题 │
│      上下文            │      RAG             │      之间的去重           │
└───────────────────────┴──────────────────────┴──────────────────────────┘
```

---

## 核心源码解析

### 1) `Memory`：embedding 单一入口

`gpt_researcher/memory/embeddings.py`

```python
_SUPPORTED_PROVIDERS = {
    "openai", "azure_openai", "cohere", "gigachat",
    "google_vertexai", "google_genai", "fireworks",
    "ollama", "together", "mistralai", "huggingface",
    "nomic", "voyageai", "dashscope", "custom",
    "bedrock", "aimlapi", "netmind", "openrouter", "minimax",
}

class Memory:
    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs):
        _embeddings = None
        match embedding_provider:
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                if "openai_api_base" not in embedding_kwargs and os.environ.get("OPENAI_BASE_URL"):
                    embedding_kwargs["openai_api_base"] = os.environ["OPENAI_BASE_URL"]
                _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)

            case "ollama":
                from langchain_ollama import OllamaEmbeddings
                _embeddings = OllamaEmbeddings(
                    model=model,
                    base_url=os.environ["OLLAMA_BASE_URL"],
                    **embedding_kwargs,
                )

            case "custom":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                    openai_api_base=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),  # lmstudio default
                    check_embedding_ctx_length=False,           # ★ 关掉 OpenAI 客户端的 ctx_length 校验
                    **embedding_kwargs,
                )

            # ... 其它 16 个分支
            case _:
                raise Exception("Embedding not found.")
        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
```

**关键设计点**：

- **lazy import**：每个 provider 的依赖只在使用时才 `from langchain_xxx import ...`，跟 04 篇 retrievers 的策略完全一致——你只装你用到的。
- **OpenAI 兼容路径有 4 条**：`openai`（官方 + 可选 base url）、`custom`（lmstudio / vLLM 本地）、`aimlapi` / `openrouter` / `minimax`（OpenAI-compatible SaaS）。它们都用同一个 `OpenAIEmbeddings` 类，只换 base_url——**这是接入新 OpenAI-compatible 服务最快的复用通道**。
- **`check_embedding_ctx_length=False`**：lmstudio / vLLM 经常没实现 OpenAI 的 token 计数接口，这个开关绕过校验。

### 2) `VectorStoreWrapper`：薄到几乎不见的一层

`vector_store/vector_store.py`

```python
class VectorStoreWrapper:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def load(self, documents):
        """
        documents 是 GPT-Researcher 风格：[{url, raw_content, ...}, ...]
        →  转 LangChain Document → 切块 → add_documents
        """
        langchain_documents = self._create_langchain_documents(documents)
        splitted_documents = self._split_documents(langchain_documents)
        self.vector_store.add_documents(splitted_documents)

    def _create_langchain_documents(self, data):
        return [Document(page_content=item["raw_content"],
                         metadata={"source": item["url"]}) for item in data]

    def _split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    async def asimilarity_search(self, query, k, filter):
        return await self.vector_store.asimilarity_search(query=query, k=k, filter=filter)
```

> **设计意图**：业务代码全程只跟 `VectorStoreWrapper` 打交道，不直接知道是 Chroma 还是 Pinecone。这样替换实现完全无感——LangChain 的 `VectorStore` 抽象在这里做了真正的工作。

### 3) `ContextCompressor`：核心 RAG 引擎

`context/compression.py:85-178`

```python
class ContextCompressor:
    def __init__(self, documents, embeddings, max_results=5, prompt_family=PromptFamily, **kwargs):
        self.max_results = max_results
        self.documents = documents                        # [{url, raw_content, title}, ...]
        self.embeddings = embeddings                      # langchain Embeddings 实例
        self.similarity_threshold = os.environ.get("SIMILARITY_THRESHOLD", 0.35)
        self.prompt_family = prompt_family
```

> ⚠️ 注意 `os.environ.get("SIMILARITY_THRESHOLD", 0.35)` 没做 `float()` 转换——env 进来是字符串！这是 bug 还是特性？看下游 `EmbeddingsFilter` 是否会自动转。一般 LangChain 接受 float，所以 env 设这个值时**务必显式 cast 或在 prompt-side 测一下**。安全做法：手动 `float(os.environ["SIMILARITY_THRESHOLD"])`。

#### 慢路径：组装 ContextualCompressionRetriever

```python
def __get_contextual_retriever(self):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    relevance_filter = EmbeddingsFilter(
        embeddings=self.embeddings,
        similarity_threshold=self.similarity_threshold,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]      # ← 顺序执行：先切再筛
    )
    base_retriever = SearchAPIRetriever(pages=self.documents)   # ← 自定义 retriever
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )
```

`SearchAPIRetriever`（`context/retriever.py:10`）就是个适配器：

```python
class SearchAPIRetriever(BaseRetriever):
    pages: List[Dict] = []
    def _get_relevant_documents(self, query, *, run_manager):
        # 不真做检索！直接把 pages 全部转 Document 返回，让上游 splitter+filter 处理
        return [
            Document(
                page_content=page.get("raw_content", ""),
                metadata={"title": page.get("title", ""),
                          "source": page.get("url", "")},
            )
            for page in self.pages
        ]
```

> 💡 这是个**易被误读**的细节：`SearchAPIRetriever._get_relevant_documents` 完全无视 `query` 参数，把所有 pages 当 Document 输出。**真正的"按 query 过滤"发生在 `EmbeddingsFilter` 这一步**——所以"召回"在 04 篇 retrievers 里、"过滤"才在这里。命名容易误导。

#### `async_get_context`：双路径决策

```python
async def async_get_context(self, query, max_results=5, cost_callback=None) -> str:
    total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
    chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

    # 快路径：内容已经很短，没必要再切再 embed
    if total_chars < chunk_threshold and len(self.documents) <= max_results:
        direct_docs = [
            Document(page_content=doc.get('raw_content', ''), metadata=doc)
            for doc in self.documents[:max_results]
        ]
        return self.prompt_family.pretty_print_docs(direct_docs, max_results)

    # 慢路径：完整压缩管线
    compressed_docs = self.__get_contextual_retriever()
    if cost_callback:
        cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL,
                                              docs=self.documents))
    relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
    return self.prompt_family.pretty_print_docs(relevant_docs, max_results)
```

**两条路径的实际差异**（实测大致量级）：

| 路径 | 触发条件 | 延迟 | embedding 成本 |
|---|---|---|---|
| 快路径 | `total_chars < 8000` 且 `len(docs) ≤ 5` | < 50 ms | 0 |
| 慢路径 | 不满足 | 1-3 s（取决于 embed provider） | 取决于文本量 |

`.env.example` 里专门标了这一点：
```
# Smart context compression: Skip expensive embedding-based filtering for small documents
# This can reduce latency by 40-50% for queries with small result sets
# Default: 8000 characters (8KB threshold)
#COMPRESSION_THRESHOLD=8000
```

### 4) `WrittenContentCompressor`：detailed_report 的"防重复"机制

```python
class WrittenContentCompressor:
    def __init__(self, documents, embeddings, similarity_threshold, **kwargs):
        # documents 形如 [{section_title, written_content}, ...]
        self.documents = documents
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def __get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold,
        )
        pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, relevance_filter])
        base_retriever = SectionRetriever(sections=self.documents)
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )

    async def async_get_context(self, query, max_results=5, cost_callback=None) -> list[str]:
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        return self.__pretty_docs_list(relevant_docs, max_results)

    def __pretty_docs_list(self, docs, top_n) -> list[str]:
        return [f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n"
                for i, d in enumerate(docs) if i < top_n]
```

它在 `ContextManager.get_similar_written_contents_by_draft_section_titles`（02 篇）里被调用：

```python
all_queries = [current_subtopic] + draft_section_titles    # 多 query 并发查
results = await asyncio.gather(*[process_query(q) for q in all_queries])
relevant_contents = set().union(*results)                  # 求并集去重
```

> **应用场景**：写"详尽报告"时，每个子主题章节都要避开"前面已经写过的内容"——`WrittenContentCompressor` 就是用 embedding 帮你找到"已经写过的相似段"，配合 prompt 里的"don't repeat"指令完成防重。

### 5) `VectorstoreCompressor`：当用户自带 RAG 时

```python
class VectorstoreCompressor:
    def __init__(self, vector_store: VectorStoreWrapper, max_results=7,
                 filter=None, prompt_family=PromptFamily, **kwargs):
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter

    async def async_get_context(self, query, max_results=5) -> str:
        results = await self.vector_store.asimilarity_search(
            query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)
```

只有 5 行有效代码——它假设用户传进来的 vector store **已经做过 chunking 和 embedding**，所以不需要再切再 embed，直接 ANN 检索。

---

## 技术原理深度解析

### A. 切块策略：1000 字 / 100 字 overlap 是怎么来的？

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

参数选择的工程经验：

| 参数 | 选 1000 字符的理由 | 备选 |
|---|---|---|
| `chunk_size=1000` | 大约 200-300 token，正好是 OpenAI / Cohere 大多数 embedding 模型的甜点（embedding 质量随 chunk 长度先升后降，~256 token 是峰值） | 500（更细但召回数翻倍）/ 2000（少调用但语义稀释） |
| `chunk_overlap=100` | 10% overlap 防止"概念正好被劈开"，常见做法 | 0 / 200 / 250 |

`RecursiveCharacterTextSplitter` 本身的"递归"含义：先按 `\n\n` 切，超长就再按 `\n` 切，再不行按 `. `，最后才硬切字符——保证语义边界尽可能完整。

> ⚠️ `VectorStoreWrapper._split_documents` 用的是 `chunk_overlap=200`（不是 100），与 `ContextCompressor` 的 100 不一致——这是历史遗留的不一致，影响有限但值得注意。

### B. `EmbeddingsFilter` 的工作机制

LangChain `EmbeddingsFilter` 大致逻辑：

```
1. 把 query embed 一次（缓存）
2. 对每个 chunk:
     emb = embedding.embed(chunk.page_content)
     score = cosine(emb, query_emb)
     if score >= similarity_threshold:
         keep
3. 按 score 倒排返回
```

成本可观——M 个网页 × 平均 K 个 chunk = M*K 次 embed 调用。这就是为什么有快路径优化、为什么 `cost_callback` 在调用前估算成本。

`SIMILARITY_THRESHOLD` 经验值：

| 阈值 | 行为 |
|---|---|
| 0.25-0.30 | 召回宽，一些噪声段也进来；适合"探索式"研究 |
| **0.35（默认）** | 平衡 |
| 0.42（DEFAULT_CONFIG.SIMILARITY_THRESHOLD） | 严，可能漏掉相关但表达不同的段 |
| > 0.50 | 太严，常返回空 list |

> ⚠️ 注意 `default.py` 里 `SIMILARITY_THRESHOLD = 0.42` 但 `compression.py` 里 fallback 是 `0.35`——它们走的不是同一条路径。`Config` 把 `SIMILARITY_THRESHOLD` 解析为 `cfg.similarity_threshold`，但 `ContextCompressor` 直接读 env，**没用 `cfg`**。这意味着：在 `default.py` 改这个值不会生效，**必须改 env**。

### C. cosine vs dot product vs L2

LangChain 的 `EmbeddingsFilter` 默认用 cosine——这与 `text-embedding-3-small` / `text-embedding-3-large` 等已 L2 归一化的模型天然兼容（cosine ≡ dot product）。如果你换 Voyage / Nomic / 自训模型，要确认它的输出是否归一化，否则相似度对比会失真。

### D. embedding 成本估算的"作弊"

```python
# costs.py:38
def estimate_embedding_cost(model, docs):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(str(doc))) for doc in docs)
    return total_tokens * EMBEDDING_COST       # 0.02 / 1000000 写死
```

它**只用 OpenAI 价格**：`$0.02 / 1M tokens`（对应 text-embedding-3-small）。
如果你用 Cohere / Voyage / 自部署 Ollama，这个数字毫无意义——只是个"提示性"指标。生产里要做精确成本核算需要自己写 `cost_callback`。

### E. 为什么 SearchAPIRetriever 不真做 retrieve？

LangChain `ContextualCompressionRetriever` 的设计是 **"base retriever 召回 → compressor 过滤"** 两段式。本项目的"召回"已经在 04 篇 scrapers 完成了，到这里 docs 已经在内存里——所以 base retriever 退化成"把内存里的 list 转 Document"。这是把 LangChain 的抽象**复用了一半**：只用其 splitter + filter 管线，不用其检索逻辑。

如果你完全用 LangChain VectorStore（`VectorstoreCompressor`），那条路径是真实 ANN 检索。

### F. `metadata` 在哪里被消费

```python
# compression.py:166 (快路径)
direct_docs = [
    Document(page_content=doc.get('raw_content', ''), metadata=doc)  # ← 整个 doc 都进 metadata
    for doc in self.documents[:max_results]
]
```

```python
# prompts.py:551
def pretty_print_docs(docs, top_n=None):
    return f"\n".join(
        f"Source: {d.metadata.get('source')}\n"
        f"Title: {d.metadata.get('title')}\n"
        f"Content: {d.page_content}\n"
        for i, d in enumerate(docs)
        if top_n is None or i < top_n
    )
```

下游 SMART_LLM 看到的 context 形如：
```
Source: https://example.com/article
Title: Introducing X
Content: <主体文本>

Source: ...
```

> **引用就是这样被注入的**——后续 `add_references` 才能把 URL 抽出来加到报告末尾（→ 02 篇）。

---

## 关键设计决策

| 决策 | 取舍 |
|---|---|
| **20 个 embedding provider 用 if/elif** | 与 retrievers / LLM provider 思路一致，lazy import 只装需要的；缺点是新增 provider 要改两处（`_SUPPORTED_PROVIDERS` + `match` 分支） |
| **三胞胎 Compressor 而非一个** | 输入形态不同（list / vector store / sections），合并会让接口更糟；现在每个类只做一件事 |
| **`SearchAPIRetriever` 退化成"列表→Document"适配器** | 复用 LangChain 的 splitter+filter，不做真召回——**召回与过滤被刻意解耦**，因为召回早在 retrievers/scrapers 层完成 |
| **快路径短路** | 小文档没必要 embed；用 `total_chars < 8000` 这种简单门槛代替更复杂的判断 |
| **`SIMILARITY_THRESHOLD` 走 env 而非 cfg** | 这是个易踩的坑（与 `default.py` 不同步）；正确做法是在 `Config` 里统一读 |
| **`chunk_size=1000 / overlap=100`（compression）vs `1000 / 200`（vector_store）** | 历史遗留不一致；建议未来统一 |
| **`metadata=doc` 把整个 dict 塞进 metadata** | 下游 prompt 只取 `source` 和 `title`，多余字段不影响；但意味着如果 doc 里有大字段，会跟着 Document 走 |
| **embedding 成本写死 OpenAI 价** | 只是提示，不准确——但对 OpenAI 主流用户够用 |

替代方案讨论：

- **Reranking 没做**：项目当前只用 cosine 阈值过滤，没有 cross-encoder rerank。要加 Cohere Rerank / BGE-Reranker，自然的扩展点是在 `__get_contextual_retriever` 的 `transformers=[splitter, relevance_filter]` 后追加一个 `CrossEncoderReranker`。
- **没有 hybrid search**：纯 dense embedding，没有 BM25 + dense 融合。如果你的 corpus 里有大量"专有名词 / 代号"（容易被 dense embedding 拉远），最好接一层 sparse。
- **没有 HyDE / multi-query expansion**：sub-query 在 03 篇 generate_sub_queries 里做了，但每个 sub-query 内部就直接 embed 了，没有再扩展。

---

## 与其他模块的关联

```
本模块输入：
  ├─ Config（→ 01）：embedding_provider / embedding_model / embedding_kwargs
  ├─ Pages（→ 04）：scraper 给出的 [{url, raw_content, title, image_urls}]
  └─ env：SIMILARITY_THRESHOLD / COMPRESSION_THRESHOLD / OPENAI_BASE_URL ...

本模块输出：
  ├─ ContextCompressor.async_get_context → str (压缩后的 context)
  ├─ WrittenContentCompressor.async_get_context → list[str] (相似已写章节)
  └─ VectorstoreCompressor.async_get_context → str

被使用方：
  ├─ skills/context_manager.py → 02 篇所有 _process_sub_query 的核心
  ├─ skills/researcher.py → conduct_research 各分支
  └─ vector_store/VectorStoreWrapper 被 GPTResearcher.__init__ 使用
```

---

## 实操教程

### 例 1：换用 Cohere Embedding + 自定义 similarity 阈值

```bash
export EMBEDDING=cohere:embed-english-v3.0
export COHERE_API_KEY=...
export SIMILARITY_THRESHOLD=0.30   # 召回宽一点
export COMPRESSION_THRESHOLD=4000  # 更激进地走快路径
python -c "
import asyncio
from dotenv import load_dotenv; load_dotenv()
from gpt_researcher import GPTResearcher

async def main():
    r = GPTResearcher(query='vector database benchmarks 2025', verbose=True)
    await r.conduct_research()
    print('Total $:', r.get_costs())
asyncio.run(main())
"
```

### 例 2：用本地 Ollama 做 embedding（无 API 调用）

```bash
# 先起 Ollama 并 pull 一个嵌入模型
ollama pull nomic-embed-text

export EMBEDDING=ollama:nomic-embed-text
export OLLAMA_BASE_URL=http://localhost:11434

python scripts/min_research.py    # 沿用 01 篇示例
```

### 例 3：注入 LangChain VectorStore 当作 RAG 知识库

```python
# scripts/byo_chroma.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from gpt_researcher import GPTResearcher

# 1. 准备本地 Chroma（演示用 in-memory）
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs  = Chroma(embedding_function=emb)
vs.add_documents([
    Document(page_content="LangGraph supports persistent checkpoints via SqliteSaver and PostgresSaver.",
             metadata={"source": "internal-doc-1"}),
    Document(page_content="LangGraph uses StateGraph as the core data type, parameterized by a TypedDict.",
             metadata={"source": "internal-doc-2"}),
])

async def main():
    r = GPTResearcher(
        query="How does LangGraph persist state?",
        report_type="research_report",
        report_source="langchain_vectorstore",   # ← 关键
        vector_store=vs,                          # ← 注入
        verbose=True,
    )
    await r.conduct_research()
    print(await r.write_report())

asyncio.run(main())
```

> 走 `report_source="langchain_vectorstore"` 时，`ResearchConductor._get_context_by_vectorstore` 会用 `VectorstoreCompressor` 直接 ANN 查询——完全跳过 web retriever / scraper。

### 例 4：直接调用 ContextCompressor 做"问答"（最小 RAG 演示）

```python
# scripts/demo_compressor.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import OpenAIEmbeddings
from gpt_researcher.context.compression import ContextCompressor

pages = [
    {"url": "doc1", "title": "MoE basics",
     "raw_content": "Mixture-of-Experts is a sparse model architecture where a router selects a subset of experts per token."},
    {"url": "doc2", "title": "Switch Transformer",
     "raw_content": "Switch Transformer simplifies MoE by routing each token to exactly one expert."},
    {"url": "doc3", "title": "Cooking recipe",
     "raw_content": "Add salt, then garlic, then olive oil to the pasta water."},
]

async def main():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    cc  = ContextCompressor(documents=pages, embeddings=emb)
    ctx = await cc.async_get_context("What is Mixture of Experts?", max_results=3)
    print(ctx)

asyncio.run(main())
# 你会看到 doc1, doc2 被保留，doc3（cooking）被 EmbeddingsFilter 过滤
```

### 常见问题与 Debug 技巧

| 症状 | 排查 |
|---|---|
| 永远走慢路径，跑得很慢 | `total_chars` 或 `len(docs)` 超阈值；调高 `COMPRESSION_THRESHOLD` 或减少 `MAX_SEARCH_RESULTS_PER_QUERY` |
| 永远空返回 | `SIMILARITY_THRESHOLD` 设太高了；从默认 0.35 开始降 |
| Cohere 报维度不匹配 | 之前用 OpenAI embed 写的 vector_store 不能直接换 Cohere（维度不同）；要重建 vector_store |
| Ollama 报 `OLLAMA_BASE_URL` 未设 | 必须显式设 env，默认 `http://localhost:11434` 要自己写 |
| LMStudio 接入报 token 计数错 | 用 `EMBEDDING=custom:<model>`（自动加 `check_embedding_ctx_length=False`），不要用 `EMBEDDING=openai:` |
| 改 `default.py` SIMILARITY_THRESHOLD 没生效 | `ContextCompressor` 直接读 env 不读 cfg；改 env |
| Voyage 提示 `VOYAGE_API_KEY 未设` | Memory 里硬编码了 `os.environ["VOYAGE_API_KEY"]`，必须设 |
| 不知道某段为啥被过滤掉 | LangChain 的 EmbeddingsFilter 没有 verbose；自己写一段验证：`emb.embed_query(query)` + `emb.embed_documents([chunk])` 然后手动算 cosine |

调试时最有用的 LangChain 日志：

```python
import logging
logging.getLogger('langchain.retrievers.contextual_compression').setLevel(logging.DEBUG)
logging.getLogger('langchain.retrievers.document_compressors.embeddings_filter').setLevel(logging.DEBUG)
```

### 进阶练习建议

1. **接入 Cohere Rerank**：在 `ContextCompressor.__get_contextual_retriever` 的 `transformers=[splitter, relevance_filter]` 后追加 `CohereRerank()`。对比 rerank 前后的 SMART_LLM 报告质量。
2. **加 BM25 hybrid**：用 `langchain.retrievers.BM25Retriever` 与现有 dense 召回融合（用 RRF 或加权），看专有名词 query 的召回提升。
3. **修复 SIMILARITY_THRESHOLD 不一致**：让 `ContextCompressor` 读 `cfg.similarity_threshold`（需要把 cfg 传进来），统一配置入口。
4. **写自定义 cost_callback**：根据 `cfg.embedding_provider` 用不同价格表估算（Cohere $0.10/1M / Voyage $0.13/1M / OpenAI $0.02/1M）。
5. **更准的快路径门槛**：把 "字符数" 改成 "token 数"（`tiktoken.encode`），更准确。

---

## 延伸阅读

1. [LangChain ContextualCompressionRetriever](https://python.langchain.com/docs/how_to/contextual_compression/) — 本项目压缩管线的官方文档。
2. [Reciprocal Rank Fusion (RRF) 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Hybrid 检索融合的经典方法。
3. [text-embedding-3 技术 blog](https://openai.com/index/new-embedding-models-and-api-updates/) — 理解 chunk size/overlap 选取依据。
4. [Cohere Reranker](https://cohere.com/blog/rerank) — 进阶练习 1 的目标。
5. [Voyage AI Embedding Benchmarks](https://blog.voyageai.com/) — 选 embedding 时的对比基准。

---

> ✅ 本篇结束。下一篇 **`06_multi_agents_part1_architecture.md`** 进入项目"另一形态"——LangGraph 多 Agent。我们会从 `ChiefEditorAgent` 的 `StateGraph` 装配讲起，把整个图的节点 / 边 / 条件边全部画清，并对比它跟单 Agent 形态在哪些地方"换汤不换药"。
> 回复 **"继续"** 即可。
