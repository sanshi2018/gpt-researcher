# 模块四：沙里淘金的深度 RAG (Advanced RAG & Reranking)

在上一模块中，我们的爬虫从四面八方抓取了大量的网页正文（Raw Content）。假设我们抓了 10 个网页，每个网页平均 5000 字，那就足足有 5 万字。

如果把这 5 万字直接塞给大语言模型（LLM）去写报告，会产生两个致命问题：
1. **烧钱且缓慢**：Token 消耗巨大，API 账单会让你心痛，且回复速度极慢。
2. **迷失在中间 (Lost in the Middle)**：LLM 常常会忽略长文本中间的关键信息，导致生成的报告依然质量不佳。

解决这个问题的银弹，就是 **RAG（检索增强生成）** 中的核心技术：**上下文压缩与重排过滤 (Context Compression & Reranking)**。

---

## 1. 核心机制：只把最纯的“金子”喂给大模型

想象你是一个 HR，面对 10 份长达 5 页的简历（原始网页）。你不可能一字不落地把这 50 页纸全部背下来。你的做法一定是：
1. **切片 (Chunking)**：把每份简历按段落剪开。
2. **打分 (Embedding & Scoring)**：拿着岗位需求（Query），给每个段落打个相关性分数。
3. **过滤 (Filtering/Reranking)**：扔掉废话（比如候选人的业余爱好），只把得分最高的前 5 段话拼在一起，作为最终的评估依据。

这就是 `gpt-researcher` 核心模块 `context/compression.py` 所做的事情！

---

## 2. 源码解剖：`ContextCompressor` 压缩器

打开 `gpt_researcher/context/compression.py`，聚焦到 `ContextCompressor` 类。它完美结合了 Langchain 的经典组件：

### 步骤一：配置文档压缩流水线 (`DocumentCompressorPipeline`)
```python
# gpt_researcher/context/compression.py 节选
def __get_contextual_retriever(self):
    # 1. 文本切块器 (Text Splitter)：每块 1000 字符，重叠 100 字符防止切断上下文
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # 2. 相关性过滤器 (Embeddings Filter)：向量化打分，丢弃低于设定阈值（如0.35）的废话
    relevance_filter = EmbeddingsFilter(
        embeddings=self.embeddings,
        similarity_threshold=self.similarity_threshold
    )
    
    # 3. 组装流水线
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    
    # ...
    return contextual_retriever
```
**解析**：
这里并没有调用传统的本地 ChromaDB/Faiss 向量数据库，而是采用了一种轻量级的内存过滤机制。它把刚爬下来的新鲜文本，在内存里切碎、向量化（转成浮点数数组）、计算余弦相似度，最后大浪淘沙留下精华。

### 步骤二：智能短路优化 (Fast Path Optimization)
源码中还隐藏着一个极其聪明的优化细节：
```python
async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
    # 优化点：计算当前所有网页内容的总字数
    total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
    chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

    # 如果总字数本身就很少（比如不到8000字），直接跳过昂贵的向量化压缩流水线！
    if total_chars < chunk_threshold and len(self.documents) <= max_results:
        # 直接返回，不浪费时间和 API 调用费
        ...
```
**看点**：在工程实践中，**最好的计算就是不计算**。当网页总内容并不长时，直接塞给 LLM 反而更划算。这是纯正的商业级代码才有的成本控制意识。

---

## 3. 实操体验：在内存中沙里淘金 (MRE)

下面我们来写一个独立脚本，模拟这个“文本切块 -> 向量评分 -> 过滤提取”的 RAG 过程。

### 环境准备
因为涉及向量化，你需要安装 `langchain-openai` 以及确保你的 `.env` 中有 `OPENAI_API_KEY`（我们会用 OpenAI 的 embedding 模型将文本转成向量）。

### MRE 代码：微型 RAG 上下文压缩器
在根目录下新建文件 `mre_module4.py`：

```python
import asyncio
import os
from langchain_openai import OpenAIEmbeddings
from gpt_researcher.context.compression import ContextCompressor
from gpt_researcher.prompts import PromptFamily

async def main():
    # 1. 设定用户的检索目标 (Query)
    query = "苹果公司最近在人工智能领域有什么动作？"
    print(f"🎯 检索目标: {query}\n")

    # 2. 模拟从爬虫抓取回来的长篇大论（故意掺杂大量无关信息）
    mock_documents = [
        {
            "title": "今日科技新闻",
            "url": "https://example.com/news",
            "raw_content": (
                "今天天气不错，很多人去公园散步。" # 无关废话
                "马斯克又在推特上发了火箭发射的照片。" # 无关废话
                "据内部消息透露，苹果公司(Apple)在今年的 WWDC 大会上正式推出了 Apple Intelligence，"
                "宣布全面进军人工智能领域，并与 OpenAI 达成深度合作，将 ChatGPT 整合进 iOS 18。" # 核心金子
                "另外，由于通货膨胀，各大超市的鸡蛋价格上涨了 10%。" # 无关废话
            ) * 50 # 故意把文本放大多倍，模拟长篇网页，迫使触发压缩机制
        }
    ]

    # 3. 初始化 OpenAI 的向量嵌入模型 (用来给文本打分)
    # 注意：需要配置好 OPENAI_API_KEY 环境变量
    embeddings_model = OpenAIEmbeddings()

    # 4. 实例化我们的沙里淘金压缩器
    print("⏳ 正在对长文本进行切块、向量化评分与压缩...\n")
    compressor = ContextCompressor(
        documents=mock_documents,
        embeddings=embeddings_model,
        max_results=3, # 我们只要最相关的 3 个切片
        prompt_family=PromptFamily(config=None)
    )

    # 5. 执行压缩！
    # 会在内存中切块并计算相关度，丢掉鸡蛋涨价和天气不错的内容
    compressed_context = await compressor.async_get_context(query=query)

    print("✅ 淘金完毕！请看最终喂给大模型的精华 Context：")
    print("-" * 50)
    print(compressed_context)
    print("-" * 50)

if __name__ == "__main__":
    # 请确保环境变量中存在 OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 请先设置 OPENAI_API_KEY 环境变量再运行此示例。")
    else:
        asyncio.run(main())
```

### 运行结果预期
运行这个脚本（`python mre_module4.py`），向量模型会把那段包含“天气”、“马斯克”、“鸡蛋”的废话全部过滤掉，最后输出的 `compressed_context` 中，只会精准地包含“苹果公司推出 Apple Intelligence，将 ChatGPT 整合进 iOS 18”的关键句子。

正是有了这一步大浪淘沙，GPT-Researcher 才能保证无论抓多少网页，大模型永远能吃到最纯粹、最相关的核心干货。

---

## 下一步 (Next Steps)

到这里，我们已经把最纯正的“事实原料”提炼出来了。
下一步，我们要把这些原料放进搅拌机，让 Agent 动笔写出一篇万字长文！

这就涉及到极度考验 Prompt 功底的**报告生成阶段**。如果报告超长，它又是怎么保证生成不中断的？

**当您阅读完毕并领略了内存向量过滤的魅力后，请回复“继续输出模块五”！**
