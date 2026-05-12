import warnings
from datetime import date, datetime, timezone

from langchain_core.documents import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## 提示词家族 (Prompt Families) #############################################################

class PromptFamily:
    """用于提示词格式化的通用类。

    该类可以被特定模型的派生类覆盖。其方法分为两组：

    1. 提示词生成器 (Prompt Generators)：这些方法遵循标准格式，并与 ReportType 枚举相关联。
       应通过 get_prompt_by_report_type 来访问它们。

    2. 提示词方法 (Prompt Methods)：这些是针对特定情况的方法，没有标准签名，直接在智能体(agent)代码中调用。

    所有派生类必须保留相同的方法名称集，但可以覆盖单个方法。
    """

    def __init__(self, config: Config):
        """使用 config 实例进行初始化。派生类可以使用它来根据配置的模型
        和/或提供商选择正确的提示词。
        """
        self.cfg = config

    # MCP 专用提示词
    @staticmethod
    def generate_mcp_tool_selection_prompt(query: str, tools_info: List[Dict], max_tools: int = 3) -> str:
        """
        生成基于 LLM 的 MCP 工具选择的提示词。

        Args:
            query: 研究查询
            tools_info: 可用工具及其元数据的列表
            max_tools: 要选择的最大工具数

        Returns:
            str: 工具选择提示词
        """
        import json

        return f"""你是一名研究助手，正在协助为一项研究查询选择最相关的工具。

研究查询："{query}"

可用工具：
{json.dumps(tools_info, indent=2, ensure_ascii=False)}

任务：分析这些工具，并选择正好 {max_tools} 个与研究给定查询最相关的工具。

选择标准：
- 选择能够提供与查询相关的信息、数据或见解的工具
- 优先选择能够搜索、检索或访问相关内容的工具
- 考虑相互补充的工具（例如，不同的数据源）
- 排除明显与研究主题无关的工具

返回一个具有以下精确格式的 JSON 对象：
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "关于为什么此工具相关的详细解释"
    }}
  ],
  "selection_reasoning": "关于整体选择策略的解释"
}}

请准确选择 {max_tools} 个工具，并按其与研究查询的相关性进行排序。
"""

    @staticmethod
    def generate_mcp_research_prompt(query: str, selected_tools: List) -> str:
        """
        生成使用所选工具执行 MCP 研究的提示词。

        Args:
            query: 研究查询
            selected_tools: 所选 MCP 工具的列表

        Returns:
            str: 研究执行提示词
        """
        # 处理 selected_tools 可能是字符串或具有 .name 属性的对象的情况
        tool_names = []
        for tool in selected_tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            else:
                tool_names.append(str(tool))

        return f"""你是一名拥有专用工具访问权限的研究助手。你的任务是研究以下查询，并提供全面、准确的信息。

研究查询："{query}"

说明：
1. 使用可用工具收集有关查询的相关信息。
2. 如果需要，调用多个工具以获得全面的覆盖。
3. 如果工具调用失败或返回空结果，请尝试替代方法。
4. 尽可能综合来自多个来源的信息。
5. 专注于直接解决查询的、相关且基于事实的信息。

可用工具：{tool_names}

请进行彻底的研究并提供你的发现。战略性地使用这些工具，以收集最相关和最全面的信息。"""

    # 图像生成提示词
    @staticmethod
    def generate_image_analysis_prompt(
            query: str,
            sections: List[Dict[str, Any]],
            max_images: int = 3,
    ) -> str:
        """生成提示词，用于分析哪些报告章节需要图像。

        Args:
            query: 研究查询。
            sections: 包含标题和内容的报告章节列表。
            max_images: 建议的最大图像数。

        Returns:
            str: 分析提示词。
        """
        sections_text = "\n\n".join([
            f"### 第 {i + 1} 节：{s['header']}\n{s['content'][:500]}..."
            for i, s in enumerate(sections)
        ])

        return f"""分析以下研究报告的章节，并确定哪 {max_images} 个章节最能从视觉插图或图表中受益。

研究主题：{query}

报告章节：
{sections_text}

对于每个建议的章节，请提供：
1. 章节编号（从 1 开始索引）
2. 一个具体、详细的图像提示词，用于创建信息丰富的插图
3. 简要解释为什么该章节受益于可视化

重要准则：
- 选择那些通过视觉表现能真正有助于理解的章节
- 关注本质上适合视觉展示的概念、过程、比较、数据流或统计数据
- 避免纯文本分析、引言或结论的章节
- 图像提示词必须足够具体，以生成相关的专业插图
- 图像应当具有信息量和教育意义，而非纯粹装饰
- 考虑使用图表、流程图、对比图或概念插图

以 JSON 格式回复：
{{
    "suggestions": [
        {{
            "section_number": 1,
            "section_header": "章节标题",
            "image_prompt": "用于生成信息丰富的插图的详细提示词...",
            "image_type": "diagram|flowchart|comparison|concept|data_visualization",
            "reason": "为什么该章节能从可视化中受益"
        }}
    ]
}}

仅返回 JSON 内容，不要有任何其他附加文本。"""

    @staticmethod
    def generate_image_prompt_enhancement(
            base_prompt: str,
            section_content: str,
            research_topic: str,
    ) -> str:
        """为图像提示词增加上下文，以获得更好的生成效果。

        Args:
            base_prompt: 基础图像生成提示词。
            section_content: 报告章节的内容。
            research_topic: 主要研究主题。

        Returns:
            str: 增强后的图像提示词。
        """
        return f"""为一份研究报告创建一张专业、信息丰富的插图。

研究主题：{research_topic}

图像描述：{base_prompt}

来自报告的上下文：
{section_content[:800]}

风格要求：
- 适合学术/商业报告的专业、整洁的设计
- 清晰、易于理解的视觉元素
- 现代、极简的美学风格
- 使用专业的调色板（蓝色、青色、灰色）
- 避免图像中出现过多文字
- 高对比度以提高可读性
- 如果展示数据或比较，请使用清晰的标签和图例
- 既适合数字屏幕观看，也适合打印"""

    @staticmethod
    def generate_search_queries_prompt(
            question: str,
            parent_query: str,
            report_type: str,
            max_iterations: int = 3,
            context: List[Dict[str, Any]] = [],
    ):
        """为给定问题生成搜索查询的提示词。
        Args:
            question (str): 需要生成搜索查询提示词的问题
            parent_query (str): 主问题（仅与详细报告相关）
            report_type (str): 报告类型
            max_iterations (int): 要生成的最大搜索查询数量
            context (str): 通过实时网络信息更好地理解任务的上下文

        Returns: str: 给定问题的搜索查询提示词
        """

        if (
                report_type == ReportType.DetailedReport.value
                or report_type == ReportType.SubtopicReport.value
        ):
            task = f"{parent_query} - {question}"
        else:
            task = question

        context_prompt = f"""
你是一位经验丰富的研究助手，负责生成搜索查询，以寻找与以下任务相关的在线信息："{task}"。
上下文信息：{context}

利用上述上下文来告知和完善你的搜索查询。上下文提供了实时网络信息，能帮助你生成更具体、相关的查询。考虑上下文中提到的任何时事、最新进展或具体细节，从而增强搜索查询的质量。
""" if context else ""

        dynamic_example = ", ".join([f'"query {i + 1}"' for i in range(max_iterations)])

        return f"""编写 {max_iterations} 个针对以下任务的 Google 搜索查询语句，通过在线搜索形成客观意见："{task}"

如果需要，请假设当前日期为 {datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。

{context_prompt}
你必须以字符串列表的形式回复，格式如下：[{dynamic_example}]。
回复必须且只能包含该列表。
"""

    @staticmethod
    def generate_report_prompt(
            question: str,
            context,
            report_source: str,
            report_format="apa",
            total_words=1000,
            tone=None,
            language="chinese",
    ):
        """为给定问题和研究摘要生成报告的提示词。
        Args: question (str): 需要生成报告提示词的问题
                research_summary (str): 需要生成报告提示词的研究摘要
        Returns: str: 给定问题和研究摘要的报告提示词
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源 URL 作为参考文献列出，并确保不添加重复的来源，每个来源仅提供一个引用。
每个 URL 都应添加超链接格式：[网站名称](url)
此外，只要报告中引用了相关的 URL，你必须在对应位置包含该超链接：

例如：作者, A. A. (年份, 月份 日期). 网页标题. 网站名称. [网站名称](url)
"""
        else:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的源文档名称作为参考文献列出，并确保不添加重复的来源，每个来源仅提供一个引用。"
"""

        tone_prompt = f"以 {tone.value} 的语气编写报告。" if tone else ""

        return f"""
信息："{context}"
---
使用上述信息，在详细报告中回答以下查询或任务："{question}" ——
报告应侧重于对查询的解答，要求结构良好、信息丰富、深入且全面，如果可能的话应包含事实和数字，至少包含 {total_words} 个字。
你应该尽可能地使用所提供的所有相关且必要的信息，把报告写得越长越好。

请在你的报告中遵循以下所有指导原则：
- 你必须基于所提供的信息形成自己明确且有效的观点。不要得出泛泛而谈或毫无意义的结论。
- 你必须使用 Markdown 语法和 {report_format} 格式撰写报告。
- 用清晰的 Markdown 标题构建报告结构：使用 # 作为主标题，## 作为主要章节，### 作为子章节。
- 在展示结构化数据或进行比较时，使用 Markdown 表格以增强可读性。
- 你必须优先考虑所用来源的相关性、可靠性和重要性。选择可信的来源，而不是不可靠的来源。
- 如果来源可信，你还必须优先采用新文章而非旧文章。
- 你绝对不能包含目录，但必须使用适当的 Markdown 标题 (# ## ###) 清晰地构建你的报告。
- 使用 {report_format} 格式的文中引用，并以 Markdown 超链接的形式放置在引用它们的句子或段落的末尾，如下所示：([文中引用](url))。
- 别忘了在报告末尾添加 {report_format} 格式的参考文献列表，提供完整的 URL 链接，不要隐藏在超链接文本中。
- {reference_prompt}
- {tone_prompt}
你必须使用以下语言编写报告：{language}。
请全力以赴，这对我的职业生涯非常重要。
假设当前日期是 {date.today()}。
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""你的目标是评估和精选针对研究任务："{query}" 所提供的网页抓取内容。
    同时，优先保留相关且高质量的信息，尤其是包含统计数据、数字或具体数据的来源。

最终策划的列表将用作撰写研究报告的上下文，因此请优先考虑：
- 尽可能保留最多的原始信息，并特别重视包含定量数据或独特见解的来源
- 涵盖广泛的视角和见解
- 仅过滤掉明显无关或完全无法使用的内容

评估准则：
1. 评估每个来源的标准：
   - 相关性：包含直接或部分与研究查询相关的来源。宁可多留，不可错删。
   - 可信度：倾向于权威来源，但除非明显不可信，否则也保留其他来源。
   - 时效性：优先考虑最新信息，除非旧数据是必不可少或具有价值的。
   - 客观性：如果带有偏见的来源提供了独特或互补的视角，也应保留。
   - 定量价值：给予包含统计数据、数字或其他具体数据的来源更高的优先级。
2. 来源选择：
   - 包含尽可能多相关的来源，最多 {max_results} 个，侧重于覆盖面和多样性。
   - 优先选择含有统计数据、数值数据或可验证事实的来源。
   - 内容有重叠是可以接受的，只要它能增加深度，特别是在涉及数据时。
   - 仅当来源完全无关、严重过时或因内容质量差而无法使用时，才将其排除。
3. 内容保留：
   - 绝对不要重写、总结或浓缩任何来源的内容。
   - 保留所有可用信息，仅清理明显的乱码或排版格式问题。
   - 如果边缘相关或不完整的来源包含有价值的数据或见解，请保留它们。

待评估的来源列表：
{sources}

你必须以原始来源完全相同的 JSON 列表格式返回响应。
响应不能包含任何 markdown 格式或附加文本（如 ```json），只能是 JSON 列表！
"""

    @staticmethod
    def generate_resource_report_prompt(
            question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="chinese"
    ):
        """为给定问题和研究摘要生成资源报告的提示词。

        Args:
            question (str): 需要生成资源报告提示词的问题。
            context (str): 需要生成资源报告提示词的研究摘要。

        Returns:
            str: 给定问题和研究摘要的资源报告提示词。
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
            你必须包含所有相关的来源 URL。
            每个 URL 都应添加超链接：[网站名称](url)
            """
        else:
            reference_prompt = f"""
            你必须在报告末尾将所有使用过的源文档名称作为参考文献列出，并确保不添加重复的来源，每个来源仅提供一个引用。"
        """

        return (
            f'"""{context}"""\n\n基于上述信息，生成一份针对以下问题或主题的文献推荐报告："{question}"。'
            "该报告应对每项推荐的资源进行详细分析，解释每个来源如何有助于解答该研究问题。\n"
            "重点关注每个来源的相关性、可靠性和重要性。\n"
            "确保报告结构良好、信息丰富、有深度，并遵循 Markdown 语法。\n"
            "适当时使用 Markdown 表格和其他格式功能，以清晰地组织和展示信息。\n"
            "只要有可能，就应纳入相关事实、图表和数字。\n"
            f"报告的最短长度应为 {total_words} 个字。\n"
            f"你必须使用以下语言撰写报告：{language}。\n"
            "你必须包含所有相关的来源 URL。\n"
            "每个 URL 都应添加超链接：[网站名称](url)\n"
            f"{reference_prompt}"
        )

    @staticmethod
    def generate_custom_report_prompt(
            query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000,
            language: str = "chinese"
    ):
        return f'"{context}"\n\n{query_prompt}'

    @staticmethod
    def generate_outline_report_prompt(
            question, context, report_source: str, report_format="apa", tone=None, total_words=1000,
            language: str = "chinese"
    ):
        """为给定问题和研究摘要生成大纲报告的提示词。
        Args: question (str): 需要生成大纲报告提示词的问题
                research_summary (str): 需要生成大纲报告提示词的研究摘要
        Returns: str: 给定问题和研究摘要的大纲报告提示词
        """

        return (
            f'"""{context}""" 使用上述信息，生成一份针对以下问题或主题的 Markdown 语法形式的研究报告大纲："{question}"。'
            "大纲应为研究报告提供一个结构良好的框架，包括主要章节、子章节以及要涵盖的关键点。"
            f"该研究报告要求详细、信息丰富、深入，且至少 {total_words} 字。"
            "使用适当的 Markdown 语法来格式化大纲，并确保可读性。"
            "考虑使用 Markdown 表格和其他格式功能，以增强信息的表现力。"
        )

    @staticmethod
    def generate_deep_research_prompt(
            question: str,
            context: str,
            report_source: str,
            report_format="apa",
            tone=None,
            total_words=2000,
            language: str = "chinese"
    ):
        """生成深度研究报告的提示词，专门处理分层的研究结果。
        Args:
            question (str): 研究问题
            context (str): 包含带引用的学习内容的研究上下文
            report_source (str): 研究来源 (如 web 等)
            report_format (str): 报告格式化风格
            tone: 写作时的语气
            total_words (int): 最小字数
            language (str): 输出语言
        Returns:
            str: 深度研究报告提示词
        """
        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源 URL 作为参考文献列出，并确保不添加重复的来源，每个来源仅提供一个引用。
每个 URL 都应添加超链接格式：[网站名称](url)
此外，只要报告中引用了相关的 URL，你必须在对应位置包含该超链接：

例如：作者, A. A. (年份, 月份 日期). 网页标题. 网站名称. [网站名称](url)
"""
        else:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的源文档名称作为参考文献列出，并确保不添加重复的来源，每个来源仅提供一个引用。"
"""

        tone_prompt = f"以 {tone.value} 的语气编写报告。" if tone else ""

        return f"""
使用以下层级式研究信息和引用内容：

"{context}"

撰写一份全面的研究报告，回答此查询："{question}"

该报告应当：
1. 综合来自多个研究深度层级的信息
2. 整合不同研究分支的发现
3. 呈现从基础到高级见解循序渐进连贯一致的叙述
4. 贯穿全文保持适当的来源引用格式
5. 结构良好，具有清晰的章节和子章节
6. 至少 {total_words} 个字
7. 采用 {report_format} 格式，遵循 Markdown 语法
8. 在展示比较数据、统计信息或结构化信息时，使用 Markdown 表格、列表和其他格式功能

附加要求：
- 优先突出从更深层次研究中涌现的深刻见解
- 强调不同研究分支之间的联系
- 包含相关的统计数据、具体数据和具体示例
- 你必须基于提供的信息形成自己明确且有效的观点。不要得出泛泛而谈或毫无意义的结论。
- 你必须优先考虑所用来源的相关性、可靠性和重要性。选择可信的来源，而不是不可靠的来源。
- 如果来源可信，你还必须优先采用新文章而非旧文章。
- 使用 {report_format} 格式的文中引用，并以 Markdown 超链接的形式放置在引用它们的句子或段落的末尾，如下所示：([文中引用](url))。
- {tone_prompt}
- 请用 {language} 写作

{reference_prompt}

请写出一份透彻的、经过充分研究的报告，将所有收集到的信息综合成一个连贯的整体。
假设当前日期是 {datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。
"""

    @staticmethod
    def auto_agent_instructions():
        return """
这项任务涉及研究给定主题，无论其复杂性如何，或是否存在明确答案。研究由特定的服务器进行，具体取决于其类型和角色，每台服务器都需要不同的指令。
Agent（智能体）
服务器是根据主题的领域和可用于研究提供主题的服务器的具体名称来确定的。代理按其专业领域进行分类，并且每种服务器类型都有关联的表情符号。

示例：
任务："我应该投资苹果股票吗？"
回复：
{
    "server": "💰 金融智能体",
    "agent_role_prompt": "你是一位经验丰富的金融分析师 AI 助手。你的主要目标是基于提供的数据和趋势，撰写全面、敏锐、公正且系统化编排的财务报告。"
}
任务："倒卖运动鞋会变得有利可图吗？"
回复：
{
    "server":  "📈 商业分析师智能体",
    "agent_role_prompt": "你是一位经验丰富的 AI 商业分析师助手。你的主要目标是根据提供的商业数据、市场趋势和战略分析，生成全面、深刻、公正且结构系统的商业报告。"
}
任务："特拉维夫最有趣的景点有哪些？"
回复：
{
    "server":  "🌍 旅游智能体",
    "agent_role_prompt": "你是一位周游世界的 AI 导游助手。你的主要目的是针对给定地点起草引人入胜、有见地、公正且结构良好的旅游报告，内容包括历史、景点和文化见解。"
}
"""

    @staticmethod
    def generate_summary_prompt(query, data):
        """为给定问题和文本生成摘要提示词。
        Args: question (str): 需要生成摘要提示词的问题
                text (str): 需要生成摘要提示词的文本
        Returns: str: 给定问题和文本的摘要提示词
        """

        return (
            f'{data}\n 使用上述文本，根据以下任务或查询对其进行总结："{query}"。\n'
            f"如果无法使用文本回答该查询，你必须对文本进行简短的概括总结。\n"
            f"如果可能的话，请包含所有的事实信息，例如数字、统计数据、引言等。"
        )

    @staticmethod
    def generate_quick_summary_prompt(query: str, context: str) -> str:
        """为给定问题和上下文生成快速摘要提示词。
        Args:
            query (str): 需要生成摘要的查询
            context (str): 需要总结的搜索结果
        Returns:
            str: 快速摘要提示词
        """
        return f"""
仅根据所提供的搜索结果，综合出对以下查询的全面回答。
查询："{query}"

搜索结果：
{context}

说明：
1. 提供一段连续、完整的叙述性总结。
2. 使用数字 [1]、[2] 等引用你的来源，这些数字对应于搜索结果。
3. 如果结果不足以回答查询，请明确说明。
4. 关注准确性和相关性。
"""

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        """将文档列表压缩为上下文格式的字符串"""
        return f"\n".join(f"来源：{d.metadata.get('source')}\n"
                          f"标题：{d.metadata.get('title')}\n"
                          f"内容：{d.page_content}\n"
                          for i, d in enumerate(docs)
                          if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        """将本地 Web 文档与从互联网抓取的上下文连接起来"""
        return f"来自本地文档的上下文：{docs_context}\n\n来自网络来源的上下文：{web_context}"

    ################################################################################################

    # 详细报告提示词 (DETAILED REPORT PROMPTS)

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return """
提供主要主题：

{task}

以及研究数据：

{data}

- 构建一个子主题列表，该列表反映要生成的针对该任务的报告文档中的各个标题。
- 这些是可能的子主题列表：{subtopics}。
- 不应有任何重复的子主题。
- 将子主题数量限制在最多 {max_subtopics} 个。
- 最后，按照它们的任务顺序对子主题进行排列，顺序应当相关且有意义，能够呈现在详细的报告中。

“重要提示！”：
- 每个子主题绝对必须仅与主要主题和提供的研究数据相关！

{format_instructions}
"""

    @staticmethod
    def generate_subtopic_report_prompt(
            current_subtopic,
            existing_headers: list,
            relevant_written_contents: list,
            main_topic: str,
            context,
            report_format: str = "apa",
            max_subsections=5,
            total_words=800,
            tone: Tone = Tone.Objective,
            language: str = "chinese",
    ) -> str:
        return f"""
上下文：
"{context}"

主要主题与子主题：
使用现有的最新信息，在主要主题：{main_topic} 的范畴下，构建一份关于子主题：{current_subtopic} 的详细报告。
你必须将子章节的数量限制在最多 {max_subsections} 个。

内容重点：
- 报告应侧重于回答该问题，结构合理、信息丰富、深入，如有事实和数字应予以包含。
- 使用 Markdown 语法并遵循 {report_format.upper()} 格式。
- 在展示数据、比较或结构化信息时，使用 Markdown 表格以增强可读性。

重要提示：内容与章节的唯一性：
- 这一部分的说明至关重要，以确保内容是独一无二的，且不与现有的报告发生重叠。
- 在编写任何新的子章节之前，仔细审查下面提供的现有标题和现有撰写内容。
- 杜绝生成任何在现有撰写内容中已经被覆盖过的内容。
- 不要将任何现有标题用作新的子章节标题。
- 不要重复在现有内容中已经涉及的信息或其高度相似的变体，以避免重复。
- 如果有嵌套的子章节，请确保它们是独特的，且不包含在现有的内容中。
- 确保你的内容是全新的，并且不与先前子主题报告中涵盖的任何信息重叠。

“现有的子主题报告”：
- 现有的子主题报告及其各节标题：

    {existing_headers}

- 现有的、从之前子主题报告中撰写的内容：

    {relevant_written_contents}

“结构与格式”：
- 由于这个子报告将是一份更大报告的一部分，因此仅需包含主要正文并划分成合适的子主题，不要有任何引言或结论部分。

- 你必须在报告中提到来源 URL 时添加 Markdown 超链接，例如：

    ### 章节标题

    这是一段示例文本 ([文中引用](url))。

- 使用 H2 (##) 作为主子主题标题，H3 (###) 作为子章节标题。
- 使用较小的 Markdown 标题（例如 H2 或 H3）来构建内容结构，避免使用最大的标题 (H1)，因为 H1 将留给整体大型报告的标题使用。
- 将你的内容划分为相互区分的各节，用于补充而非重叠现有报告。
- 当向报告中添加相似或相同的子章节时，你应该清楚地指出新内容与先前子主题报告现有内容之间的差异。例如：

    ### 新标题（与现有标题类似）

    前面的章节讨论了 [主题 A]，而本节将探讨 [主题 B]。”

“日期”：
如有需要，请假设当前日期是 {datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。

“重要提示！”：
- 你必须使用以下语言编写报告：{language}。
- 焦点必须集中在主要主题上！你必须排除任何与其无关的信息！
- 绝对不能有任何引言、结论、摘要或参考书目部分。
- 你必须使用 {report_format.upper()} 格式的文中引用，并以 Markdown 超链接的形式将其置于引用它的句子或段落末尾，例如：([文中引用](url))。
- 在添加相似或相同子章节的必要情况下，你必须在报告中提及现有内容与新内容之间的差异。
- 报告应至少包含 {total_words} 个字。
- 在整篇报告中保持 {tone.value} 的语调。

不要添加结论部分。
"""

    @staticmethod
    def generate_draft_titles_prompt(
            current_subtopic: str,
            main_topic: str,
            context: str,
            max_subsections: int = 5
    ) -> str:
        return f"""
“上下文”：
"{context}"

“主要主题与子主题”：
利用最新信息，在主要主题：{main_topic} 的基础上，为子主题：{current_subtopic} 的详细报告起草章节标题草案。

“任务”：
1. 为该子主题报告创建一个草案章节标题列表。
2. 每个标题应该简洁且与子主题相关。
3. 标题不应过于宏大空泛，而需足够详细以涵盖子主题的主要方面。
4. 采用 Markdown 语法表示标题，使用 H3 (###)，因为 H1 和 H2 将被用于更大报告的标题中。
5. 确保标题涵盖了子主题的主要方面。

“结构与格式”：
以 Markdown 语法的列表格式提供标题草案，例如：

### 标题 1
### 标题 2
### 标题 3

“重要提示！”：
- 焦点必须集中在主要主题上！你必须排除任何与其无关的信息！
- 绝对不能包含任何引言、结论、摘要或参考文献部分的标题。
- 专心生成标题，不要生成内容本身。
"""

    @staticmethod
    def generate_report_introduction(question: str, research_summary: str = "", language: str = "chinese",
                                     report_format: str = "apa") -> str:
        return f"""{research_summary}\n
利用上述最新信息，撰写一份关于该主题的详细报告引言—— {question}。
- 引言应简洁、结构良好、信息丰富，并采用 Markdown 语法。
- 由于引言将作为更大报告的一部分，因此不要包含在报告中常见的其他任何章节。
- 引言应以一个适合整篇报告主题的 H1 标题开头。
- 你必须使用 {report_format.upper()} 格式的文中引用，并以 Markdown 超链接的形式放置在引用该内容的句子或段落末尾，如：([文中引用](url))。
如有需要，假设当前日期为 {datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。
- 输出必须使用 {language} 语言。
"""

    @staticmethod
    def generate_report_conclusion(query: str, report_content: str, language: str = "chinese",
                                   report_format: str = "apa") -> str:
        """
        生成简洁的结论，总结研究报告的主要发现及其影响。

        Args:
            query (str): 研究任务或问题。
            report_content (str): 研究报告的内容。
            language (str): 撰写结论应使用的语言。

        Returns:
            str: 总结了报告主要发现和影响的简明结论。
        """
        prompt = f"""
    根据以下研究报告和研究任务，请写出一个简洁的结论，概括主要发现及其影响：

    研究任务：{query}

    研究报告：{report_content}

    你的结论应当：
    1. 回顾研究的关键点
    2. 强调最重要的发现
    3. 探讨任何潜在的影响或下一步行动
    4. 篇幅大约 2-3 个段落

    如果报告末尾还没有写 "## 结论" 这个章节标题，请在你的结论最上方加上它。
    你必须使用 {report_format.upper()} 格式的文中引用，并以 Markdown 超链接的形式放置在引用它的句子或段落的末尾，如下所示：([文中引用](url))。

    重要提示：整个结论必须用 {language} 语言撰写。

    撰写结论：
    """

        return prompt


class GranitePromptFamily(PromptFamily):
    """IBM Granite 模型的提示词"""

    def _get_granite_class(self) -> type[PromptFamily]:
        """根据版本号获取正确的 Granite 提示词系列类"""
        if "3.3" in self.cfg.smart_llm:
            return Granite33PromptFamily
        if "3" in self.cfg.smart_llm:
            return Granite3PromptFamily
        # 如果不是已知版本，则返回默认类
        return PromptFamily

    def pretty_print_docs(self, *args, **kwargs) -> str:
        return self._get_granite_class().pretty_print_docs(*args, **kwargs)

    def join_local_web_documents(self, *args, **kwargs) -> str:
        return self._get_granite_class().join_local_web_documents(*args, **kwargs)


class Granite3PromptFamily(PromptFamily):
    """IBM Granite 3.X 模型 (早于 3.3 版本) 的提示词"""

    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        if not docs:
            return ""
        all_documents = "\n\n".join([
            f"文档 {doc.metadata.get('source', i)}\n" + \
            f"标题：{doc.metadata.get('title')}\n" + \
            doc.page_content
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """使用 Granite 偏好的格式连接本地和 Web 文档"""
        if isinstance(docs_context, str) and docs_context.startswith(cls._DOCUMENTS_PREFIX):
            docs_context = docs_context[len(cls._DOCUMENTS_PREFIX):]
        if isinstance(web_context, str) and web_context.endswith(cls._DOCUMENTS_SUFFIX):
            web_context = web_context[:-len(cls._DOCUMENTS_SUFFIX)]
        all_documents = "\n\n".join([docs_context, web_context])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])


class Granite33PromptFamily(PromptFamily):
    """IBM Granite 3.3 模型的提示词"""

    _DOCUMENT_TEMPLATE = """<|start_of_role|>document {{"document_id": "{document_id}"}}<|end_of_role|>
{document_content}<|end_of_text|>
"""

    @staticmethod
    def _get_content(doc: Document) -> str:
        doc_content = doc.page_content
        if title := doc.metadata.get("title"):
            doc_content = f"标题：{title}\n{doc_content}"
        return doc_content.strip()

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join([
            cls._DOCUMENT_TEMPLATE.format(
                document_id=doc.metadata.get("source", i),
                document_content=cls._get_content(doc),
            )
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """使用 Granite 偏好的格式连接本地和 Web 文档"""
        return "\n\n".join([docs_context, web_context])


## 工厂方法 (Factory) ######################################################################

# 这是各类提示词生成函数的函数签名
PROMPT_GENERATOR = Callable[
    [
        str,  # question
        str,  # context
        str,  # report_source
        str,  # report_format
        str | None,  # tone
        int,  # total_words
        str,  # language
    ],
    str,
]

report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
}


def get_prompt_by_report_type(
        report_type: str,
        prompt_family: type[PromptFamily] | PromptFamily,
):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"无效的报告类型：{report_type}。\n"
            f"请使用以下类型之一：{', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"正在使用默认报告类型的提示词：{default_report_type}。",
            UserWarning,
        )
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(default_report_type))
    return prompt_by_type


prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
    PromptFamilyEnum.Granite.value: GranitePromptFamily,
    PromptFamilyEnum.Granite3.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite31.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite32.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite33.value: Granite33PromptFamily,
}


def get_prompt_family(
        prompt_family_name: PromptFamilyEnum | str, config: Config,
) -> PromptFamily:
    """按名称或值获取提示词家族实例。"""
    if isinstance(prompt_family_name, PromptFamilyEnum):
        prompt_family_name = prompt_family_name.value
    if prompt_family := prompt_family_mapping.get(prompt_family_name):
        return prompt_family(config)
    warnings.warn(
        f"无效的提示词家族：{prompt_family_name}。\n"
        f"请使用以下选项之一：{', '.join([enum_value for enum_value in prompt_family_mapping.keys()])}\n"
        f"正在使用默认提示词家族：{PromptFamilyEnum.Default.value}。",
        UserWarning,
    )
    return PromptFamily(config)