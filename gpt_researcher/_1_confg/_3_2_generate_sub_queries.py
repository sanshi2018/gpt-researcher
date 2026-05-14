# scripts/demo_subqueries.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from gpt_researcher.config import Config
from gpt_researcher.actions.query_processing import generate_sub_queries

async def main():
    cfg = Config()
    qs = await generate_sub_queries(
        query="Why is RAG with reranking necessary?",
        parent_query="",
        report_type="research_report",
        context=[],
        cfg=cfg,
    )
    print("Sub-queries:", qs)

asyncio.run(main())