# scripts/min_research.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from gpt_researcher import GPTResearcher

async def main():
    r = GPTResearcher(
        query="What are the latest breakthroughs in small language models in 2026?",
        report_type="research_report",
        verbose=True,
    )
    await r.conduct_research()
    md = await r.write_report()

    print("---REPORT---")
    print(md[:600], "...\n")
    print("Cost USD:", r.get_costs())
    print("Per step:", r.get_step_costs())
    print("Sources :", len(r.get_source_urls()))

asyncio.run(main())