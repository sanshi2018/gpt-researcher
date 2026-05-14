# scripts/demo_subtopics.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from gpt_researcher import GPTResearcher

async def main():
    r = GPTResearcher(query="Modern RAG evaluation methods", verbose=True)
    await r.conduct_research()
    subs = await r.get_subtopics()
    # subs 是 Subtopics(BaseModel) 实例
    for st in subs.subtopics:
        print("- ", st.task)

asyncio.run(main())