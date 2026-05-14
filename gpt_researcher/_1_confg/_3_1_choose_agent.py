# scripts/demo_choose_agent.py
import asyncio
from dotenv import load_dotenv; load_dotenv()
from gpt_researcher.config import Config
from gpt_researcher.actions.agent_creator import choose_agent
from gpt_researcher.prompts import PromptFamily

async def main():
    cfg = Config()
    pf  = PromptFamily(cfg)

    queries = [
        "Should I invest in Anthropic if it goes public in 2026?",
        "What is the cleanest beach in Tel Aviv for kids?",
        "Compare Postgres pgvector vs Qdrant for production RAG",
        "为什么大语言模型推理时 KV-cache 是必需的？",
    ]
    for q in queries:
        agent, role = await choose_agent(q, cfg, prompt_family=pf)
        print(f"\n[{q[:50]}]")
        print(f"  agent: {agent}")
        print(f"  role : {role[:120]}...")

asyncio.run(main())