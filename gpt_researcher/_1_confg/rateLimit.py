import asyncio, time
from gpt_researcher.utils.workers import WorkerPool

pool = WorkerPool(max_workers=3, rate_limit_delay=3)  # 同时 3 个、间隔 ≥ 0.5s

async def fake_request(i):
    async with pool.throttle():
        t = time.time()
        print(f"req{i} starts at {t:.2f}")
        await asyncio.sleep(4)   # 模拟下游耗时

async def main():
    await asyncio.gather(*[fake_request(i) for i in range(8)])

asyncio.run(main())
# 你会看到 req0~req2 几乎同时开始，但任两次开始之间至少差 0.5s