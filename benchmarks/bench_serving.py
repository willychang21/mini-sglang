"""Chapter 5 (and Part I Chapter 5 metrics): end-to-end serving benchmark.

Sends streaming requests to a running server at a fixed concurrency and reports
TTFT, TPOT, end-to-end latency and throughput.

    python -m minisgl --model /path/to/Qwen3-0.6B --device cpu &
    python benchmarks/bench_serving.py --url http://127.0.0.1:1919 --concurrency 1,4,8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True}
    start = time.perf_counter()
    token_times = []
    async with client.stream("POST", url + "/v1/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if json.loads(line[6:])["choices"][0].get("text"):
                token_times.append(time.perf_counter() - start)
    end = time.perf_counter() - start
    ttft = token_times[0] if token_times else end
    tpot = (end - ttft) / max(max_tokens - 1, 1)
    return ttft, tpot, end


async def run_level(url: str, concurrency: int, num_requests: int, max_tokens: int):
    rng = random.Random(concurrency)
    words = "the of and to in is was for on that with as by at from his her an it".split()
    prompts = [" ".join(rng.choice(words) for _ in range(rng.randrange(20, 60))) for _ in range(num_requests)]
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient(timeout=600) as client:
        async def worker(p):
            async with sem:
                results.append(await one_request(client, url, p, max_tokens))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - t0
    ttft = sorted(r[0] for r in results)
    tpot = [r[1] for r in results]
    e2e = [r[2] for r in results]
    p = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    print(
        f"| {concurrency} | {num_requests} | {statistics.mean(ttft):.2f} | {p(ttft, 0.99):.2f} | "
        f"{1000 * statistics.mean(tpot):.0f} | {statistics.mean(e2e):.2f} | {num_requests * max_tokens / wall:.1f} | {num_requests / wall:.2f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    print("| concurrency | requests | mean TTFT s | p99 TTFT s | mean TPOT ms | mean E2E s | output tok/s | req/s |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in map(int, args.concurrency.split(",")):
        asyncio.run(run_level(args.url, c, args.num_requests, args.max_tokens))


if __name__ == "__main__":
    main()
