"""Chapters 6 & 7: continuous batching and paged KV memory.

1. Throughput: the same workload with ``max_running_req=1`` (one request at a
   time) vs continuous batching.
2. Concurrency under a fixed KV budget: paged allocation reserves only what each
   request needs; contiguous pre-allocation would reserve ``max_seq_len`` per
   request (Chapter 7, Figure 7-1).

    python benchmarks/bench_batching.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from _common import add_engine_args, describe, engine_kwargs
from minisgl import SamplingParams
from minisgl.llm import LLM


def workload(n: int, seed: int = 0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(100, 10000) for _ in range(rng.randrange(16, 96))] for _ in range(n)]
    sps = [SamplingParams(max_tokens=rng.randrange(16, 64), ignore_eos=True) for _ in range(n)]
    return prompts, sps


def run(llm: LLM, prompts, sps):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sps)
    elapsed = time.perf_counter() - t0
    tokens = sum(len(o.token_ids) for o in outs)
    ttft = [o.ttft for o in outs]
    return outs, elapsed, tokens, ttft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=16)
    add_engine_args(ap)
    args = ap.parse_args()
    print(describe(args))
    prompts, sps = workload(args.num_requests)

    print("## Throughput: sequential vs continuous batching\n")
    print("| mode | max_running_req | wall s | output tok/s | mean TTFT s | p90 TTFT s | mean batch size |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    results = {}
    for name, max_running in [("one at a time", 1), ("continuous batching", 64)]:
        llm = LLM(args.model, load_tokenizer_=False, **engine_kwargs(args), max_total_tokens=16384,
                  max_seq_len=1024, max_running_req=max_running, cache_type="naive")
        llm.generate([prompts[0][:8]], SamplingParams(max_tokens=2))  # warm-up
        llm.scheduler.stats.batch_sizes.clear()
        outs, elapsed, tokens, ttft = run(llm, prompts, sps)
        results[name] = [o.token_ids for o in outs]
        decode_sizes = llm.scheduler.stats.batch_sizes
        ttft_sorted = sorted(ttft)
        print(
            f"| {name} | {max_running} | {elapsed:.1f} | {tokens / elapsed:.1f} | {statistics.mean(ttft):.2f} "
            f"| {ttft_sorted[int(0.9 * len(ttft_sorted)) - 1]:.2f} | {statistics.mean(decode_sizes):.1f} |"
        )
        llm.shutdown()
    same = results["one at a time"] == results["continuous batching"]
    print(f"\nidentical outputs in both modes: {same}\n")

    print("## Concurrency under a fixed KV budget (paged vs contiguous)\n")
    budget, max_seq_len = 2048, 1024
    llm = LLM(args.model, load_tokenizer_=False, **engine_kwargs(args), max_total_tokens=budget,
              max_seq_len=max_seq_len, max_running_req=64, cache_type="naive", page_size=16)
    llm.generate(prompts, sps)
    peak = max(llm.scheduler.stats.batch_sizes)
    need = statistics.mean(len(p) + sp.max_tokens for p, sp in zip(prompts, sps))
    print(f"KV budget = {budget} tokens, max_seq_len = {max_seq_len}, mean request needs {need:.0f} tokens")
    print(f"- contiguous pre-allocation (reserve max_seq_len each): {budget // max_seq_len} concurrent requests")
    print(f"- paged allocation (page_size=16, measured peak batch):  {peak} concurrent requests")


if __name__ == "__main__":
    main()
