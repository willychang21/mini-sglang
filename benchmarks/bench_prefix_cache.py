"""Chapter 8: RadixAttention prefix caching.

Requests share a long "system prompt" and differ in a short question, the
common case for chat services. With the radix cache only the first request
computes the shared prefix; later ones reuse its K/V.

    python benchmarks/bench_prefix_cache.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from minisgl import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--system-len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = random.Random(0)
    system = [rng.randrange(100, 10000) for _ in range(args.system_len)]
    prompts = [system + [rng.randrange(100, 10000) for _ in range(rng.randrange(8, 32))] for _ in range(args.num_requests)]
    sp = SamplingParams(max_tokens=8, ignore_eos=True)

    print(f"{args.num_requests} requests, shared prefix {args.system_len} tokens\n")
    print("| cache | prompt tokens | computed prefill tokens | cache hit rate | wall s | mean TTFT s (sequential arrival) |")
    print("|---|---:|---:|---:|---:|---:|")
    outputs = {}
    for cache_type in ["naive", "radix"]:
        llm = LLM(args.model, load_tokenizer_=False, device=args.device, max_total_tokens=16384,
                  max_seq_len=2048, cache_type=cache_type)
        ttfts, outs = [], []
        t0 = time.perf_counter()
        for p in prompts:  # one at a time, like users arriving one after another
            o = llm.generate([p], sp)[0]
            ttfts.append(o.ttft)
            outs.append(o.token_ids)
        wall = time.perf_counter() - t0
        st = llm.scheduler.prefill_manager.stats
        computed = llm.scheduler.stats.prefill_tokens
        print(
            f"| {cache_type} | {st.prompt_tokens} | {computed} | {st.cached_tokens / st.prompt_tokens:.1%} "
            f"| {wall:.1f} | {statistics.mean(ttfts):.3f} |"
        )
        outputs[cache_type] = outs
        llm.shutdown()
    print(f"\nidentical outputs: {outputs['naive'] == outputs['radix']}")


if __name__ == "__main__":
    main()
