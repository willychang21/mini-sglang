"""Part IV Ch1 Ex.2 / Ch2 Ex.3 lab: a credible before/after comparison.

Changes exactly ONE setting, keeps everything else fixed (model, prompts, output
length, seed, warmup, cache state), repeats each configuration N times, reports
mean and range, and checks correctness (greedy outputs must be identical). The
profiler is NOT enabled here: profile to explain, benchmark without it to measure.

    python labs/part4/ab_compare.py --setting cache_type --a naive --b radix
    python labs/part4/ab_compare.py --setting prefill_budget --a 128 --b 1024 --repeats 3
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from minisgl import SamplingParams
from minisgl.llm import LLM


def workload(n: int, shared: int, seed: int):
    rng = random.Random(seed)
    system = [rng.randrange(100, 20000) for _ in range(shared)]
    return [system + [rng.randrange(100, 20000) for _ in range(rng.randrange(16, 64))] for _ in range(n)]


def run(model: str, setting: str, value, prompts, max_tokens: int, repeats: int):
    kw = dict(load_tokenizer_=False, device="cpu", max_total_tokens=8192, max_seq_len=2048)
    kw[setting] = value
    rows, outputs = [], None
    for _ in range(repeats):
        llm = LLM(model, **kw)  # fresh engine each repeat: identical (empty) cache state
        llm.generate([[5] * 32], SamplingParams(max_tokens=2, ignore_eos=True))  # identical warmup
        sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t0
        ttft = statistics.mean(o.ttft for o in outs)
        tpot = statistics.mean((o.token_times[-1] - o.ttft) / (len(o.token_ids) - 1) for o in outs)
        rows.append((wall, ttft, tpot, sum(len(o.token_ids) for o in outs) / wall))
        toks = [o.token_ids for o in outs]
        assert outputs is None or toks == outputs, "non-deterministic output across repeats"
        outputs = toks
        llm.shutdown()
        del llm
    return rows, outputs


def fmt(xs, scale=1.0):
    return f"{statistics.mean(xs) * scale:.2f} [{min(xs) * scale:.2f}, {max(xs) * scale:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/models/Qwen3-0.6B")
    ap.add_argument("--setting", default="cache_type")
    ap.add_argument("--a", default="naive")
    ap.add_argument("--b", default="radix")
    ap.add_argument("--num-requests", type=int, default=6)
    ap.add_argument("--shared-prefix", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    cast = lambda v: int(v) if v.isdigit() else v  # noqa: E731
    prompts = workload(args.num_requests, args.shared_prefix, seed=0)
    res = {v: run(args.model, args.setting, cast(v), prompts, args.max_tokens, args.repeats) for v in (args.a, args.b)}
    print(f"\n{args.num_requests} requests, shared prefix {args.shared_prefix}, {args.max_tokens} output tokens, "
          f"{args.repeats} repeats; mean [min, max]\n")
    print(f"| {args.setting} | wall s | mean TTFT s | mean TPOT ms | output tok/s |")
    print("|---|---:|---:|---:|---:|")
    for v, (rows, _) in res.items():
        cols = list(zip(*rows))
        print(f"| {v} | {fmt(cols[0])} | {fmt(cols[1])} | {fmt(cols[2], 1e3)} | {fmt(cols[3])} |")
    same = res[args.a][1] == res[args.b][1]
    print(f"\ncorrectness: greedy outputs identical between '{args.a}' and '{args.b}': {same}")


if __name__ == "__main__":
    main()
