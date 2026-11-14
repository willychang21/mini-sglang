"""Chapter 6 follow-up: overlap scheduling (SGLang ``event_loop_overlap``).

normal_loop:  schedule -> forward -> process, strictly in sequence.
overlap_loop: the forward of batch N+1 runs on the main thread while a CPU
thread processes batch N and schedules N+2.

The win is bounded by the CPU time per step (scheduling, metadata, result
processing) that can hide behind the forward, so it matters when that time is
a large fraction of a step: small models, small batches, fast GPUs. On a CPU
there is no asynchronous device: the forward already uses every core and
needs the GIL between ops, so CPU work running beside it slows it down. The
table therefore reports, per step, the forward time and the CPU time that was
*not* hidden ((wall - forward time) / steps).

    python benchmarks/bench_overlap.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time

from minisgl import SamplingParams
from minisgl.llm import LLM


def workload(n: int, max_tokens: int, seed: int = 0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(100, 400) for _ in range(rng.randrange(16, 64))] for _ in range(n)]
    return prompts, SamplingParams(max_tokens=max_tokens, ignore_eos=True)


def measure(model: str, overlap: bool, prompts, sp, repeats: int, **kw):
    llm = LLM(model, load_tokenizer_=False, device="cpu", disable_overlap=not overlap, cache_type="naive", **kw)
    llm.generate([prompts[0][:8]], SamplingParams(max_tokens=4, ignore_eos=True))  # warm-up
    fwd_times = []
    orig = llm.engine.forward_batch

    def timed(*a, **kw):
        t = time.perf_counter()
        out = orig(*a, **kw)
        fwd_times.append(time.perf_counter() - t)
        return out

    llm.engine.forward_batch = timed
    walls, outs, fwd, steps = [], None, [], []
    for _ in range(repeats):
        fwd_times.clear()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        walls.append(time.perf_counter() - t0)
        fwd.append(sum(fwd_times))
        steps.append(len(fwd_times))
    tokens = sum(len(o.token_ids) for o in outs)
    tpot = statistics.mean(
        statistics.mean(b - a for a, b in zip(o.token_times, o.token_times[1:])) for o in outs
    )
    llm.shutdown()
    i = walls.index(statistics.median(walls))
    per_step = (fwd[i] / steps[i], (walls[i] - fwd[i]) / steps[i])
    return walls[i], tokens, tpot, per_step, [o.token_ids for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)  # odd, so the median is a real run
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from conftest import make_tiny_qwen3  # noqa: E402

    tiny = make_tiny_qwen3(tempfile.mkdtemp(prefix="tiny-qwen3-"))
    cases = [
        ("tiny Qwen3 (CPU-overhead bound)", tiny, 16, 64),
        ("tiny Qwen3 (CPU-overhead bound)", tiny, 64, 64),
        (f"{os.path.basename(os.path.normpath(args.model))} (compute bound)", args.model, 4, 32),
        (f"{os.path.basename(os.path.normpath(args.model))} (compute bound)", args.model, 16, 32),
    ]
    print("| model | requests | loop | wall s | output tok/s | mean TPOT ms | forward ms/step "
          "| un-hidden CPU ms/step | speedup | identical |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for name, model, n, max_tokens in cases:
        prompts, sp = workload(n, max_tokens)
        base = None
        for overlap in (False, True):
            wall, tokens, tpot, (fwd, cpu), outs = measure(model, overlap, prompts, sp, args.repeats,
                                               max_total_tokens=8192, max_seq_len=512)
            if base is None:
                base = (wall, outs)
            print(f"| {name} | {n} | {'overlap' if overlap else 'normal'} | {wall:.2f} | {tokens / wall:.1f} "
                  f"| {tpot * 1e3:.1f} | {fwd * 1e3:.2f} | {cpu * 1e3:.2f} | {base[0] / wall:.2f}x | {outs == base[1]} |")


if __name__ == "__main__":
    main()
