"""Chapter 4: generation with vs without a KV cache.

Without a cache every step re-runs the whole prefix, so the time per output
token (TPOT) grows with the sequence length. With the cache each decode step
processes one new token.

    python benchmarks/bench_kv_cache.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time

import torch

from _common import add_engine_args, describe, engine_kwargs
from minisgl import SamplingParams
from minisgl.config import EngineConfig
from minisgl.engine import Engine
from minisgl.llm import generate_without_kv_cache
from minisgl.speculative import autoregressive_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-lens", default="16,64,128")
    add_engine_args(ap)
    args = ap.parse_args()
    print(describe(args))

    engine = Engine(EngineConfig(args.model, **engine_kwargs(args), max_total_tokens=4096, max_seq_len=2048))
    prompt = torch.randint(100, 10000, (args.prompt_len,), generator=torch.Generator().manual_seed(0)).tolist()
    print(f"prompt_len={args.prompt_len}")
    print(f"| output tokens | no cache: total s | no cache: TPOT ms | KV cache: total s | KV cache: TPOT ms | speedup | same output |")
    print("|---:|---:|---:|---:|---:|---:|:---:|")
    for n in map(int, args.output_lens.split(",")):
        t0 = time.perf_counter()
        a = generate_without_kv_cache(engine, prompt, n)
        t_naive = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = autoregressive_generate(engine, prompt, SamplingParams(max_tokens=n, ignore_eos=True))
        t_cache = time.perf_counter() - t0
        print(
            f"| {n} | {t_naive:.2f} | {1000 * t_naive / n:.1f} | {t_cache:.2f} | {1000 * t_cache / n:.1f} "
            f"| {t_naive / t_cache:.1f}x | {a == b} |"
        )


if __name__ == "__main__":
    main()
