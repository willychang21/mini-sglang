"""Part III Ch2 lab: how many tokens / requests fit in the KV cache at each precision.

Reproduces Figure 5 of Chapter 2 (Qwen3-32B on one 96 GB H20 with
mem_fraction_static=0.85) and lets you plug in any model / GPU. It also checks
the per-token formula against mini-sglang's real ``MHAKVCache`` allocation.

    bytes/token = 2 (K,V) * layers * kv_heads * head_dim * bytes_per_elem

    python labs/part3_ch02_kv_capacity.py
    python labs/part3_ch02_kv_capacity.py --gpu-gb 80 --mem-fraction 0.9 --req-len 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    params_b: float  # total parameters (billions)
    layers: int
    kv_heads: int
    head_dim: int
    embed_params_b: float  # embedding + lm_head parameters that stay 16-bit


# Shapes from each model's config.json on Hugging Face. Checkpoint sizes below are
# the ones quoted in the chapter (Qwen3-32B BF16 = 65.5 GB).
MODELS = {
    "Qwen3-0.6B": Model("Qwen3-0.6B", 0.60, 28, 8, 128, 0.16),  # tied embeddings
    "Qwen3-8B": Model("Qwen3-8B", 8.19, 36, 8, 128, 1.24),
    "Qwen3-32B": Model("Qwen3-32B", 32.8, 64, 8, 128, 1.56),
}

# Weight bytes/param for the *linear* layers, incl. scale metadata.
WEIGHT_SCHEMES = {"BF16": 2.0, "FP8": 1.0 + 4 / (128 * 128), "AWQ-INT4": 0.5 + 4 / 128}
KV_DTYPES = {"bf16": 2.0, "fp8_e4m3": 1.0, "fp4_e2m1": 0.5 + 1 / 16}  # fp4 KV with a fp8 scale per 16


def kv_bytes_per_token(m: Model, kv_bytes: float, tp: int = 1) -> float:
    # GQA: a rank stores max(1, kv_heads / tp) heads (replication once tp > kv_heads)
    heads = max(1, m.kv_heads // tp) if tp <= m.kv_heads else 1
    return 2 * m.layers * heads * m.head_dim * kv_bytes


def weight_gb(m: Model, scheme: str, checkpoint_override: dict | None = None) -> float:
    if checkpoint_override and (m.name, scheme) in checkpoint_override:
        return checkpoint_override[(m.name, scheme)]
    linear = m.params_b - m.embed_params_b
    return linear * WEIGHT_SCHEMES[scheme] + m.embed_params_b * 2


# Sizes the chapter uses for Figure 5 (GB). Using them reproduces its numbers.
CHAPTER_WEIGHTS = {("Qwen3-32B", "BF16"): 65.5, ("Qwen3-8B", "BF16"): 16.38,
                   ("Qwen3-8B", "FP8"): 9.44, ("Qwen3-8B", "AWQ-INT4"): 6.10}


def table(m: Model, gpu_gb: float, frac: float, req_len: int, override: dict | None) -> None:
    budget = gpu_gb * frac
    print(f"### {m.name}: budget {gpu_gb} GB x {frac} = {budget:.1f} GB, request length {req_len} tokens\n")
    print("| weights | KV dtype | weight GB | KV pool GB | KiB/token | max tokens | full-length requests |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for ws in WEIGHT_SCHEMES:
        w = weight_gb(m, ws, override)
        for kd, kb in KV_DTYPES.items():
            if kd == "fp4_e2m1" and ws != "AWQ-INT4":
                continue
            per_tok = kv_bytes_per_token(m, kb)
            pool = budget - w
            toks = max(0.0, pool * 1e9 / per_tok)
            print(f"| {ws} | {kd} | {w:.1f} | {pool:.1f} | {per_tok / 1024:.0f} | {toks / 1e3:,.0f}K | {int(toks // req_len)} |")
    print()


def check_against_minisgl() -> None:
    """The formula must match what mini-sglang actually allocates."""
    import torch

    from minisgl.distributed import set_tp_info
    from minisgl.kvcache import MHAKVCache

    set_tp_info(0, 1)
    m = MODELS["Qwen3-0.6B"]
    for dtype, nbytes in [(torch.bfloat16, 2), (torch.float8_e4m3fn, 1)]:
        pool = MHAKVCache(num_kv_heads=m.kv_heads, num_layers=m.layers, head_dim=m.head_dim,
                          num_pages=256, page_size=1, dtype=dtype, device=torch.device("cpu"))
        measured = pool.num_bytes / 256
        assert measured == kv_bytes_per_token(m, nbytes)
        print(f"mini-sglang MHAKVCache {str(dtype):>20}: {measured:,.0f} B/token, formula "
              f"{kv_bytes_per_token(m, nbytes):,.0f} B/token")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-gb", type=float, default=96.0)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--req-len", type=int, default=1536)
    ap.add_argument("--models", nargs="*", default=["Qwen3-32B", "Qwen3-8B", "Qwen3-0.6B"])
    args = ap.parse_args()
    check_against_minisgl()
    for name in args.models:
        table(MODELS[name], args.gpu_gb, args.mem_fraction, args.req_len, CHAPTER_WEIGHTS)
    m = MODELS["Qwen3-8B"]
    b = args.gpu_gb * args.mem_fraction
    gain = (b - weight_gb(m, "AWQ-INT4", CHAPTER_WEIGHTS)) / (b - weight_gb(m, "BF16", CHAPTER_WEIGHTS)) - 1
    assert 0.10 < gain < 0.25  # the chapter's "only about 16%"
    print(f"Qwen3-8B BF16 -> INT4 grows KV capacity by only {100 * gain:.0f}% (weights are a small share of memory)")
    print("\nTP note: with 8 KV heads, per-rank KV bytes/token for Qwen3-32B at bf16:")
    for tp in (1, 2, 4, 8, 16):
        print(f"  tp={tp:<2} -> {kv_bytes_per_token(MODELS['Qwen3-32B'], 2, tp) / 1024:.0f} KiB/token/rank")


if __name__ == "__main__":
    main()
