"""Part III Ch5 lab: KV-transfer size and time for prefill-decode disaggregation.

    S_transfer = prompt_tokens * bytes_per_token(model, kv_dtype) - prefix already on D
    T_transfer = S_transfer / B_effective

Prints a table over prompt lengths and links, answers the chapter's Exercise 3,
and shows how sharing the link between concurrent transfers inflates TTFT. It
also measures the actual per-request K/V bytes mini-sglang would have to ship,
straight from its page table / KV pool (so the formula is checked against code).

    python labs/part3_ch05_pd_transfer.py
    python labs/part3_ch05_pd_transfer.py --model qwen3-32b --links 25 50 400
"""

from __future__ import annotations

from _common import REAL_MODEL  # labs/_common.py (MINISGL_TEST_MODEL or /tmp/models/Qwen3-0.6B)

import argparse

MODELS = {  # layers, kv_heads, head_dim (full-attention KV); MLA stores one latent per token
    "qwen3-0.6b": dict(layers=28, kv_heads=8, head_dim=128),
    "qwen3-8b": dict(layers=36, kv_heads=8, head_dim=128),
    "qwen3-32b": dict(layers=64, kv_heads=8, head_dim=128),
    # DeepSeek-V3 MLA: kv_lora_rank 512 + rope 64 per layer, K and V share the latent
    "deepseek-v3 (MLA)": dict(layers=61, kv_heads=1, head_dim=576, mla=True),
}
DTYPES = {"bf16": 2.0, "fp8": 1.0}


def bytes_per_token(m: dict, kv_bytes: float) -> float:
    kv = 1 if m.get("mla") else 2
    return kv * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes


def measured_minisgl(model_path: str, n: int) -> None:
    from minisgl import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(model_path, load_tokenizer_=False, device="cpu", max_total_tokens=4096, max_seq_len=2048,
              dtype="bfloat16")
    req_tokens = [(i * 31) % 20000 + 100 for i in range(n)]
    llm.generate([req_tokens], SamplingParams(max_tokens=1))
    # what P would send: for every layer, K and V rows of the request's slots
    kv = llm.engine.kv_cache
    slots = llm.engine.page_table[0, :n]  # row 0 was the request's row (freed, but slots still recorded)
    nbytes = sum(kv.flat_k_cache(l)[slots].nbytes + kv.flat_v_cache(l)[slots].nbytes for l in range(kv.num_layers))
    assert nbytes == n * bytes_per_token(MODELS["qwen3-0.6b"], 2)
    print(f"mini-sglang, Qwen3-0.6B bf16, {n}-token prompt: K/V to ship = {nbytes / 2**20:.1f} MiB "
          f"(formula: {n * bytes_per_token(MODELS['qwen3-0.6b'], 2) / 2**20:.1f} MiB)\n")
    llm.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(MODELS))
    ap.add_argument("--kv-dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--links", type=float, nargs="*", default=[5, 10, 20, 50],
                    help="effective bandwidth, GiB/s (e.g. ~5 TCP, ~20-45 one 400G RDMA NIC, ~400+ NVLink)")
    ap.add_argument("--lengths", type=int, nargs="*", default=[1024, 4096, 16384, 65536, 131072])
    ap.add_argument("--hf-model", default=REAL_MODEL)
    ap.add_argument("--skip-measure", action="store_true")
    args = ap.parse_args()
    if not args.skip_measure:
        measured_minisgl(args.hf_model, 512)

    m = MODELS[args.model]
    bpt = bytes_per_token(m, DTYPES[args.kv_dtype])
    print(f"### {args.model}, KV {args.kv_dtype}: {bpt / 1024:.0f} KiB per token\n")
    print("| prompt tokens | KV GiB | " + " | ".join(f"{b:g} GiB/s" for b in args.links) + " |")
    print("|---:|---:|" + "---:|" * len(args.links))
    for n in args.lengths:
        s = n * bpt / 2**30
        print(f"| {n} | {s:.2f} | " + " | ".join(f"{1e3 * s / b:.0f} ms" for b in args.links) + " |")

    print("\n### Exercise 3: 2 GiB at 10 GiB/s")
    print(f"  T = 2/10 = {1e3 * 2 / 10:.0f} ms; with contention halving bandwidth: 2/5 = {1e3 * 2 / 5:.0f} ms")

    print("\n### Link sharing: k simultaneous 16K-token transfers over one 20 GiB/s link")
    s = 16384 * bpt / 2**30
    for k in (1, 2, 4, 8):
        print(f"  k={k}: each waits ~{1e3 * s * k / 20:.0f} ms for its KV (fair share), "
              f"vs {1e3 * s / 20:.0f} ms alone")

    print("\n### Prefix already on D (only the missing suffix moves)")
    for hit in (0.0, 0.5, 0.9):
        print(f"  64K prompt, {int(hit * 100)}% prefix resident on D: "
              f"{(1 - hit) * 65536 * bpt / 2**30:.2f} GiB to send")


if __name__ == "__main__":
    main()
