"""Part I Ch4 lab: how much memory does the KV cache take?

What it shows
  1. bytes/token = 2 (K,V) x layers x kv_heads x head_dim x dtype_bytes, checked against
     the tutorial's Qwen3-4B example (14,745,600 B for T=100, 0.56 -> 4.50 GiB for 4K -> 32K).
  2. Ch4 exercise 3: Qwen2.5-1.5B, T=200 prompt tokens, l=100 generated tokens.
  3. MHA vs GQA vs MQA for the same model: KV size scales with the number of KV heads.
  4. How many requests fit in a KV budget, and how much of it a contiguous
     "reserve max_seq_len per request" scheme wastes compared with paged allocation
     (Ch4 exercise 4).
  5. The formula against the engine: mini-sglang's MHAKVCache.num_bytes for the tiny
     model (and the Qwen3-0.6B shape) equals per-token bytes x token slots.

    python labs/part1_ch04_kv_memory.py [--model tiny|real] [--budget-gib 16]
"""

from __future__ import annotations

import random

from _common import TINY_CONFIG, human_bytes, lab_parser, make_engine, quiet_logs, resolve_model, section, table

MiB, GiB = 2**20, 2**30

# Config numbers checked against the HuggingFace config.json files (Nov 2026).
MODELS = {
    # name: layers, q_heads, kv_heads, head_dim, hidden, params (for the weights comparison)
    "Qwen2.5-1.5B": dict(L=28, hq=12, hkv=2, d=128, hidden=1536, params=1.54e9),  # head_dim = 1536 / 12
    "Qwen3-0.6B": dict(L=28, hq=16, hkv=8, d=128, hidden=1024, params=0.6e9),
    "Qwen3-4B-Thinking-2507": dict(L=36, hq=32, hkv=8, d=128, hidden=2560, params=4.0e9),
    "LLaMA-7B (Ch2, MHA)": dict(L=32, hq=32, hkv=32, d=128, hidden=4096, params=6.7e9),
}


def kv_bytes_per_token(L: int, hkv: int, d: int, s: int = 2) -> int:
    """Ch4 Sec.4: one K and one V vector per KV head, per layer, per token."""
    return 2 * L * hkv * d * s


def kv_bytes(tokens: int, L: int, hkv: int, d: int, s: int = 2) -> int:
    return tokens * kv_bytes_per_token(L, hkv, d, s)


def per_tok(name: str, s: int = 2) -> int:
    m = MODELS[name]
    return kv_bytes_per_token(m["L"], m["hkv"], m["d"], s)


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--budget-gib", type=float, default=16.0, help="KV cache budget for section 4")
    p.add_argument("--max-seq-len", type=int, default=32768, help="contiguous reservation per request")
    p.add_argument("--page-size", type=int, default=16)
    args = p.parse_args(argv)
    quiet_logs()

    # ------------------------------------------------------------------ 1
    section("1. bytes per token, and the tutorial's Qwen3-4B example")
    rows = []
    for name, m in MODELS.items():
        b = per_tok(name)
        rows.append([name, m["L"], m["hkv"], m["d"], f"{b:,}", f"{b / 1024:.0f} KiB"])
    print(table(rows, ["model", "layers", "kv_heads", "head_dim", "bytes/token (bf16)", ""]))
    print()
    q4 = "Qwen3-4B-Thinking-2507"
    t100, t101, t300 = (kv_bytes(t, 36, 8, 128) for t in (100, 101, 300))
    rows = [
        ["prefill T=100", f"{t100:,} B = {t100 / MiB:.2f} MiB", "14,745,600 B ~ 14.06 MB"],
        ["one decode token", f"{per_tok(q4) / MiB:.3f} MiB", "0.141 MB"],
        ["after 1 decode step", f"{t101 / MiB:.2f} MiB", "14.20 MB"],
        ["after l=200", f"{t300 / MiB:.2f} MiB", "42.19 MB"],
        ["4K tokens", f"{kv_bytes(4096, 36, 8, 128) / GiB:.4f} GiB", "~0.56 GB"],
        ["32K tokens", f"{kv_bytes(32768, 36, 8, 128) / GiB:.2f} GiB", "~4.50 GB"],
        ["8 x 32K tokens", f"{8 * kv_bytes(32768, 36, 8, 128) / GiB:.0f} GiB", "~36 GB"],
        ["weights 2 B x 4e9", f"{2 * 4e9 / GiB:.2f} GiB", "7.45 GB"],
    ]
    print(table(rows, ["Qwen3-4B case", "computed", "tutorial"]))
    print("-> the tutorial's 'MB'/'GB' are binary units (MiB/GiB): 14,745,600 / 2^20 = 14.06.")
    assert t100 == 14_745_600 and round(t101 / MiB, 2) == 14.20 and round(t300 / MiB, 2) == 42.19
    assert round(kv_bytes(4096, 36, 8, 128) / GiB, 2) == 0.56 and kv_bytes(32768, 36, 8, 128) == 4.5 * GiB
    assert per_tok("LLaMA-7B (Ch2, MHA)") == 512 * 1024  # Ch2: "~0.5 MB per token"

    # ------------------------------------------------------------------ 2
    section("2. Ch4 exercise 3: Qwen2.5-1.5B, T=200, l=100 (bf16)")
    q = MODELS["Qwen2.5-1.5B"]
    tok = per_tok("Qwen2.5-1.5B")
    pre, inc, tot = kv_bytes(200, q["L"], q["hkv"], q["d"]), tok, kv_bytes(300, q["L"], q["hkv"], q["d"])
    w = q["params"] * 2
    print(f"per token = 2 x {q['L']} layers x {q['hkv']} kv_heads x {q['d']} head_dim x 2 B = {tok:,} B")
    print(table([
        ["end of prefill (T=200)", f"{pre:,}", f"{pre / MiB:.2f} MiB"],
        ["each decode token", f"{inc:,}", f"{inc / 1024:.0f} KiB"],
        ["end of generation (T+l=300)", f"{tot:,}", f"{tot / MiB:.2f} MiB"],
        ["model weights (1.54e9 params x 2 B)", f"{w:,.0f}", f"{w / GiB:.2f} GiB"],
    ], ["quantity", "bytes", ""]))
    print(f"KV at the end = {tot / w * 100:.2f}% of the weights; a single request needs "
          f"{w / tok:,.0f} tokens before its KV outweighs the weights.")
    assert tok == 28_672 and pre == 5_734_400 and tot == 8_601_600 and tot == pre + 100 * inc

    # ------------------------------------------------------------------ 3
    section("3. MHA vs GQA vs MQA (same layers/head_dim, only the number of KV heads changes)")
    rows = []
    for name in ("Qwen2.5-1.5B", "Qwen3-4B-Thinking-2507"):
        m = MODELS[name]
        for label, hkv in (("MHA", m["hq"]), ("GQA (actual)", m["hkv"]), ("MQA", 1)):
            b = kv_bytes_per_token(m["L"], hkv, m["d"])
            rows.append([name, label, hkv, f"{b / 1024:.0f} KiB", f"{kv_bytes(32768, m['L'], hkv, m['d']) / GiB:.2f} GiB",
                         f"{m['hq'] // hkv}x"])
    print(table(rows, ["model", "attention", "kv_heads", "per token", "32K-token request", "Q heads / KV head"]))
    m = MODELS["Qwen2.5-1.5B"]
    assert kv_bytes_per_token(m["L"], m["hq"], m["d"]) == 6 * tok  # MHA would be 6x larger

    # ------------------------------------------------------------------ 4
    budget = int(args.budget_gib * GiB)
    section(f"4. a {args.budget_gib:g} GiB KV budget, Qwen2.5-1.5B: contiguous max-length reservation vs paged")
    rng = random.Random(0)  # deterministic synthetic workload: chat-like lengths
    lengths = [rng.randint(200, 3000) + rng.randint(50, 1000) for _ in range(10_000)]
    ps = args.page_size
    capacity_tokens = budget // tok
    contiguous_fit = capacity_tokens // args.max_seq_len
    used, paged_fit = 0, 0
    for n in lengths:  # admit requests in order until the pool is full
        need = -(-n // ps) * ps
        if used + need > capacity_tokens:
            break
        used, paged_fit = used + need, paged_fit + 1
    actual_contig = sum(lengths[:contiguous_fit])
    rows = [
        [f"contiguous, reserve max_seq_len={args.max_seq_len}", contiguous_fit,
         f"{actual_contig / (contiguous_fit * args.max_seq_len) * 100:.1f}%"],
        [f"paged, page_size={ps}", paged_fit, f"{sum(lengths[:paged_fit]) / used * 100:.1f}%"],
    ]
    print(f"budget = {capacity_tokens:,} token slots; request length (prompt+output) {min(lengths)}..{max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.0f}")
    print(table(rows, ["allocation", "concurrent requests", "slots holding real KV"]))
    print(f"-> paged fits {paged_fit / contiguous_fit:.1f}x more requests; its only waste is the last, partly"
          f" filled page (< {ps} tokens per request).")
    full_ctx = budget // (args.max_seq_len * tok)
    print(f"-> even at full {args.max_seq_len}-token context every request costs {human_bytes(args.max_seq_len * tok)};"
          f" {full_ctx} such requests fit.")
    assert paged_fit > 5 * contiguous_fit
    assert sum(lengths[:paged_fit]) / used > 0.99

    # ------------------------------------------------------------------ 5
    path = resolve_model(args.model)
    section(f"5. formula vs. mini-sglang's MHAKVCache ({args.model} model)")
    import torch

    from minisgl.config import ModelConfig
    from minisgl.kvcache import create_kvcache_pool

    eng = make_engine(path, max_total_tokens=512, page_size=4)
    cfg = eng.model_config
    slots = eng.num_pages * eng.page_size
    b = kv_bytes_per_token(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, eng.kv_cache._kv_buffer.element_size())
    rows = [[f"Engine({args.model}), {eng.dtype}", tuple(eng.kv_cache._kv_buffer.shape), b, slots,
             eng.kv_cache.num_bytes, b * slots]]
    assert eng.kv_cache.num_bytes == b * slots
    if args.model == "tiny":
        assert (cfg.num_layers, cfg.num_kv_heads, cfg.head_dim) == (
            TINY_CONFIG["num_hidden_layers"], TINY_CONFIG["num_key_value_heads"], TINY_CONFIG["head_dim"])
    # The real Qwen3-0.6B *shape* in bf16 (pool only; no weights are loaded).
    import os

    from _common import REAL_MODEL
    if os.path.exists(os.path.join(REAL_MODEL, "config.json")):
        mc = ModelConfig.from_pretrained(REAL_MODEL)
        pool = create_kvcache_pool(mc, num_pages=64, page_size=1, dtype=torch.bfloat16, device=torch.device("cpu"))
        b06 = kv_bytes_per_token(mc.num_layers, mc.num_kv_heads, mc.head_dim, 2)
        rows.append(["Qwen3-0.6B shape, bf16", tuple(pool._kv_buffer.shape), b06, 64, pool.num_bytes, b06 * 64])
        assert pool.num_bytes == b06 * 64 and b06 == per_tok("Qwen3-0.6B") == 114_688
    print(table(rows, ["pool", "_kv_buffer shape [2,L,pages,page,kv_heads,dim]", "B/token", "slots",
                       "num_bytes", "formula"]))
    print("-> the engine allocates the whole pool up front (python/minisgl/engine/engine.py);")
    print("   the scheduler hands out slots from it (scheduler/cache.py), so the formula is exact.")


if __name__ == "__main__":
    main()
