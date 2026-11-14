"""Part I Ch2 + Ch3 lab: roofline model, arithmetic intensity, prefill vs. decode.

What it shows
  1. Ridge points of the GPUs the tutorial cites (peak FLOP/s / HBM bandwidth).
  2. The tutorial's LLaMA-7B numbers recomputed: prefill FLOPs, decode FLOPs and bytes,
     AI_prefill = N exactly, AI_decode ~= 1 (Ch2 Sec. 4-5).
  3. Qwen3-0.6B (shapes from config.json): AI of every linear layer as a GEMM
     [B, C] x [C, C'] for B = tokens in the step (prefill tokens or decode batch size),
     and the batch size at which decode linears become compute-bound.
  4. A whole decode step (linears + LM head + attention over S cached tokens): the KV
     reads grow with B as fast as the FLOPs do, so long-context decode can stay
     memory-bound at *any* batch size.
  5. Measured on this CPU: fp32 matmul [N, C] x [C, C] for N = 1 ... 512.
     N = 1 is a GEMV: time is set by reading the weight; N = 512 reuses each weight
     element 512 times and runs at a much higher FLOP/s.

    python labs/part1_ch02_roofline.py [--model real|tiny|<path>] [--skip-measure]

``--model`` only selects which config.json is analysed (default: the real Qwen3-0.6B
config, falling back to built-in Qwen3-0.6B constants); no weights are loaded.
"""

from __future__ import annotations

import json
import math
import os
import time

from _common import REAL_MODEL, lab_parser, section, table

# ------------------------------------------------------------------ hardware
# Peak dense BF16 Tensor Core FLOP/s and HBM bandwidth, *as quoted by the tutorial*.
GPUS = [
    # name, peak FLOP/s, bytes/s, source
    ("A100 40GB", 312e12, 1.6e12, "Ch2 Sec.4.3/5.2: 312 TFLOPS, 1.6 TB/s"),
    ("A100 80GB PCIe", 312e12, 2.0e12, "Ch3 Sec.2.2.5/3.3.1: 312 TFLOPS, ~2 TB/s"),
    ("H100 SXM", 990e12, 3.35e12, "Ch2 Sec.3.1: ~990 TFLOPS; 3.35 TB/s is NVIDIA's datasheet, not the tutorial"),
]

# LLaMA-7B as used by Ch2 (d=4096, L=32, FFN = 8/3 d, MHA, bf16).
LLAMA7B = dict(L=32, d=4096)

# Qwen3-0.6B fallback (identical to /tmp/models/Qwen3-0.6B/config.json).
QWEN3_06B = dict(
    hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, vocab_size=151936,
)


def ridge(peak: float, bw: float) -> float:
    return peak / bw


def gemm_ai(m: int, k: int, n: int, dtype_bytes: int = 2) -> float:
    """AI of C[m,n] = A[m,k] @ B[k,n]: 2mkn FLOPs over (mk + kn + mn) elements moved."""
    return 2 * m * k * n / ((m * k + k * n + m * n) * dtype_bytes)


# ------------------------------------------------- Ch2's analytical formulas
def llama_prefill_flops(N: int, L: int, d: int) -> float:
    return L * (24 * N * d * d + 4 * N * N * d)


def llama_decode_flops(S: int, L: int, d: int) -> float:
    return L * (24 * d * d + 4 * S * d)


def ai_prefill(N: int, d: int) -> float:  # Ch2: (24Nd^2 + 4N^2 d) / (L(12d^2 + 2Nd) * 2) per layer
    return (24 * N * d * d + 4 * N * N * d) / ((12 * d * d + 2 * N * d) * 2)


def ai_decode(S: int, d: int) -> float:
    return (24 * d * d + 4 * S * d) / ((12 * d * d + 2 * d + 2 * S * d) * 2)


# ------------------------------------------------------- Qwen3 decode step
def linear_shapes(cfg: dict) -> list:
    """(name, in, out) of the four weight GEMMs in one decoder layer, as mini-sglang
    builds them (python/minisgl/models/utils.py: RopeAttn, GatedMLP)."""
    C, I, D = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    return [
        ("qkv_proj", C, (hq + 2 * hkv) * D),
        ("o_proj", hq * D, C),
        ("gate_up_proj", C, 2 * I),
        ("down_proj", I, C),
    ]


def decode_step(cfg: dict, B: int, S: int, dtype_bytes: int = 2):
    """FLOPs and HBM bytes of one decode step: B requests, each attending to S cached tokens.

    Bytes = all weights once (shared by the batch) + every request's own KV history.
    Activations are ignored (they are O(B*C), tiny next to weights and KV).
    """
    L, C, V, D = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"], cfg["head_dim"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    p_lin = sum(i * o for _, i, o in linear_shapes(cfg))  # per layer
    flops = B * (2 * L * p_lin + 2 * V * C + L * 4 * S * hq * D)  # + QK^T and PV per head
    weight_bytes = (L * p_lin + V * C) * dtype_bytes  # LM head (tied embedding) read once
    kv_bytes = B * (S + 1) * L * 2 * hkv * D * dtype_bytes  # read S, write 1
    return flops, weight_bytes + kv_bytes


def decode_ai_limit(cfg: dict, S: int) -> float:
    """AI of a decode step as B -> infinity: weights are amortised away, KV is not."""
    f1, b1 = decode_step(cfg, 1, S)
    f2, b2 = decode_step(cfg, 2, S)
    return (f2 - f1) / (b2 - b1)


def min_batch(ai_of_b, target: float, b_max: int = 1 << 20):
    if ai_of_b(b_max) < target:
        return None
    lo, hi = 1, b_max
    while lo < hi:
        mid = (lo + hi) // 2
        if ai_of_b(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def load_cfg(name: str) -> tuple:
    path = REAL_MODEL if name == "real" else name
    if name == "tiny":
        from _common import TINY_CONFIG

        return dict(TINY_CONFIG), "tiny lab model (labs/_common.py TINY_CONFIG)"
    cfg_file = os.path.join(path, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        cfg.setdefault("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        return cfg, cfg_file
    return dict(QWEN3_06B), "built-in Qwen3-0.6B constants (config.json not found)"


# ------------------------------------------------------------- CPU measure
def measure_cpu(C: int, ns, target_bytes: int = 400 << 20):
    """fp32 [N, C] x [C, C] on this CPU. Several distinct weight matrices are cycled
    so that their total size exceeds the last-level cache and every GEMV really
    streams its weight from DRAM."""
    import torch

    torch.manual_seed(0)
    n_mats = max(2, math.ceil(target_bytes / (C * C * 4)))
    weights = [torch.randn(C, C) for _ in range(n_mats)]
    rows = []
    for n in ns:
        x = torch.randn(n, C)
        for w in weights:  # warm-up pass
            torch.mm(x, w)
        reps = 3 if n <= 64 else 1
        t0 = time.perf_counter()
        for _ in range(reps):
            for w in weights:
                torch.mm(x, w)
        dt = (time.perf_counter() - t0) / (reps * n_mats)
        flops = 2 * n * C * C
        nbytes = (C * C + 2 * n * C) * 4
        rows.append((n, dt, flops / dt, nbytes / dt, flops / nbytes))
    return rows, n_mats


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0], default_model="real")
    p.add_argument("--hidden", type=int, default=2048, help="C for the CPU matmul measurement")
    p.add_argument("--skip-measure", action="store_true")
    args = p.parse_args(argv)

    # ------------------------------------------------------------------ 1
    section("1. ridge point = peak FLOP/s / memory bandwidth (numbers quoted by the tutorial)")
    rows = [[n, f"{pk / 1e12:.0f}", f"{bw / 1e12:.2f}", f"{ridge(pk, bw):.0f}", src] for n, pk, bw, src in GPUS]
    print(table(rows, ["GPU", "BF16 TFLOP/s", "TB/s", "ridge FLOP/B", "source"]))
    assert round(ridge(312e12, 1.6e12)) == 195  # Ch2 Sec.5.2: "~195 FLOPs/Byte"
    r_a100 = ridge(312e12, 1.6e12)

    # ------------------------------------------------------------------ 2
    section("2. Ch2's LLaMA-7B worked example, recomputed")
    L, d = LLAMA7B["L"], LLAMA7B["d"]
    pf = llama_prefill_flops(3000, L, d)
    df = llama_decode_flops(3000, L, d)
    kv_tok = 2 * L * d * 2  # K and V, every layer, d values each, bf16 (MHA)
    kv_3000 = 3000 * kv_tok
    w_bytes = 14e9
    dec_bytes = w_bytes + kv_3000
    rows = [
        ["prefill FLOPs, N=3000", f"{pf / 1e12:.1f} TFLOP", "43.4 TFLOPs"],
        ["  -> time at 312 TFLOP/s", f"{pf / 312e12 * 1e3:.0f} ms", "0.14 s"],
        ["decode FLOPs/token, S=3000", f"{df / 1e9:.1f} GFLOP", "14.4 GFLOPs"],
        ["  -> time at 312 TFLOP/s", f"{df / 312e12 * 1e3:.3f} ms", "0.05 ms"],
        ["KV bytes per token (MHA)", f"{kv_tok} B = {kv_tok / 2**20:.2f} MiB", "0.5 MB"],
        ["KV bytes, 3000 tokens", f"{kv_3000 / 1e9:.2f} GB", "1.5 GB"],
        ["decode bytes/token (14 GB W + KV)", f"{dec_bytes / 1e9:.1f} GB", "15.5 GB"],
        ["  -> time at 1.6 TB/s", f"{dec_bytes / 1.6e12 * 1e3:.1f} ms", "9.7 ms"],
        ["AI_prefill, N=100", f"{ai_prefill(100, d):.1f} FLOP/B", "~100"],
        ["AI_decode, S=100", f"{ai_decode(100, d):.3f} FLOP/B", "~1"],
    ]
    print(table(rows, ["quantity", "computed", "tutorial"]))
    assert abs(pf / 1e12 - 43.4) < 0.1 and abs(df / 1e9 - 14.4) < 0.1
    assert kv_tok == 2**19 and abs(dec_bytes / 1.6e12 * 1e3 - 9.7) < 0.1
    # With the tutorial's byte model, AI_prefill is *exactly* N: (24Nd^2+4N^2d)/(24d^2+4Nd) = N.
    for N in (1, 100, 3000):
        assert math.isclose(ai_prefill(N, d), N)
    assert 0.95 < ai_decode(100, d) < 1.0 and 0.99 < ai_decode(100_000, d) < 1.0
    print(f"-> AI_prefill(N) = N exactly, so LLaMA-7B prefill crosses the A100-40GB ridge at N >= "
          f"{math.ceil(r_a100)} tokens; at N=100 (AI 100 < 195) it is still memory-bound, as Ch2 says.")
    print("-> AI_decode -> 1 as S grows: every byte of weights/KV is used for ~1 FLOP (GEMV).")

    # ------------------------------------------------------------------ 3
    cfg, src = load_cfg(args.model)
    section(f"3. per-layer linear GEMMs of the analysed model ({src})")
    shapes = linear_shapes(cfg)
    bs = [1, 8, 64, 256, 1024, 4096]
    rows = [[name, f"{i}x{o}"] + [f"{gemm_ai(b, i, o):.1f}" for b in bs] for name, i, o in shapes]
    print(table(rows, ["layer", "weight [in x out]"] + [f"AI B={b}" for b in bs]))
    print("B = tokens in the step: prompt tokens for prefill, batch size for decode (1 token/request).")
    for name, i, o in shapes:
        assert gemm_ai(1, i, o) < 1.0  # GEMV: < 1 FLOP per byte in bf16
        assert gemm_ai(4096, i, o) > r_a100  # a long prefill is compute-bound
    rows = []
    for gname, pk, bw, _ in GPUS:
        r = ridge(pk, bw)
        need = [min_batch(lambda b, i=i, o=o: gemm_ai(b, i, o), r) for _, i, o in shapes]
        rows.append([gname, f"{r:.0f}"] + need)
    print()
    print("smallest B with AI >= ridge (decode linears become compute-bound):")
    print(table(rows, ["GPU", "ridge"] + [n for n, _, _ in shapes]))
    print("-> roughly B ~ ridge: in bf16 a GEMM with B << C does ~B FLOPs per weight byte.")

    # ------------------------------------------------------------------ 4
    section("4. whole decode step: weights are shared by the batch, KV is not")
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    contexts = [0, 512, 4096, 32768]
    rows = []
    for S in contexts:
        lim = decode_ai_limit(cfg, S)
        need = min_batch(lambda b, S=S: (lambda f, by: f / by)(*decode_step(cfg, b, S)), r_a100)
        ai1 = (lambda f, by: f / by)(*decode_step(cfg, 1, S))
        ai64 = (lambda f, by: f / by)(*decode_step(cfg, 64, S))
        rows.append([S, f"{ai1:.2f}", f"{ai64:.1f}", f"{lim:.1f}" if S else "inf",
                     need if need is not None else "never"])
    print(table(rows, ["context S", "AI B=1", "AI B=64", "AI as B->inf", "B for A100-40GB ridge"]))
    print(f"GQA: {hq} query heads share {hkv} KV heads, so attention alone does "
          f"~{hq // hkv} FLOP per KV byte no matter how large B is (each request reads its own KV).")
    assert decode_ai_limit(cfg, 32768) < r_a100  # long-context decode never reaches the ridge
    assert min_batch(lambda b: (lambda f, by: f / by)(*decode_step(cfg, b, 0)), r_a100) is not None

    # ------------------------------------------------------------------ 5
    if args.skip_measure:
        return
    section(f"5. measured on this CPU: fp32 [N, {args.hidden}] x [{args.hidden}, {args.hidden}]")
    ns = [1, 8, 64, 512]
    meas, n_mats = measure_cpu(args.hidden, ns)
    print(f"({n_mats} distinct weight matrices of {args.hidden**2 * 4 / 2**20:.0f} MiB cycled, "
          f"{n_mats * args.hidden**2 * 4 / 2**20:.0f} MiB total, to defeat the CPU caches)")
    print(table([[n, f"{dt * 1e3:.2f}", f"{gf / 1e9:.1f}", f"{gb / 1e9:.1f}", f"{ai:.2f}"]
                 for n, dt, gf, gb, ai in meas],
                ["N", "ms / matmul", "GFLOP/s", "GB/s", "AI (fp32) FLOP/B"]))
    t = {n: dt for n, dt, *_ in meas}
    g = {n: gf for n, _, gf, *_ in meas}
    print(f"time N=8 / N=1 = {t[8] / t[1]:.2f}x for 8x the FLOPs  -> memory-bound: the weight read dominates")
    print(f"GFLOP/s N=512 / N=1 = {g[512] / g[1]:.1f}x            -> weight reuse moves toward the compute roof")
    est_ridge = g[512] / (meas[0][3])
    print(f"rough CPU ridge = best GFLOP/s / GEMV GB/s = {est_ridge:.0f} FLOP/B (fp32)")
    assert t[8] < 4 * t[1], "8 rows should cost far less than 8x one row (weight read dominates)"
    assert g[512] > 8 * g[1], "N=512 should run at a much higher FLOP/s than a GEMV"


if __name__ == "__main__":
    main()
