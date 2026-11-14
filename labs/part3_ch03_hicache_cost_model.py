"""Part III Ch3 lab: restore-vs-recompute cost model for hierarchical caching.

Part A (measured, CPU): on Qwen3-0.6B with mini-sglang, compare
  * recompute: prefill a prefix of L tokens (what a cache miss costs), with
  * restore:   copy that prefix's K/V for all layers from a "host" tensor into
               mini-sglang's KV pool slots (what an L2 hit costs; on CPU this is a
               memcpy standing in for the H2D copy).
Recompute grows ~linearly-to-quadratically in L (FLOPs); restore grows linearly in
bytes. Their ratio is what decides whether a lower-tier hit is worth it.

Part B (analytic, GPU): the same comparison with roofline numbers you choose
(GPU FLOP/s, PCIe/NVLink bandwidth, storage bandwidth), plus prefetch and
layer-wise overlap. Defaults are illustrative H100-class numbers, not measurements.

Part C: the chapter's Exercise 3 (150 ms recompute, 30 ms H2D, +200 ms storage).

    python labs/part3_ch03_hicache_cost_model.py --model /tmp/models/Qwen3-0.6B
    python labs/part3_ch03_hicache_cost_model.py --skip-measure
"""

from __future__ import annotations

from _common import REAL_MODEL  # labs/_common.py (MINISGL_TEST_MODEL or /tmp/models/Qwen3-0.6B)

import argparse
import time


# --------------------------------------------------------------------- Part A
def measure(model_path: str, lengths) -> None:
    import torch

    from minisgl import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(model_path, load_tokenizer_=False, device="cpu", max_total_tokens=8192,
              max_seq_len=4096, cache_type="naive")
    kv = llm.engine.kv_cache
    cfg = llm.engine.model_config
    per_tok = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * kv.dtype.itemsize
    sp = SamplingParams(max_tokens=1, ignore_eos=True)
    llm.generate([[100] * 16], sp)  # warmup
    print("## A. Measured on CPU (Qwen3-0.6B, fp32 KV, mini-sglang)\n")
    print("| prefix tokens | KV bytes | recompute (prefill) ms | restore (copy all layers) ms | recompute / restore |")
    print("|---:|---:|---:|---:|---:|")
    for n in lengths:
        prompt = [(i * 7919) % 30000 + 100 for i in range(n)]
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        t_re = time.perf_counter() - t0
        # restore: host buffer -> pool slots, layer by layer (what load_back does per layer)
        slots = torch.arange(n)
        host_k = torch.randn(cfg.num_layers, n, *kv.flat_k_cache(0).shape[1:], dtype=kv.dtype)
        host_v = torch.randn_like(host_k)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for layer in range(cfg.num_layers):
                kv.store_kv(host_k[layer], host_v[layer], slots, layer)
            best = min(best, time.perf_counter() - t0)
        print(f"| {n} | {n * per_tok / 2**20:.1f} MiB | {1e3 * t_re:.0f} | {1e3 * best:.1f} | {t_re / best:.0f}x |")
    llm.shutdown()
    print("\nOn CPU, compute is so slow that restoring is always far cheaper. The GPU picture is closer,"
          " so use Part B.\n")


# --------------------------------------------------------------------- Part B
def analytic(args) -> None:
    # Qwen3-8B-like shape by default
    L, layers, kvh, hd, params = args.prefix, args.layers, args.kv_heads, args.head_dim, args.params_b * 1e9
    hidden = args.hidden
    kv_bytes = 2 * layers * kvh * hd * args.kv_elem_bytes * L
    # prefill FLOPs: 2*params per token (linear layers) + attention 2*2*L^2*hidden/2 (causal) per layer
    flops = 2 * params * L + layers * 2 * L * L * hidden
    t_recompute = flops / (args.gpu_tflops * 1e12 * args.mfu)
    t_h2d = kv_bytes / (args.h2d_gbps * 1e9)
    t_storage = kv_bytes / (args.storage_gbps * 1e9) + args.storage_latency_ms / 1e3
    print(f"## B. Analytic GPU model: prefix={L} tokens, KV={kv_bytes / 2**30:.2f} GiB\n")
    print(f"assumptions: {args.gpu_tflops} TFLOP/s x MFU {args.mfu}, H2D {args.h2d_gbps} GB/s, "
          f"storage {args.storage_gbps} GB/s + {args.storage_latency_ms} ms\n")
    print("| path | time ms | vs recompute |")
    print("|---|---:|---:|")
    rows = [("recompute (miss)", t_recompute), ("L2 hit: H2D restore", t_h2d),
            ("L3 hit: storage->host->GPU", t_storage + t_h2d),
            ("L3 hit, storage read fully hidden by prefetch", t_h2d)]
    for name, t in rows:
        print(f"| {name} | {1e3 * t:.1f} | {t / t_recompute:.2f}x |")
    # layer-wise overlap: with per-layer transfer t_h2d/layers and per-layer compute of the *new* tokens
    t_new = (2 * params * args.new_tokens + layers * 2 * args.new_tokens * (L + args.new_tokens) * hidden) / (
        args.gpu_tflops * 1e12 * args.mfu)
    serial = t_h2d + t_new
    per_layer_x, per_layer_c = t_h2d / layers, t_new / layers
    overlapped = per_layer_x + (layers - 1) * max(per_layer_x, per_layer_c) + per_layer_c
    print(f"\nLayer-wise restore for {args.new_tokens} new tokens: serial {1e3 * serial:.1f} ms -> "
          f"overlapped {1e3 * overlapped:.1f} ms (bound by the slower of transfer/compute per layer)")
    # break-even prefix length for an L3 hit
    print("\nBreak-even: smallest prefix for which an L3 hit (no prefetch) beats recompute:")
    for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        kb = 2 * layers * kvh * hd * args.kv_elem_bytes * n
        tr = (2 * params * n + layers * 2 * n * n * hidden) / (args.gpu_tflops * 1e12 * args.mfu)
        tl3 = kb / (args.storage_gbps * 1e9) + args.storage_latency_ms / 1e3 + kb / (args.h2d_gbps * 1e9)
        print(f"  {n:>6} tokens: recompute {1e3 * tr:7.1f} ms, L3 restore {1e3 * tl3:7.1f} ms -> "
              f"{'restore' if tl3 < tr else 'recompute'}")
    print()


# --------------------------------------------------------------------- Part C
def exercise3() -> None:
    recompute, h2d, storage = 150, 30, 200
    print("## C. Chapter 3, Exercise 3\n")
    assert h2d < recompute < storage + h2d  # L2 helps, L3 alone does not
    print(f"recompute {recompute} ms; L2 hit = {h2d} ms ({'faster' if h2d < recompute else 'slower'}); "
          f"L3 hit = {storage}+{h2d} = {storage + h2d} ms ({'faster' if storage + h2d < recompute else 'slower'})")
    for hidden in (0, 100, 200):
        t = max(0, storage - hidden) + h2d
        print(f"  if prefetch hides {hidden} ms of the storage read while queued: L3 hit costs {t} ms "
              f"-> {'beats' if t < recompute else 'loses to'} recompute")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REAL_MODEL)
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="*", default=[256, 512, 1024, 2048])
    # analytic defaults: Qwen3-8B on an H100-class GPU (illustrative, not measured)
    ap.add_argument("--prefix", type=int, default=8192)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--params-b", type=float, default=8.19)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--kv-elem-bytes", type=float, default=2)
    ap.add_argument("--gpu-tflops", type=float, default=989)
    ap.add_argument("--mfu", type=float, default=0.5)
    ap.add_argument("--h2d-gbps", type=float, default=50)
    ap.add_argument("--storage-gbps", type=float, default=5)
    ap.add_argument("--storage-latency-ms", type=float, default=2)
    args = ap.parse_args()
    if not args.skip_measure:
        measure(args.model, args.lengths)
    analytic(args)
    exercise3()


if __name__ == "__main__":
    main()
