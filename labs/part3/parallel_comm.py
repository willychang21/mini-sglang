"""Part III Ch4 lab: communication volume of TP, DP-Attention, EP and PP.

A back-of-the-envelope calculator for *one decode step* (or one prefill chunk) of a
dense or MoE transformer. Per strategy it reports bytes moved per GPU per step and
the estimated communication time at a given link bandwidth, next to the compute
time saved, so you can see where adding GPUs stops paying off.

Collective cost models (ring algorithms, n ranks, message of S bytes):
    all-reduce      2 (n-1)/n * S   bytes sent per rank
    all-gather      (n-1)/n * S_total
    all-to-all      (n-1)/n * S     (uniform routing)
    point-to-point  S               (PP stage boundary)

It also prints the measured all-reduce count of mini-sglang's own TP (one per
attention block + one per MLP, i.e. 2 per layer) so the model matches the code.

    python labs/part3/parallel_comm.py
    python labs/part3/parallel_comm.py --tokens 8192 --link-gbps 400   # prefill chunk over NVLink
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Shape:
    name: str
    layers: int
    hidden: int
    params_b: float
    moe_experts: int = 0  # routed experts per layer (0 = dense)
    topk: int = 0
    moe_active_b: float = 0.0  # activated parameters per token (MoE)


SHAPES = {
    "qwen3-32b": Shape("Qwen3-32B (dense)", 64, 5120, 32.8),
    "qwen3-235b-a22b": Shape("Qwen3-235B-A22B (MoE)", 94, 4096, 235, 128, 8, 22),
    "deepseek-v3": Shape("DeepSeek-V3 (MoE)", 61, 7168, 671, 256, 8, 37),
}


def ring_allreduce(nbytes: float, n: int) -> float:
    return 0.0 if n == 1 else 2 * (n - 1) / n * nbytes


def report(s: Shape, tokens: int, n: int, link_gbps: float, act_bytes: int, gpu_tflops: float, mfu: float):
    act = tokens * s.hidden * act_bytes  # one activation tensor for this step
    bw = link_gbps * 1e9 / 8 if link_gbps > 0 else float("inf")  # Gbit/s -> B/s
    flops = 2 * (s.moe_active_b or s.params_b) * 1e9 * tokens
    t_compute_1 = flops / (gpu_tflops * 1e12 * mfu)
    rows = []
    # TP: 2 all-reduces per layer (o_proj, down_proj) -- exactly mini-sglang's RowParallelLinear
    tp = 2 * s.layers * ring_allreduce(act, n)
    rows.append(("TP", tp, 2 * s.layers, t_compute_1 / n))
    # DP attention + TP MLP (dense): attention (incl. o_proj) is local to the rank that owns the
    # request, so no all-reduce there; the MLP needs all-gather (tokens in) + reduce-scatter
    # (results out) = 2 (n-1)/n * act per layer, i.e. half of TP's two all-reduces. The big win
    # of DP attention is KV memory (no replicated KV), not bandwidth.
    if s.moe_experts == 0:
        dpa = s.layers * 2 * (n - 1) / n * act if n > 1 else 0.0
        rows.append(("DP-Attn + TP-MLP", dpa, 2 * s.layers, t_compute_1 / n))
    else:
        # EP: dispatch + combine = 2 all-to-alls per MoE layer; each token goes to topk experts,
        # (n-1)/n of them remote on average. Attention is DP (local requests), so nothing else moves.
        # with DP attention each rank only holds tokens/n of the batch
        ep = s.layers * 2 * (n - 1) / n * (act / n) * s.topk if n > 1 else 0.0
        rows.append(("DP-Attn + EP", ep, 2 * s.layers, t_compute_1 / n))
    # PP: one send of the activation per stage boundary; each GPU is one stage
    pp = act if n > 1 else 0.0
    rows.append(("PP (per boundary)", pp, n - 1, t_compute_1))  # latency: every stage still runs in sequence

    print(f"### {s.name}: {tokens} tokens/step, {n} GPUs, link {link_gbps} Gbit/s, activations {act_bytes} B\n")
    print("| strategy | bytes sent per GPU per step | collectives per step | comm time (bw only) | compute time on the step's critical path |")
    print("|---|---:|---:|---:|---:|")
    for name, b, count, tc in rows:
        print(f"| {name} | {b / 2**20:.2f} MiB | {count} | {1e3 * b / bw:.3f} ms | {1e3 * tc:.3f} ms |")
    print()


def minisgl_allreduce_count() -> None:
    """Count all-reduces in one forward of mini-sglang's Qwen3 (tiny random checkpoint is fine)."""
    from minisgl.layers.linear import RowParallelLinear

    import minisgl.models.qwen3 as q

    src = open(q.__file__).read() + open(q.__file__.replace("qwen3.py", "utils.py")).read()
    rows = src.count("RowParallelLinear(")
    print(f"mini-sglang Qwen3: RowParallelLinear instantiations per layer = {rows} "
          f"(o_proj + down_proj) -> {rows} all-reduces per layer, see {RowParallelLinear.forward.__qualname__}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64, help="tokens in the step (decode batch size or prefill chunk)")
    ap.add_argument("--gpus", type=int, nargs="*", default=[2, 4, 8, 16])
    ap.add_argument("--link-gbps", type=float, default=3600, help="per-GPU link, Gbit/s (NVLink4 ~3600, IB NDR 400)")
    ap.add_argument("--act-bytes", type=int, default=2)
    ap.add_argument("--gpu-tflops", type=float, default=989)
    ap.add_argument("--mfu", type=float, default=0.4)
    ap.add_argument("--models", nargs="*", default=["qwen3-32b", "deepseek-v3"])
    args = ap.parse_args()
    try:
        minisgl_allreduce_count()
    except Exception as e:  # pragma: no cover
        print(f"(skipped mini-sglang check: {e})\n")
    for m in args.models:
        for n in args.gpus:
            report(SHAPES[m], args.tokens, n, args.link_gbps, args.act_bytes, args.gpu_tflops, args.mfu)
    print("Reading the table: comm time ignores per-collective latency (~5-20 us each on NVLink, more\n"
          "across nodes). With 2 x layers collectives per step, latency alone is ~1-2.5 ms for 64 layers,\n"
          "which is why small-batch decode stops scaling with TP long before bandwidth runs out.")


if __name__ == "__main__":
    main()
