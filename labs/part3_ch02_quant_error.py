"""Part III Ch2 lab: quantization error vs. bit width on real Qwen3-0.6B weights.

Three experiments, all CPU-only, all "fake quantization" (quantize -> dequantize
back to fp32), which isolates *rounding error* from kernel speed:

1. Weight-level error of every scheme on a few representative linear layers:
   relative RMS error ||W - Q(W)|| / ||W|| and the fraction of weights rounded to 0.
2. The outlier effect from Section 4.1: one large value in a group coarsens the
   grid for everyone else (per-tensor vs per-channel vs group-128 vs group-16).
3. End-to-end: quantize *all* decoder linear weights of Qwen3-0.6B, run greedy
   generation with mini-sglang, and compare against the fp32 model:
   next-token logit KL divergence and greedy token agreement.

    python labs/part3_ch02_quant_error.py --model /tmp/models/Qwen3-0.6B
    python labs/part3_ch02_quant_error.py --model /tmp/models/Qwen3-0.6B --skip-e2e   # ~5 s
"""

from __future__ import annotations

from _common import REAL_MODEL  # labs/_common.py (MINISGL_TEST_MODEL or /tmp/models/Qwen3-0.6B)

import argparse
import time
from typing import Callable, Dict

import safetensors
import torch

# --------------------------------------------------------------------------- formats
FP4_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round |x| to the nearest value of a non-negative grid, keep the sign."""
    mag = x.abs().unsqueeze(-1)
    idx = (mag - grid).abs().argmin(dim=-1)
    return grid[idx] * x.sign()


def _grouped(w: torch.Tensor, group: int) -> torch.Tensor:
    """[out, in] -> [out, in/group, group]; group=-1 means one group per row
    (per-channel), group=0 means one group for the whole tensor."""
    if group == 0:
        return w.reshape(1, 1, -1)
    if group == -1:
        return w.reshape(w.shape[0], 1, w.shape[1])
    assert w.shape[1] % group == 0
    return w.reshape(w.shape[0], w.shape[1] // group, group)


def quant_int(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Symmetric INT-b with absmax scale per group (no zero point)."""
    g = _grouped(w, group)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax)
    return (q * scale).reshape(w.shape)


def quant_fp8(w: torch.Tensor, group: int) -> torch.Tensor:
    """FP8 E4M3 (max 448) with an absmax scale per group, via torch's float8 dtype."""
    g = _grouped(w, group)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    q = (g / scale).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale).reshape(w.shape)


def quant_fp4(w: torch.Tensor, group: int, fp8_scale: bool = True) -> torch.Tensor:
    """FP4 E2M1 (max 6). NVFP4 = group 16 with an FP8-E4M3 scale per group."""
    g = _grouped(w, group)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 6.0
    if fp8_scale:  # the scale itself is stored in FP8 (NVFP4); ignore the global fp32 scale
        scale = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp_min(1e-12)
    q = _round_to_grid(g / scale, FP4_E2M1_GRID)
    return (q * scale).reshape(w.shape)


SCHEMES: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "bf16": lambda w: w.to(torch.bfloat16).float(),
    "fp8-e4m3 per-tensor": lambda w: quant_fp8(w, 0),
    "fp8-e4m3 per-channel": lambda w: quant_fp8(w, -1),
    "int8 per-channel": lambda w: quant_int(w, 8, -1),
    "int4 per-channel": lambda w: quant_int(w, 4, -1),
    "int4 group-128 (AWQ/GPTQ-like)": lambda w: quant_int(w, 4, 128),
    "fp4-e2m1 group-16 (NVFP4-like)": lambda w: quant_fp4(w, 16),
    "int3 group-128": lambda w: quant_int(w, 3, 128),
    "int2 group-128": lambda w: quant_int(w, 2, 128),
}


def bits_per_weight(name: str) -> float:
    """Storage incl. scale metadata (16-bit scales for INT groups, 8-bit for NVFP4)."""
    table = {
        "bf16": 16, "fp8-e4m3 per-tensor": 8, "fp8-e4m3 per-channel": 8, "int8 per-channel": 8,
        "int4 per-channel": 4, "int4 group-128 (AWQ/GPTQ-like)": 4 + 16 / 128,
        "fp4-e2m1 group-16 (NVFP4-like)": 4 + 8 / 16, "int3 group-128": 3 + 16 / 128,
        "int2 group-128": 2 + 16 / 128,
    }
    return table[name]


def rel_err(w: torch.Tensor, wq: torch.Tensor) -> float:
    return float((w - wq).norm() / w.norm())


# --------------------------------------------------------------------------- experiments
def weight_level(model_path: str) -> None:
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.14.mlp.down_proj.weight",
        "model.layers.27.mlp.gate_proj.weight",
    ]
    with safetensors.safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
        ws = {n: f.get_tensor(n).float() for n in names}
    print("## 1. Weight-level rounding error (relative RMS error, % of weights that became 0)\n")
    print("| scheme | bits/weight incl. scales | " + " | ".join(n.split("model.")[1].replace(".weight", "") for n in names) + " |")
    print("|---|---:|" + "---:|" * len(names))
    for name, fn in SCHEMES.items():
        cells = []
        for w in ws.values():
            wq = fn(w)
            zeros = float(((wq == 0) & (w != 0)).float().mean()) * 100
            cells.append(f"{rel_err(w, wq):.4f} ({zeros:.1f}% →0)")
        print(f"| {name} | {bits_per_weight(name):.3g} | " + " | ".join(cells) + " |")
    w = ws[names[1]]
    print(f"\n`{names[1]}`: absmax={w.abs().max():.3f}, std={w.std():.4f}, "
          f"absmax/std={float(w.abs().max() / w.std()):.0f}  (large ratio = outliers)\n")


def outlier_demo() -> None:
    print("## 2. Outliers coarsen the grid (Figure 6 of the chapter)\n")
    small = torch.tensor([0.10, -0.05, 0.12, 0.03, -0.08, 0.07, -0.11])
    with_outlier = torch.cat([small, torch.tensor([4.0])]).view(1, -1)
    q_shared = quant_int(with_outlier, 4, 0)[0, :7]
    q_alone = quant_int(small.view(1, -1), 4, 0)[0]
    print("| value | INT4, shares scale with 4.0 | INT4, outlier kept separately |")
    print("|---:|---:|---:|")
    for a, b, c in zip(small.tolist(), q_shared.tolist(), q_alone.tolist()):
        print(f"| {a:+.2f} | {b:+.3f} | {c:+.3f} |")
    print(f"\nscale with outlier = {4.0 / 7:.3f}, without = {0.12 / 7:.4f}; "
          f"zeros: {(q_shared == 0).sum().item()}/7 vs {(q_alone == 0).sum().item()}/7\n")


@torch.inference_mode()
def end_to_end(model_path: str, max_tokens: int) -> None:
    from minisgl import SamplingParams
    from minisgl.llm import LLM

    print("## 3. End-to-end on Qwen3-0.6B (all decoder linear weights fake-quantized)\n")
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Explain the KV cache of a transformer in one sentence:",
        "1, 1, 2, 3, 5, 8, 13,",
    ]
    # naive cache: the radix cache would reuse K/V computed with *other* weights
    llm = LLM(model_path, device="cpu", max_total_tokens=4096, max_seq_len=512, cache_type="naive")
    model = llm.engine.model
    linear = {n: p for n, p in model.named_parameters() if n.endswith("proj.weight")}
    original = {n: p.detach().clone() for n, p in linear.items()}
    ids = [llm.tokenize(p) for p in prompts]
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)

    def run():
        outs = llm.generate(ids, sp)
        return [o.token_ids for o in outs]

    t0 = time.perf_counter()
    ref_tokens = run()
    ref_logits = _next_token_logits(llm, ids)
    print(f"fp32 reference: {time.perf_counter() - t0:.1f} s\n")
    print("| scheme | greedy tokens identical to fp32 | first divergence (median position) | next-token KL(fp32‖q), mean | top-1 agree on next token |")
    print("|---|---:|---:|---:|---:|")
    for name in ["bf16", "fp8-e4m3 per-channel", "int8 per-channel", "int4 group-128 (AWQ/GPTQ-like)",
                 "fp4-e2m1 group-16 (NVFP4-like)", "int4 per-channel", "int3 group-128", "int2 group-128"]:
        fn = SCHEMES[name]
        for n, p in linear.items():
            p.copy_(fn(original[n]))
        toks = run()
        logits = _next_token_logits(llm, ids)
        same = sum(a == b for r, q in zip(ref_tokens, toks) for a, b in zip(r, q))
        total = sum(len(r) for r in ref_tokens)
        div = sorted(next((i for i, (a, b) in enumerate(zip(r, q)) if a != b), len(r)) for r, q in zip(ref_tokens, toks))
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), ref_logits.log_softmax(-1),
                                        log_target=True, reduction="none").sum(-1)
        top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()
        print(f"| {name} | {same}/{total} ({100 * same / total:.0f}%) | {div[len(div) // 2]} | "
              f"{kl.mean():.4f} | {100 * top1:.0f}% |")
    for n, p in linear.items():
        p.copy_(original[n])
    llm.shutdown()


def _next_token_logits(llm, ids) -> torch.Tensor:
    """Run each prompt's prefill directly through the engine to grab raw logits."""
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.kvcache import create_prefix_cache

    eng = llm.engine
    sched = llm.scheduler
    rows = []
    for x in ids:
        t = torch.tensor(x, dtype=torch.int64)
        idx = sched.table_manager.allocate()
        sched.table_manager.token_pool[idx, : len(t)] = t
        req = Req(input_ids=t, table_idx=idx, cached_len=0, output_len=1, uid=-1,
                  sampling_params=SamplingParams(), cache_handle=create_prefix_cache(eng.device, "naive").match_prefix(t[:0]).cuda_handle)
        batch = Batch(reqs=[req], phase="prefill")
        fi = sched._prepare_batch(batch)
        batch.input_ids = sched.table_manager.token_pool[fi.input_mapping]
        out = eng.forward_batch(batch, None)
        rows.append(out.logits[-1].float())
        # give the pages back
        sched.cache_manager._free(eng.page_table[idx, : len(t)].clone())
        sched.table_manager.free(idx)
    return torch.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REAL_MODEL)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    weight_level(args.model)
    outlier_demo()
    if not args.skip_e2e:
        end_to_end(args.model, args.max_tokens)


if __name__ == "__main__":
    main()
