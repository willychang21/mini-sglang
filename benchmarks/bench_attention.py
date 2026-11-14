"""Part III Ch1: attention backends and CUDA Graphs.

1. Kernel level: one decode / one prefill attention call, torch (gather + SDPA)
   vs triton (paged, flash-decoding with ``num_kv_splits``), over batch size
   and KV length. Inputs come from a scattered page table (page_size 16).
2. End to end: decode TPOT of a batch of requests through the scheduler, for
   each backend with and without CUDA Graphs.

    python benchmarks/bench_attention.py --model /path/to/Qwen3-0.6B --device cuda

On CPU the triton kernels run in the Triton interpreter; use ``--quick`` (tiny
shapes) only as a smoke test, the numbers mean nothing there.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from _common import add_engine_args, describe, engine_kwargs  # noqa: E402
from minisgl import SamplingParams  # noqa: E402
from minisgl.attention import TorchAttnBackend  # noqa: E402
from minisgl.attention.triton_backend import TritonAttnBackend  # noqa: E402
from minisgl.config import ModelConfig  # noqa: E402
from minisgl.core import Batch  # noqa: E402
from minisgl.kvcache import MHAKVCache  # noqa: E402
from minisgl.llm import LLM  # noqa: E402


@dataclass
class _Req:
    table_idx: int
    cached_len: int
    device_len: int

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


def _timeit(fn, device: str, iters: int) -> float:
    """Median milliseconds per call."""
    fn()  # warm-up (Triton JIT compile)
    times = []
    for _ in range(iters):
        if device.startswith("cuda"):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append(1000 * (time.perf_counter() - t0))
    return statistics.median(times)


def _make(bs, kv_len, q_len, cfg: ModelConfig, page_size, dtype, device):
    pages_per_req = -(-kv_len // page_size)
    num_pages = bs * pages_per_req
    cache = MHAKVCache(cfg.num_kv_heads, 1, cfg.head_dim, num_pages, page_size, dtype, torch.device(device))
    cache._kv_buffer.normal_()
    perm = torch.randperm(num_pages)  # scattered pages
    page_table = torch.zeros((bs + 1, pages_per_req * page_size), dtype=torch.int64, device=device)
    offs = torch.arange(page_size)
    for b in range(bs):
        pages = perm[b * pages_per_req : (b + 1) * pages_per_req]
        page_table[b] = ((pages * page_size).unsqueeze(1) + offs).flatten().to(device)
    reqs = [_Req(b, kv_len - q_len, kv_len) for b in range(bs)]
    batch = Batch(reqs=reqs, phase="decode" if q_len == 1 else "prefill")
    batch.out_loc = torch.cat([page_table[b, kv_len - q_len : kv_len] for b in range(bs)])
    n = bs * q_len
    q = torch.randn(n, cfg.num_qo_heads, cfg.head_dim, dtype=dtype, device=device)
    k = torch.randn(n, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    return cache, page_table, batch, q, k, v


def kernel_bench(args, cfg: ModelConfig) -> None:
    device = args.device
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    if args.quick:
        bss, lens, prefill_lens, iters = [1, 4], [64], [32], 2
    else:
        bss, lens, prefill_lens, iters = [1, 8, 32, 64], [512, 2048, 8192], [512, 2048], 20
    split_options = [1, 4, 8, 16]

    print("\n## Decode attention, one layer (ms, median)\n")
    print("| bs | kv_len | torch (gather+SDPA) | " + " | ".join(f"triton splits={s}" for s in split_options) + " |")
    print("|---:|---:|---:|" + "---:|" * len(split_options))
    for bs in bss:
        for kv_len in lens:
            cache, pt, batch, q, k, v = _make(bs, kv_len, 1, cfg, 16, dtype, device)
            row = []
            tb = TorchAttnBackend(cache, pt)
            tb.prepare_metadata(batch)
            row.append(_timeit(lambda: tb.forward(q, k, v, 0, batch), device, iters))
            for s in split_options:
                tr = TritonAttnBackend(cache, pt, cfg, num_kv_splits=s)
                tr.prepare_metadata(batch)
                row.append(_timeit(lambda: tr.forward(q, k, v, 0, batch), device, iters))
            print(f"| {bs} | {kv_len} | " + " | ".join(f"{t:.3f}" for t in row) + " |")

    print("\n## Prefill attention (no prefix), one layer (ms, median)\n")
    print("| bs | q_len = kv_len | torch (gather+SDPA) | triton extend |")
    print("|---:|---:|---:|---:|")
    for bs in bss[:2]:
        for n in prefill_lens:
            cache, pt, batch, q, k, v = _make(bs, n, n, cfg, 16, dtype, device)
            tb = TorchAttnBackend(cache, pt)
            tb.prepare_metadata(batch)  # metadata lives on the batch: time one backend at a time
            t_torch = _timeit(lambda: tb.forward(q, k, v, 0, batch), device, iters)
            tr = TritonAttnBackend(cache, pt, cfg)
            tr.prepare_metadata(batch)
            t_triton = _timeit(lambda: tr.forward(q, k, v, 0, batch), device, iters)
            print(f"| {bs} | {n} | {t_torch:.3f} | {t_triton:.3f} |")


def e2e_bench(args) -> None:
    on_cuda = args.device.startswith("cuda")
    bss = [1, 4] if args.quick else [1, 8, 32]
    prompt_len, max_tokens = (16, 4) if args.quick else (256, 64)
    # (backend, disable_cuda_graph). The torch backend cannot be captured.
    configs = [("torch", True), ("triton", True)] + ([("triton", False)] if on_cuda else [])
    print("\n## End-to-end decode (scheduler path)\n")
    print("| backend | CUDA graph | bs | decode TPOT ms | output tok/s | same as torch |")
    print("|---|:---:|---:|---:|---:|:---:|")
    ref = {}
    for backend, disable_graph in configs:
        kw = engine_kwargs(args)
        kw.update(attention_backend=backend, disable_cuda_graph=disable_graph)
        llm = LLM(args.model, load_tokenizer_=False, max_total_tokens=16384, max_seq_len=2048,
                  max_running_req=max(bss), **kw)
        llm.generate([[1, 2, 3]], SamplingParams(max_tokens=2))  # warm-up
        for bs in bss:
            g = torch.Generator().manual_seed(bs)
            vocab = llm.engine.model_config.vocab_size
            prompts = torch.randint(0, min(vocab, 10000), (bs, prompt_len), generator=g).tolist()
            sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
            outs = llm.generate(prompts, sp)
            # decode time = span between first and last token of each request
            tpot = statistics.mean(
                (o.token_times[-1] - o.token_times[0]) / (len(o.token_times) - 1) for o in outs
            )
            tokens = [o.token_ids for o in outs]
            same = ref.setdefault(bs, tokens) == tokens
            graph = "on" if llm.engine.graph_runner is not None else "off"
            print(f"| {backend} | {graph} | {bs} | {1000 * tpot:.2f} | {bs / tpot:.0f} | {same} |")
        llm.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny shapes (CPU smoke test)")
    ap.add_argument("--skip-e2e", action="store_true")
    add_engine_args(ap)
    args = ap.parse_args()
    print(describe(args))
    kernel_bench(args, ModelConfig.from_pretrained(args.model))
    if not args.skip_e2e:
        e2e_bench(args)


if __name__ == "__main__":
    main()
