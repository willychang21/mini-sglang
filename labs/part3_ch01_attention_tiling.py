"""Part III Ch1 lab: what FlashAttention-style kernels do, in plain PyTorch.

1. Naive attention materializes the full [Lq, Lk] score matrix (the 32K example:
   2 GiB per head per layer in 16-bit).
2. Tiled attention with an *online softmax* (running max m, running sum l,
   rescaled accumulator o) processes K/V in blocks and never stores the matrix,
   yet gives the same output up to float rounding.
3. Split-KV decode (FlashInfer / Triton decode / FlashMLA): one query, the KV
   sequence cut into segments computed independently, then merged with each
   segment's log-sum-exp. This is exactly ``merge_state`` in SGLang
   (python/sglang/srt/layers/attention/merge_state.py).
4. The same numbers checked against mini-sglang's TorchAttnBackend math
   (``F.scaled_dot_product_attention`` with a causal mask shifted by cached_len).

    python labs/part3_ch01_attention_tiling.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


def naive(q, k, v, causal_offset: int | None):
    s = (q @ k.T) / math.sqrt(q.shape[-1])  # [Lq, Lk]  <- the matrix FlashAttention avoids
    if causal_offset is not None:
        lq, lk = s.shape
        mask = torch.arange(lk)[None, :] <= (causal_offset + torch.arange(lq))[:, None]
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v


def tiled(q, k, v, causal_offset: int | None, block_k: int = 128):
    """Single pass over K/V blocks with online softmax (FlashAttention-1/2 inner loop)."""
    lq, d = q.shape
    m = torch.full((lq, 1), float("-inf"))  # running row max
    l = torch.zeros(lq, 1)  # running softmax denominator
    o = torch.zeros(lq, v.shape[-1])  # running (unnormalized) output
    for start in range(0, k.shape[0], block_k):
        kb, vb = k[start : start + block_k], v[start : start + block_k]
        s = (q @ kb.T) / math.sqrt(d)  # only a [Lq, block_k] tile lives at a time
        if causal_offset is not None:
            cols = start + torch.arange(kb.shape[0])
            s = s.masked_fill(cols[None, :] > (causal_offset + torch.arange(lq))[:, None], float("-inf"))
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        alpha = torch.exp(m - m_new)  # rescale what we accumulated so far
        p = torch.exp(s - m_new)
        p = torch.nan_to_num(p)  # fully-masked tiles
        l = alpha * l + p.sum(-1, keepdim=True)
        o = alpha * o + p @ vb
        m = m_new
    return o / l


def split_kv_decode(q, k, v, num_splits: int):
    """Decode: 1 query row. Each split returns (partial output, log-sum-exp)."""
    parts = []
    for kb, vb in zip(k.chunk(num_splits), v.chunk(num_splits)):
        s = (q @ kb.T) / math.sqrt(q.shape[-1])
        lse = torch.logsumexp(s, -1, keepdim=True)
        parts.append((torch.softmax(s, -1) @ vb, lse))
    # merge (what merge_state / the FlashInfer "merge" kernel does)
    lses = torch.cat([p[1] for p in parts], -1)  # [1, S]
    lse_total = torch.logsumexp(lses, -1, keepdim=True)
    w = torch.exp(lses - lse_total)  # weight of each split
    return sum(w[:, i : i + 1] * parts[i][0] for i in range(len(parts)))


def main() -> None:
    torch.manual_seed(0)
    print("## 1. Size of the score matrix FlashAttention never writes to HBM\n")
    for n in (4096, 32768, 131072):
        print(f"  L={n:>6}: one head, one layer, 16-bit scores = {n * n * 2 / 2**30:8.2f} GiB")

    d = 128
    lq, lk = 256, 2048
    q, k, v = torch.randn(lq, d), torch.randn(lk, d), torch.randn(lk, d)
    cached = lk - lq  # chunked prefill: 1792 cached tokens, 256 new ones
    ref = naive(q, k, v, cached)
    out = tiled(q, k, v, cached)
    assert (out - ref).abs().max() < 1e-5
    print("\n## 2. Tiled + online softmax == naive (prefill chunk with cached prefix)\n")
    print(f"  max |diff| = {(out - ref).abs().max():.2e};  peak score storage: naive {lq * lk} floats,"
          f" tiled {lq * 128} floats ({lk // 128}x less)")

    print("\n## 3. Split-KV decode: merge with log-sum-exp\n")
    q1 = torch.randn(1, d)
    kd, vd = torch.randn(8192, d), torch.randn(8192, d)
    full = naive(q1, kd, vd, None)
    for s in (1, 4, 16, 64):
        diff = (split_kv_decode(q1, kd, vd, s) - full).abs().max()
        assert diff < 1e-5
        print(f"  splits={s:>2}: max |diff| vs unsplit = {diff:.2e}")

    print("\n## 4. Same math as mini-sglang's TorchAttnBackend (SDPA + mask k <= cached_len + t)\n")
    mask = torch.arange(lk)[None, :] <= (cached + torch.arange(lq))[:, None]
    sdpa = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], attn_mask=mask)[0, 0]
    assert (out - sdpa).abs().max() < 1e-5
    print(f"  max |diff| tiled vs SDPA = {(out - sdpa).abs().max():.2e}")

    print("\n## 5. Why decode wants split-KV: work per query\n")
    for bs, ctx in [(1, 32768), (64, 1024)]:
        ctas = bs * 16  # one CTA per (request, q-head) without split-KV, Qwen3-0.6B has 16 q-heads
        verdict = "under-fills the GPU -> split the KV" if ctas < 132 else "enough parallelism already"
        print(f"  batch={bs:>2}, context={ctx:>5}: {ctas:>4} independent (request, head) tasks vs ~132 SMs -> {verdict}")
    t0 = time.perf_counter()
    for _ in range(20):
        naive(q, k, v, cached)
    t1 = time.perf_counter()
    for _ in range(20):
        tiled(q, k, v, cached)
    t2 = time.perf_counter()
    print(f"\n(CPU timing, not representative of GPUs: naive {1e3 * (t1 - t0) / 20:.1f} ms, "
          f"python-loop tiled {1e3 * (t2 - t1) / 20:.1f} ms. Tiling pays off only when the tile lives in SRAM.)")


if __name__ == "__main__":
    main()
