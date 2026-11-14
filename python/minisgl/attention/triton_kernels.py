"""Triton paged-attention kernels (Part III Ch1).

Both kernels read K/V **through the page table**: for key position ``j`` of the
request in page-table row ``r`` the physical token slot is::

    page_start = page_table[r, (j // PAGE_SIZE) * PAGE_SIZE]   # first slot of j's page
    slot       = page_start + j % PAGE_SIZE

(``page_table`` stores raw token slots, one entry per token, so the entry of the
first token of a page is that page's base slot.) Nothing is gathered into a
padded ``[B, max_len]`` tensor like ``torch_backend.py`` does: each program walks
its request's pages with an *online softmax*, so memory traffic is exactly the
K/V that is actually needed.

Layouts
-------
``q``            ``[N, Hq, D]`` packed query tokens of the whole batch
``k/v_cache``    ``[num_slots, Hkv, D]`` one layer of the pool (``flat_k_cache``)
``page_table``   ``[max_running_req + 1, max_seq_len]`` raw token slots

Decode (one query token per request) uses *flash-decoding*: the KV sequence is
cut into ``num_kv_splits`` chunks processed by separate programs (stage 1, more
parallelism when the batch is small), then merged with their log-sum-exp
(stage 2). Extend/prefill handles many query tokens per request with a causal
mask shifted by the cached prefix: query ``t`` sees key ``j`` iff
``j <= cached_len + t``.

All accumulation is fp32; ``tl.dot`` uses IEEE precision so fp32 runs are not
silently downgraded to TF32 on Ampere+ GPUs.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _next_pow2(x: int, at_least: int = 16) -> int:
    # tl.dot needs every dimension >= 16 on GPUs, and tl.arange a power of two.
    return max(at_least, triton.next_power_of_2(x))


def _dot_fp32(dtype: torch.dtype) -> bool:
    # The Triton *interpreter* (CPU) computes tl.dot with numpy, which has no
    # bfloat16 and returns garbage for it. Only there, upcast the tiles to fp32
    # before tl.dot. On a GPU bf16/fp16 tiles go to the tensor cores as they are.
    return os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16


# --------------------------------------------------------------------- decode
@triton.jit
def _decode_stage1_kernel(
    Q, K, V, PageTable, Rows, SeqLens, MidO, MidLse,
    sm_scale,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_vt, stride_vh,
    stride_pt,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    GROUP: tl.constexpr,  # query heads per KV head (GQA)
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    DOT_FP32: tl.constexpr,
):
    # One program = (request b, KV head, KV split). It serves all GROUP query
    # heads that share this KV head, so each K/V tile is loaded once per group.
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(SeqLens + b)
    row = tl.load(Rows + b).to(tl.int64)
    chunk = tl.cdiv(seq_len, NUM_SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, seq_len)

    offs_g = tl.arange(0, BLOCK_H)
    offs_h = kv_head * GROUP + offs_g
    mask_h = offs_g < GROUP
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    q = tl.load(
        Q + b * stride_qt + offs_h[:, None] * stride_qh + offs_d[None, :],
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )
    if DOT_FP32:
        q = q.to(tl.float32)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end
        page_start = tl.load(
            PageTable + row * stride_pt + (offs_n // PAGE_SIZE) * PAGE_SIZE, mask=mask_n, other=0
        )
        slot = (page_start + offs_n % PAGE_SIZE).to(tl.int64)
        kv_mask = mask_n[:, None] & mask_d[None, :]
        k = tl.load(
            K + slot[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=kv_mask, other=0.0
        )
        if DOT_FP32:
            k = k.to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * sm_scale  # [H, N]
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        v = tl.load(
            V + slot[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=kv_mask, other=0.0
        )
        if DOT_FP32:
            v = v.to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee").to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # A split can be empty when seq_len < NUM_SPLITS: it reports lse = -inf so
    # the merge gives it zero weight.
    has_kv = l_i > 0
    out = acc / tl.where(has_kv, l_i, 1.0)[:, None]
    lse = tl.where(has_kv, m_i + tl.log(tl.where(has_kv, l_i, 1.0)), float("-inf"))
    tl.store(
        MidO + b * stride_ob + offs_h[:, None] * stride_oh + split * stride_os + offs_d[None, :],
        out,
        mask=mask_h[:, None] & mask_d[None, :],
    )
    tl.store(MidLse + b * stride_lb + offs_h * stride_lh + split, lse, mask=mask_h)


@triton.jit
def _decode_merge_kernel(
    MidO, MidLse, O,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    stride_out_t, stride_out_h,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # Stage 2: o = sum_s exp(lse_s - M) * o_s / sum_s exp(lse_s - M).
    # Each o_s is already normalized by its own softmax denominator, and
    # exp(lse_s) *is* that denominator (times exp(m_s)), so this re-weights the
    # partial results exactly as a single softmax over the whole sequence would.
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    lse = tl.load(MidLse + b * stride_lb + h * stride_lh + offs_s, mask=mask_s, other=float("-inf"))
    o = tl.load(
        MidO + b * stride_ob + h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    )
    m = tl.max(lse, 0)
    w = tl.exp(lse - m)  # empty splits: exp(-inf) = 0
    out = tl.sum(w[:, None] * o, 0) / tl.sum(w, 0)
    tl.store(O + b * stride_out_t + h * stride_out_h + offs_d, out, mask=mask_d)


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    rows: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    page_size: int,
    num_kv_splits: int = 1,
    block_n: int = 32,
) -> torch.Tensor:
    """Paged decode attention. ``q``: ``[B, Hq, D]`` (one token per request).

    ``rows[b]`` is request b's page-table row, ``seq_lens[b]`` its total KV
    length *including* the token being decoded (already stored in the cache).
    With ``num_kv_splits == 1`` stage 1 writes straight into the output and no
    merge runs; with ``> 1`` it is flash-decoding (split-KV + LSE merge).
    """
    bs, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_qo_heads // num_kv_heads
    o = torch.empty_like(q)
    if num_kv_splits == 1:
        mid_o = o.unsqueeze(2)  # [B, Hq, 1, D] view: stage 1 writes the final result
    else:
        mid_o = torch.empty((bs, num_qo_heads, num_kv_splits, head_dim), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((bs, num_qo_heads, num_kv_splits), dtype=torch.float32, device=q.device)
    block_d = _next_pow2(head_dim)
    _decode_stage1_kernel[(bs, num_kv_heads, num_kv_splits)](
        q, k_cache, v_cache, page_table, rows, seq_lens, mid_o, mid_lse,
        sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        page_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        GROUP=group,
        HEAD_DIM=head_dim,
        BLOCK_H=_next_pow2(group),
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        PAGE_SIZE=page_size,
        NUM_SPLITS=num_kv_splits,
        DOT_FP32=_dot_fp32(q.dtype),
    )
    if num_kv_splits > 1:
        _decode_merge_kernel[(bs, num_qo_heads)](
            mid_o, mid_lse, o,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            mid_lse.stride(0), mid_lse.stride(1),
            o.stride(0), o.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_S=triton.next_power_of_2(num_kv_splits),
            NUM_SPLITS=num_kv_splits,
        )
    return o


# --------------------------------------------------------------------- extend
@triton.jit
def _extend_kernel(
    Q, K, V, O, PageTable, Rows, QoStart, ExtendLens, PrefixLens,
    sm_scale,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_vt, stride_vh,
    stride_ot, stride_oh,
    stride_pt,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DOT_FP32: tl.constexpr,
):
    # One program = (request b, query head h, block of BLOCK_M query tokens).
    b = tl.program_id(0)
    h = tl.program_id(1)
    mb = tl.program_id(2)

    extend_len = tl.load(ExtendLens + b)
    if mb * BLOCK_M >= extend_len:
        return
    prefix_len = tl.load(PrefixLens + b)  # cached_len: K/V already in the pool
    qo_start = tl.load(QoStart + b)  # first packed index of this request's queries
    row = tl.load(Rows + b).to(tl.int64)
    kv_head = h // GROUP

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)  # offset inside this extend
    mask_m = offs_m < extend_len
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q_tok = (qo_start + offs_m).to(tl.int64)
    q = tl.load(
        Q + q_tok[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    if DOT_FP32:
        q = q.to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # The last query of this block sees keys [0, prefix_len + its offset].
    kv_end = prefix_len + tl.minimum(extend_len, (mb + 1) * BLOCK_M)
    for n0 in range(0, kv_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_end
        page_start = tl.load(
            PageTable + row * stride_pt + (offs_n // PAGE_SIZE) * PAGE_SIZE, mask=mask_n, other=0
        )
        slot = (page_start + offs_n % PAGE_SIZE).to(tl.int64)
        kv_mask = mask_n[:, None] & mask_d[None, :]
        k = tl.load(
            K + slot[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=kv_mask, other=0.0
        )
        if DOT_FP32:
            k = k.to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * sm_scale  # [M, N]
        causal = offs_n[None, :] <= prefix_len + offs_m[:, None]
        qk = tl.where(causal & mask_n[None, :], qk, float("-inf"))

        # Key 0 is visible to every row, so the first block makes m_i finite.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        v = tl.load(
            V + slot[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=kv_mask, other=0.0
        )
        if DOT_FP32:
            v = v.to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee").to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        O + q_tok[:, None] * stride_ot + h * stride_oh + offs_d[None, :],
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    rows: torch.Tensor,
    qo_start: torch.Tensor,
    extend_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_extend_len: int,
    sm_scale: float,
    page_size: int,
    block_m: int = 32,
    block_n: int = 32,
) -> torch.Tensor:
    """Paged causal attention for prefill / extend. ``q``: ``[N, Hq, D]`` packed.

    Request b owns queries ``q[qo_start[b] : qo_start[b] + extend_lens[b]]``;
    its K/V (prefix *and* the new tokens, already stored) live at positions
    ``[0, prefix_lens[b] + extend_lens[b])`` of page-table row ``rows[b]``.
    """
    num_tokens, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    bs = rows.shape[0]
    o = torch.empty_like(q)
    grid = (bs, num_qo_heads, triton.cdiv(max_extend_len, block_m))
    _extend_kernel[grid](
        q, k_cache, v_cache, o, page_table, rows, qo_start, extend_lens, prefix_lens,
        sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        o.stride(0), o.stride(1),
        page_table.stride(0),
        GROUP=num_qo_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=_next_pow2(head_dim),
        BLOCK_N=block_n,
        PAGE_SIZE=page_size,
        DOT_FP32=_dot_fp32(q.dtype),
    )
    return o
