"""A pure-PyTorch paged attention backend.

It plays the role of the reference FlashAttention/FlashInfer backends
(``attention/fa.py``) so that the engine runs anywhere, including CPU:

1. ``store_kv(k, v, out_loc)`` writes the K/V of this step's tokens into the
   pool *first* (Chapter 4, Section 5.3: the order cannot be reversed).
2. The K/V history of every request is gathered through its ``page_table`` row.
3. One padded ``scaled_dot_product_attention`` call handles the whole batch.
   Query ``t`` of a request may see key ``j`` iff ``j <= cached_len + t``
   (causal mask shifted by the cached prefix).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..core import Batch
from ..kvcache.mha_pool import MHAKVCache
from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class TorchAttnMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor  # [B]
    kv_indices: torch.Tensor  # [B, max_k] physical token slots (padded)
    q_batch_idx: torch.Tensor  # [N] request of each packed query token
    q_pos_idx: torch.Tensor  # [N] offset of the token inside its request
    mask: torch.Tensor  # [B, 1, max_q, max_k] True = may attend
    max_q: int

    def get_last_indices(self) -> torch.Tensor:
        return self.last_indices


class TorchAttnBackend(BaseAttnBackend):
    def __init__(self, kvcache: MHAKVCache, page_table: torch.Tensor) -> None:
        self.kvcache = kvcache
        self.page_table = page_table

    def prepare_metadata(self, batch: Batch) -> None:
        device = self.kvcache.device
        page_table = self.page_table
        reqs = batch.reqs
        seqlens_q = torch.tensor([r.extend_len for r in reqs], dtype=torch.int64)
        seqlens_k = torch.tensor([r.device_len for r in reqs], dtype=torch.int64)
        cached = torch.tensor([r.cached_len for r in reqs], dtype=torch.int64)
        bs, max_q, max_k = len(reqs), int(seqlens_q.max()), int(seqlens_k.max())

        kv_indices = torch.zeros((bs, max_k), dtype=torch.int64, device=device)
        for i, r in enumerate(reqs):
            kv_indices[i, : r.device_len] = page_table[r.table_idx, : r.device_len]

        q_batch_idx = torch.repeat_interleave(torch.arange(bs), seqlens_q)
        q_pos_idx = torch.cat([torch.arange(int(n)) for n in seqlens_q])
        k_ar = torch.arange(max_k).view(1, 1, max_k)
        q_ar = torch.arange(max_q).view(1, max_q, 1)
        visible = (k_ar <= cached.view(-1, 1, 1) + q_ar) & (k_ar < seqlens_k.view(-1, 1, 1))
        batch.attn_metadata = TorchAttnMetadata(
            last_indices=(torch.cumsum(seqlens_q, 0) - 1).to(device),
            kv_indices=kv_indices,
            q_batch_idx=q_batch_idx.to(device),
            q_pos_idx=q_pos_idx.to(device),
            mask=visible.unsqueeze(1).to(device),
            max_q=max_q,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, TorchAttnMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_hist = self.kvcache.flat_k_cache(layer_id)[md.kv_indices]  # [B, max_k, Hkv, D]
        v_hist = self.kvcache.flat_v_cache(layer_id)[md.kv_indices]
        bs, num_heads, head_dim = md.kv_indices.shape[0], q.shape[1], q.shape[2]
        q_pad = q.new_zeros((bs, md.max_q, num_heads, head_dim))
        q_pad[md.q_batch_idx, md.q_pos_idx] = q

        o = F.scaled_dot_product_attention(
            q_pad.transpose(1, 2),
            k_hist.transpose(1, 2),
            v_hist.transpose(1, 2),
            attn_mask=md.mask,
            enable_gqa=True,
        )  # [B, Hq, max_q, D]
        return o.transpose(1, 2)[md.q_batch_idx, md.q_pos_idx]
