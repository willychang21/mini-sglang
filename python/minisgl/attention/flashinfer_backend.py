"""FlashInfer backend (optional; registered only when ``flashinfer`` imports).

UNTESTED in this repository's CI (no GPU). It exists to show the *plan / run*
split that distinguishes FlashInfer from the Triton kernels:

* ``plan(...)`` runs once per step on the host with the CPU copies of the
  lengths. It decides the work partition (how many KV chunks per request,
  which CTA handles what) and uploads that schedule to the GPU.
* ``run(q, kv_cache)`` runs once per *layer* and only launches kernels.

FlashInfer consumes a page-granular, CSR-style description of the KV cache:
``indptr`` (cumulative pages per request), ``indices`` (page ids) and
``last_page_len`` (valid tokens in each request's last page). Our page table
stores token slots, so a page id is ``page_table[row, i * page_size] // page_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import ModelConfig
from ..core import Batch
from ..distributed import get_tp_info
from ..kvcache.mha_pool import MHAKVCache
from ..utils import div_ceil, div_even
from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class FlashInferMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor
    wrapper: object  # the planned prefill or decode wrapper

    def get_last_indices(self) -> torch.Tensor:
        return self.last_indices


class FlashInferAttnBackend(BaseAttnBackend):
    def __init__(self, kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig) -> None:
        import flashinfer

        assert kvcache.device.type == "cuda", "FlashInfer needs a CUDA device"
        self.kvcache = kvcache
        self.page_table = page_table
        self.page_size = kvcache.page_size
        tp = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp)
        self.num_kv_heads = div_even(config.num_kv_heads, tp, allow_replicate=True)
        self.head_dim = config.head_dim
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=kvcache.device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=self.num_qo_heads // self.num_kv_heads >= 4
        )

    def prepare_metadata(self, batch: Batch) -> None:
        P, device = self.page_size, self.kvcache.device
        reqs = batch.reqs
        num_pages = [div_ceil(r.device_len, P) for r in reqs]
        kv_indptr = torch.tensor([0] + num_pages, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        kv_indices = torch.cat(
            [self.page_table[r.table_idx, : r.device_len : P] // P for r in reqs]
        ).to(torch.int32)
        last_page_len = torch.tensor(
            [r.device_len - (n - 1) * P for r, n in zip(reqs, num_pages)], dtype=torch.int32
        )
        extend = [r.extend_len for r in reqs]
        qo_indptr = torch.tensor([0] + extend, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        dtype = self.kvcache.dtype
        if max(extend) == 1:
            self.decode.plan(
                kv_indptr, kv_indices, last_page_len, self.num_qo_heads, self.num_kv_heads,
                self.head_dim, P, pos_encoding_mode="NONE", q_data_type=dtype, kv_data_type=dtype,
            )
            wrapper = self.decode
        else:
            self.prefill.plan(
                qo_indptr, kv_indptr, kv_indices, last_page_len, self.num_qo_heads,
                self.num_kv_heads, self.head_dim, P, causal=True, q_data_type=dtype,
                kv_data_type=dtype,
            )
            wrapper = self.prefill
        batch.attn_metadata = FlashInferMetadata(
            last_indices=(qo_indptr[1:] - 1).to(torch.int64).to(device), wrapper=wrapper
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, FlashInferMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # k_cache(layer) is [num_pages, page_size, Hkv, D]: FlashInfer's NHD layout.
        kv = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return md.wrapper.run(q.contiguous(), kv)
