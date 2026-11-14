"""Triton paged-attention backend (Part III Ch1).

Unlike ``torch_backend.py`` the metadata is tiny: no padded index tensor, no
mask. The kernels get, per request, its page-table row and a few lengths, and
find K/V by themselves (see ``triton_kernels.py``). This is the same contract
as SGLang's ``triton_backend.py`` (``req_to_token`` + ``req_pool_indices``).

Decode batches (one query token per request) go to the split-KV
flash-decoding kernel; anything else goes to the causal extend kernel.

CUDA Graph support (Part II Ch4/7): the decode metadata lives in fixed-size
buffers (``rows``, ``seq_lens``) that ``GraphRunner`` captures once and then
refreshes in place before every replay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..config import ModelConfig
from ..core import Batch
from ..kvcache.mha_pool import MHAKVCache
from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class TritonAttnMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor  # [B] packed index of each request's last query
    rows: torch.Tensor  # [B] page-table row (Req.table_idx)
    seq_lens: torch.Tensor  # [B] total KV length (device_len), int32
    is_decode: bool  # every request has exactly one query token
    # extend only
    qo_start: torch.Tensor | None = None  # [B] exclusive cumsum of extend_lens
    extend_lens: torch.Tensor | None = None  # [B]
    prefix_lens: torch.Tensor | None = None  # [B] cached_len
    max_extend_len: int = 1

    def get_last_indices(self) -> torch.Tensor:
        return self.last_indices


@dataclass
class TritonCaptureBuffers:
    """Static decode metadata for CUDA Graphs, sized for the largest graph."""

    rows: torch.Tensor  # [max_bs]
    seq_lens: torch.Tensor  # [max_bs]
    last_indices: torch.Tensor  # [max_bs] = arange (decode: one token / request)


def _ensure_interpreter_on_cpu(device: torch.device) -> None:
    # Triton only generates GPU code. On CPU the kernels run in the Triton
    # interpreter, which must be switched on *before* `triton.jit` decorates them.
    if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        import sys

        if "minisgl.attention.triton_kernels" in sys.modules:
            raise RuntimeError(
                "triton kernels were compiled for GPU; set TRITON_INTERPRET=1 before importing "
                "minisgl to run the triton backend on CPU"
            )
        os.environ["TRITON_INTERPRET"] = "1"


class TritonAttnBackend(BaseAttnBackend):
    supports_cuda_graph = True

    def __init__(
        self,
        kvcache: MHAKVCache,
        page_table: torch.Tensor,
        config: ModelConfig,
        num_kv_splits: int | None = None,
    ) -> None:
        _ensure_interpreter_on_cpu(kvcache.device)
        from . import triton_kernels  # after the interpreter switch above

        self._kernels = triton_kernels
        self.kvcache = kvcache
        self.page_table = page_table
        self.page_size = kvcache.page_size
        self.scale = config.head_dim**-0.5
        # Flash-decoding splits: more programs per request help when the batch
        # is too small to fill the GPU. 1 = no split, no merge kernel.
        if num_kv_splits is None:
            num_kv_splits = int(os.environ.get("MINISGL_TRITON_KV_SPLITS", "8"))
        self.num_kv_splits = max(1, num_kv_splits)
        self.capture: TritonCaptureBuffers | None = None

    # ------------------------------------------------------------ metadata
    def prepare_metadata(self, batch: Batch) -> None:
        device = self.kvcache.device
        reqs = batch.reqs
        extend_lens = torch.tensor([r.extend_len for r in reqs], dtype=torch.int32)
        seq_lens = torch.tensor([r.device_len for r in reqs], dtype=torch.int32)
        rows = torch.tensor([r.table_idx for r in reqs], dtype=torch.int32)
        cu = torch.cumsum(extend_lens, 0, dtype=torch.int32)
        max_extend = int(extend_lens.max())
        md = TritonAttnMetadata(
            last_indices=(cu - 1).to(torch.int64).to(device, non_blocking=True),
            rows=rows.to(device, non_blocking=True),
            seq_lens=seq_lens.to(device, non_blocking=True),
            is_decode=max_extend == 1,
            max_extend_len=max_extend,
        )
        if not md.is_decode:
            md.qo_start = (cu - extend_lens).to(device, non_blocking=True)
            md.extend_lens = extend_lens.to(device, non_blocking=True)
            md.prefix_lens = torch.tensor([r.cached_len for r in reqs], dtype=torch.int32).to(
                device, non_blocking=True
            )
        batch.attn_metadata = md

    # ------------------------------------------------------------- forward
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, TritonAttnMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # store first, then attend
        q = q.contiguous()
        k_cache = self.kvcache.flat_k_cache(layer_id)
        v_cache = self.kvcache.flat_v_cache(layer_id)
        if md.is_decode:
            return self._kernels.decode_attention(
                q, k_cache, v_cache, self.page_table, md.rows, md.seq_lens,
                sm_scale=self.scale,
                page_size=self.page_size,
                num_kv_splits=self.num_kv_splits,
            )
        return self._kernels.extend_attention(
            q, k_cache, v_cache, self.page_table, md.rows, md.qo_start, md.extend_lens,
            md.prefix_lens,
            max_extend_len=md.max_extend_len,
            sm_scale=self.scale,
            page_size=self.page_size,
        )

    # ---------------------------------------------------------- CUDA Graph
    def init_capture_graph(self, max_bs: int, dummy_row: int) -> None:
        device = self.kvcache.device
        self.capture = TritonCaptureBuffers(
            rows=torch.full((max_bs,), dummy_row, dtype=torch.int32, device=device),
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            last_indices=torch.arange(max_bs, dtype=torch.int64, device=device),
        )

    def capture_metadata(self, bs: int) -> TritonAttnMetadata:
        """Metadata whose tensors are *views* of the static buffers: a graph
        captured with it reads whatever ``prepare_for_replay`` writes there."""
        assert self.capture is not None
        return TritonAttnMetadata(
            last_indices=self.capture.last_indices[:bs],
            rows=self.capture.rows[:bs],
            seq_lens=self.capture.seq_lens[:bs],
            is_decode=True,
        )

    def prepare_for_replay(self, md: BaseAttnMetadata, bs: int, padded_bs: int, dummy_row: int) -> None:
        """Copy the real batch's decode metadata into the static buffers; rows
        ``[bs, padded_bs)`` become the dummy request (row ``dummy_row``, length 1)."""
        assert isinstance(md, TritonAttnMetadata) and md.is_decode and self.capture is not None
        cap = self.capture
        cap.rows[:bs].copy_(md.rows)
        cap.seq_lens[:bs].copy_(md.seq_lens)
        cap.rows[bs:padded_bs].fill_(dummy_row)
        cap.seq_lens[bs:padded_bs].fill_(1)
