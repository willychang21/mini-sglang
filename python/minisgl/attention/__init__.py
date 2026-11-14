"""Attention backends (Part II Ch4, Part III Ch1 "Attention Backends").

Every backend implements the same two calls (see ``base.py``):

* ``prepare_metadata(batch)``  once per step, on the host: turn the batch's
  lengths + page table rows into whatever the kernel wants.
* ``forward(q, k, v, layer_id, batch)``  once per layer: store K/V, attend.

Available backends (``--attention-backend``):

``torch``       gather the K/V history through the page table into a padded
                tensor and call SDPA. Runs anywhere; the readable reference.
``triton``      real paged attention: Triton kernels read K/V *through* the page
                table (no gather), split-KV flash-decoding for decode. Runs on
                GPU, and on CPU through the Triton interpreter (slow, for tests).
``flashinfer``  registered only when the ``flashinfer`` package is importable.
"""

from __future__ import annotations

import importlib.util
from typing import Protocol

import torch

from ..config import ModelConfig
from ..kvcache.mha_pool import MHAKVCache
from ..utils import Registry
from .base import BaseAttnBackend, BaseAttnMetadata
from .torch_backend import TorchAttnBackend, TorchAttnMetadata


class BackendCreator(Protocol):
    def __call__(
        self, kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig
    ) -> BaseAttnBackend: ...


SUPPORTED_ATTENTION_BACKENDS: Registry[BaseAttnBackend] = Registry("attention backend")


@SUPPORTED_ATTENTION_BACKENDS.register("torch")
def _create_torch(kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig):
    return TorchAttnBackend(kvcache, page_table)


@SUPPORTED_ATTENTION_BACKENDS.register("triton")
def _create_triton(kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig):
    from .triton_backend import TritonAttnBackend  # imports triton lazily

    return TritonAttnBackend(kvcache, page_table, config)


if importlib.util.find_spec("flashinfer") is not None:  # optional, never required

    @SUPPORTED_ATTENTION_BACKENDS.register("flashinfer")
    def _create_flashinfer(kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig):
        from .flashinfer_backend import FlashInferAttnBackend

        return FlashInferAttnBackend(kvcache, page_table, config)


def default_attention_backend(device: str | torch.device) -> str:
    # CPU: the readable SDPA backend (the Triton interpreter is far too slow to
    # serve with). CUDA: real paged attention.
    if torch.device(device).type == "cuda" and importlib.util.find_spec("triton") is not None:
        return "triton"
    return "torch"


def create_attention_backend(
    name: str, kvcache: MHAKVCache, page_table: torch.Tensor, config: ModelConfig
) -> BaseAttnBackend:
    return SUPPORTED_ATTENTION_BACKENDS[name](kvcache, page_table, config)


__all__ = [
    "BaseAttnBackend",
    "BaseAttnMetadata",
    "SUPPORTED_ATTENTION_BACKENDS",
    "TorchAttnBackend",
    "TorchAttnMetadata",
    "create_attention_backend",
    "default_attention_backend",
]
