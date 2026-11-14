from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..core import Batch


class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self) -> torch.Tensor:
        """Index (into the packed token dim) of each request's last token."""


class BaseAttnBackend(ABC):
    # Backends that can run decode inside a CUDA Graph implement
    # init_capture_graph / capture_metadata / prepare_for_replay
    # (see triton_backend.py and engine/graph.py).
    supports_cuda_graph: bool = False

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Build ``batch.attn_metadata`` from the batch's lengths and page table."""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """Write the new K/V into the cache, then attend over history + new K/V."""
