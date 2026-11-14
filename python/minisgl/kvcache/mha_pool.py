"""Physical KV storage for all layers (Chapter 4, Section 4.1).

Layout: ``[2 (K/V), num_layers, num_pages, page_size, local_kv_heads, head_dim]``.
``store_kv`` views a layer as a flat ``[num_pages * page_size, heads, dim]``
token-slot array and scatters the new K/V into the ``out_loc`` slots.
"""

from __future__ import annotations

import torch

from ..distributed import get_tp_info
from ..utils import div_even
from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.zeros(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def flat_k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index].view(self._storage_shape)

    def flat_v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index].view(self._storage_shape)

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # Pure-PyTorch replacement for the reference `store_cache` CUDA kernel.
        self.flat_k_cache(layer_id)[out_loc] = k.view(-1, *self._storage_shape[1:])
        self.flat_v_cache(layer_id)[out_loc] = v.view(-1, *self._storage_shape[1:])

    @property
    def device(self) -> torch.device:
        return self._kv_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_bytes(self) -> int:
        return self._kv_buffer.numel() * self._kv_buffer.element_size()
