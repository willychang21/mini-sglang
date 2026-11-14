"""Interfaces for the physical KV pool and for prefix-cache policies (Chapter 4).

The two are deliberately independent:

* ``BaseKVCachePool`` stores floating-point K/V. It only knows layers and slots.
* ``BasePrefixCache`` stores *mappings* ``token ids -> slot indices``. It never
  touches K/V values. "naive" and "radix" are two such policies on top of the
  same pool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class BaseCacheHandle(ABC):
    """A reference to a matched prefix. ``cached_len`` tokens can be reused."""

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """Physical token slots of the matched prefix, in sequence order."""


@dataclass
class MatchResult:
    cuda_handle: BaseCacheHandle  # name kept from the reference implementation


@dataclass
class SizeInfo:
    evictable_size: int  # cached tokens nobody is using (can be evicted)
    protected_size: int  # cached tokens locked by running requests

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BasePrefixCache(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult: ...

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None: ...

    @abstractmethod
    def insert_prefix(
        self, input_ids: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[int, BaseCacheHandle]:
        """Register ``input_ids -> indices``.

        Returns ``(prefix_len, handle)`` where ``prefix_len`` is how much of the
        sequence was *already* in the cache before this call (the caller's own
        slots for that range are duplicates and may be freed) and ``handle``
        covers everything that is now cached.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """Evict at least ``size`` tokens, returning their slot indices."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    def check_integrity(self) -> None:  # pragma: no cover - optional
        pass
