"""The "naive" prefix cache: never reuses anything across requests."""

from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    def __init__(self, device: torch.device) -> None:
        self.cached_len = 0
        self._device = device

    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int64, device=self._device)


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle(self.device))

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def insert_prefix(
        self, input_ids: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[int, BaseCacheHandle]:
        return 0, NaiveCacheHandle(self.device)

    def evict(self, size: int) -> torch.Tensor:
        if size > 0:
            raise RuntimeError("naive prefix cache holds nothing to evict")
        return torch.empty(0, dtype=torch.int64, device=self.device)

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)
