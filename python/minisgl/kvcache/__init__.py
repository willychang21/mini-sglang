"""KV cache pool + prefix cache policies (Chapters 4, 7, 8)."""

from __future__ import annotations

import torch

from ..utils import Registry
from .base import BaseCacheHandle, BaseKVCachePool, BasePrefixCache, MatchResult, SizeInfo
from .mha_pool import MHAKVCache
from .naive_cache import NaivePrefixCache
from .radix_cache import RadixPrefixCache

SUPPORTED_CACHE_MANAGER: Registry[BasePrefixCache] = Registry("prefix cache")


def create_kvcache_pool(model_config, num_pages, page_size, dtype, device) -> MHAKVCache:
    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device, page_size: int = 1) -> BasePrefixCache:
    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device, page_size: int = 1) -> BasePrefixCache:
    return RadixPrefixCache(device=device, page_size=page_size)


def create_prefix_cache(device: torch.device, type: str, page_size: int = 1) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device, page_size)


__all__ = [
    "BaseCacheHandle",
    "BaseKVCachePool",
    "BasePrefixCache",
    "MatchResult",
    "SizeInfo",
    "MHAKVCache",
    "NaivePrefixCache",
    "RadixPrefixCache",
    "create_kvcache_pool",
    "create_prefix_cache",
]
