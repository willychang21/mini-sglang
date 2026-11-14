"""Chapter 8: radix tree matching, splitting, locking and LRU eviction."""

from __future__ import annotations

import pytest
import torch

from minisgl.kvcache import RadixPrefixCache


def t(xs):
    return torch.tensor(xs, dtype=torch.int64)


def test_match_insert_and_split():
    cache = RadixPrefixCache(torch.device("cpu"))
    assert cache.match_prefix(t([1, 2, 3])).cuda_handle.cached_len == 0

    prefix_len, handle = cache.insert_prefix(t([10, 11, 12, 13]), t([100, 101, 102, 103]))
    assert prefix_len == 0 and handle.cached_len == 4
    assert cache.size_info.evictable_size == 4

    # partial match splits [10,11,12,13] into [10,11] -> [12,13]
    h = cache.match_prefix(t([10, 11, 99])).cuda_handle
    assert h.cached_len == 2
    assert h.get_matched_indices().tolist() == [100, 101]
    assert cache.num_nodes() == 3  # root + 2 halves

    # inserting a sibling branch reuses the shared prefix
    prefix_len, h2 = cache.insert_prefix(t([10, 11, 20, 21, 22]), t([100, 101, 200, 201, 202]))
    assert prefix_len == 2 and h2.cached_len == 5
    assert h2.get_matched_indices().tolist() == [100, 101, 200, 201, 202]
    full = cache.match_prefix(t([10, 11, 12, 13, 14])).cuda_handle
    assert full.get_matched_indices().tolist() == [100, 101, 102, 103]
    cache.check_integrity()


def test_lock_protects_from_eviction_and_lru_order():
    cache = RadixPrefixCache(torch.device("cpu"))
    _, old = cache.insert_prefix(t([1, 2, 3]), t([10, 11, 12]))
    _, new = cache.insert_prefix(t([5, 6]), t([50, 51]))
    cache.lock_handle(new)
    assert cache.size_info.protected_size == 2 and cache.size_info.evictable_size == 3
    with pytest.raises(RuntimeError):
        cache.evict(4)  # only 3 tokens are evictable
    assert sorted(cache.evict(1).tolist()) == [10, 11, 12]  # whole leaf goes
    assert cache.size_info.evictable_size == 0
    cache.lock_handle(new, unlock=True)
    assert cache.evict(2).tolist() == [50, 51]
    cache.check_integrity()


def test_eviction_cascades_to_parent_and_prefers_old_leaves():
    cache = RadixPrefixCache(torch.device("cpu"))
    cache.insert_prefix(t([1, 2, 3, 4]), t([0, 1, 2, 3]))
    cache.insert_prefix(t([1, 2, 7, 8]), t([0, 1, 6, 7]))  # splits at [1,2]
    cache.match_prefix(t([1, 2, 7, 8, 9]))  # touch the [7,8] branch
    first = cache.evict(1).tolist()
    assert first == [2, 3]  # least recently used leaf
    rest = cache.evict(4).tolist()
    assert sorted(rest) == [0, 1, 6, 7]  # leaf, then its now-childless parent
    assert cache.num_nodes() == 1
    cache.check_integrity()


def test_split_inherits_ref_count():
    cache = RadixPrefixCache(torch.device("cpu"))
    _, h = cache.insert_prefix(t([1, 2, 3, 4]), t([0, 1, 2, 3]))
    cache.lock_handle(h)
    cache.match_prefix(t([1, 2, 9]))  # split while locked
    cache.check_integrity()
    cache.lock_handle(h, unlock=True)
    assert cache.size_info.evictable_size == 4
    cache.check_integrity()


def test_page_aligned_matching():
    cache = RadixPrefixCache(torch.device("cpu"), page_size=4)
    _, h = cache.insert_prefix(t(list(range(10))), t(list(range(100, 110))))
    assert h.cached_len == 8  # unaligned tail is not inserted
    assert cache.match_prefix(t(list(range(7)) + [99])).cuda_handle.cached_len == 4
    assert cache.match_prefix(t(list(range(3)))).cuda_handle.cached_len == 0
    m = cache.match_prefix(t(list(range(9)))).cuda_handle
    assert m.cached_len == 8 and m.get_matched_indices().tolist() == list(range(100, 108))
    cache.check_integrity()
