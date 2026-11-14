"""CacheManager: free pages + prefix cache + page table (Chapters 4, 7, 8).

* ``free_slots`` holds the *first token slot* of every free page.
* ``allocate_paged`` gives each request whole pages covering ``[cached_len, device_len)``.
* When pages run out, unreferenced radix-tree leaves are evicted (LRU).
* ``cache_req`` hands a request's computed prefix to the prefix cache and frees
  whatever is duplicated or can never be shared.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Tuple

import torch

from ..core import Req
from ..kvcache import BaseCacheHandle, BasePrefixCache, MatchResult, create_prefix_cache
from ..utils import div_ceil


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        n = (last_page - first_page) * page_size
        page_table[table_idx, first_page * page_size : last_page * page_size] = allocated[offset : offset + n]
        offset += n
    assert offset == len(allocated)


class CacheManager:
    def __init__(
        self,
        device: torch.device,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        cache_type: str = "radix",
    ) -> None:
        self.device = device
        self.num_pages = num_pages
        self.page_size = page_size
        self.page_table = page_table
        self.free_slots = torch.arange(num_pages, dtype=torch.int64, device=device) * page_size
        self.prefix_cache: BasePrefixCache = create_prefix_cache(device, cache_type, page_size)
        self._lazy_free: List[torch.Tensor] | None = None

    # --------------------------------------------------------------- capacity
    @property
    def available_size(self) -> int:
        """Tokens that can be handed out right now: free pages + evictable cache."""
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    @property
    def available_pages(self) -> int:
        """Whole pages that ``_allocate`` can hand out right now."""
        return len(self.free_slots) + self.prefix_cache.size_info.evictable_size // self.page_size

    def pages_needed(self, reqs: List[Req]) -> int:
        """Pages ``allocate_paged(reqs)`` would take (same arithmetic)."""
        return sum(
            max(0, div_ceil(r.device_len, self.page_size) - div_ceil(r.cached_len, self.page_size))
            for r in reqs
        )

    # ----------------------------------------------------------- prefix cache
    def match_req(self, input_ids: torch.Tensor) -> MatchResult:
        input_len = len(input_ids)
        assert input_len > 0
        # Never match the final prompt token: prefill must still run it to get
        # the logits for the first output token.
        return self.prefix_cache.match_prefix(input_ids[: input_len - 1])

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    # ------------------------------------------------------- page allocation
    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(self.page_size, device=self.device, dtype=pages.dtype)
        return (pages.unsqueeze(1) + offsets).flatten()

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        if needed_pages > len(self.free_slots):
            raise RuntimeError(f"out of KV cache: need {needed_pages} pages, have {len(self.free_slots)}")
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) == 0:
            return
        pages = indices[:: self.page_size]
        if self._lazy_free is not None:
            self._lazy_free.append(pages)
        else:
            self.free_slots = torch.cat([self.free_slots, pages])

    @contextmanager
    def lazy_free_region(self):
        """Batch many small ``_free`` calls into a single concatenation."""
        self._lazy_free = []
        try:
            yield
        finally:
            pending, self._lazy_free = self._lazy_free, None
            if pending:
                self.free_slots = torch.cat([self.free_slots, *pending])

    # ------------------------------------------------------------ cache_req
    def cache_req(self, req: Req, *, finished: bool) -> None:
        """Insert ``[0, req.cached_len)`` into the prefix cache.

        Regions of the request's page-table row::

            [0, old.cached_len)          matched from the cache at admission
            [old.cached_len, prefix_len) duplicates: someone else inserted the same
                                         tokens meanwhile -> free our copy
            [prefix_len, new.cached_len) newly inserted, now owned by the cache
            [new.cached_len, cached_len) unaligned tail; free it when finished
        """
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        prefix_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : prefix_len].clone())
        if finished:
            self._free(page_indices[new_handle.cached_len :].clone())
        else:
            if prefix_len > old_handle.cached_len:
                # Fix vs. the reference: the duplicate slots we just freed may be
                # reallocated, so point the still-running request at the cached
                # copy of the same tokens instead.
                matched = new_handle.get_matched_indices()
                self.page_table[req.table_idx, old_handle.cached_len : prefix_len] = matched[
                    old_handle.cached_len : prefix_len
                ]
            req.cache_handle = new_handle
            self.lock(new_handle)

    def release_for_retract(self, req: Req, short_pages: int) -> None:
        """Release a retracted request's KV.

        The decode step that triggered the retraction is ``short_pages`` pages
        short. Return that many pages from the *tail* of the request straight
        to the free list and hand the rest to the prefix cache. If we inserted
        everything instead, the allocation right after would evict the whole
        (freshest, often only evictable) radix leaf just to get a few pages and
        the resumed request would have to recompute all of it.

        Only pages the request owns can be freed: ``[0, cache_handle.cached_len)``
        belongs to the cache (matched at admission) and stays there.
        """
        ps = self.page_size
        owned_from_page = req.cache_handle.cached_len // ps
        end_page = div_ceil(req.cached_len, ps)
        keep_len = max(owned_from_page, end_page - max(short_pages, 0)) * ps
        if keep_len < req.cached_len:
            self._free(self.page_table[req.table_idx, keep_len : req.cached_len].clone())
            req.cached_len = keep_len  # the request object is discarded after this
        self.cache_req(req, finished=True)

    def free_spec_tail(self, req: Req, alloc_len: int) -> None:
        """After speculative verification: return the pages past the confirmed
        path. The round allocated pages for ``[0, alloc_len)``; the request
        keeps the pages covering ``[0, cached_len)`` (the page holding
        position ``cached_len`` too, if it is partially filled: its stale
        slots are overwritten next round). Pages that only ever held rejected
        proposals go straight back to the free list - they were never in the
        prefix cache."""
        ps = self.page_size
        keep = div_ceil(req.cached_len, ps) * ps
        end = div_ceil(alloc_len, ps) * ps
        if end > keep:
            self._free(self.page_table[req.table_idx, keep:end].clone())

    # -------------------------------------------------------------- checking
    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                f"page leak: free={len(self.free_slots)} cached={cache_pages} total={self.num_pages}"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)
        assert len(torch.unique(self.free_slots)) == len(self.free_slots), "page freed twice"
