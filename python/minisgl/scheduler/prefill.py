"""Admission of waiting requests into prefill batches (Chapters 6, 7, 8).

``PrefillAdder`` walks the waiting queue in arrival order (FCFS) and stops at
the first request that does not fit. It enforces three separate limits:

=====================  ======================================================
request slots          ``table_manager.available_size``
compute budget         ``token_budget`` prefill tokens per step (chunking)
KV capacity            ``estimated_len + reserved_size <= available_size``
=====================  ======================================================

``reserved_size`` starts with the tokens running decode requests still need and
grows with every admitted request, so admission never over-commits the pool and
a running request can never run out of KV slots mid-generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from ..core import Batch, ChunkedReq, Req, SamplingParams
from ..kvcache import BaseCacheHandle
from .cache import CacheManager
from .decode import DecodeManager
from .retract import admission_output_tokens
from .table import TableManager


@dataclass(eq=False)
class PendingReq:
    uid: int
    input_ids: torch.Tensor  # CPU int64
    sampling_params: SamplingParams
    output_len: int
    chunked_req: ChunkedReq | None = None  # progress of a chunked prefill
    retracted: bool = False  # preempted while decoding; input_ids include its output

    @property
    def input_len(self) -> int:
        return len(self.input_ids)


@dataclass
class PrefillStats:
    num_reqs: int = 0
    prompt_tokens: int = 0  # tokens of admitted prompts
    cached_tokens: int = 0  # of which reused from the prefix cache


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    stats: PrefillStats = field(default_factory=PrefillStats)
    # 1.0 = conservative admission; < 1 reserves only part of the output budget
    new_token_ratio: float = 1.0
    # speculative decoding: k proposals per round are written before rejection
    spec_reserve: int = 0

    @property
    def page_size(self) -> int:
        return self.cache_manager.page_size

    def _reserve(self, output_len: int) -> int:
        return admission_output_tokens(output_len, self.new_token_ratio) + self.spec_reserve

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None
        # Matching is not read-only: the radix walk splits nodes and refreshes
        # LRU timestamps. Skip it when even a fully cached prompt (only the
        # last token to compute) could not fit, so a request that waits for
        # memory step after step does not keep its prefix "recently used".
        best_case = 1 + self._reserve(req.output_len) + self.page_size
        if best_case + self.reserved_size > self.cache_manager.available_size:
            return None
        handle = self.cache_manager.match_req(req.input_ids).cuda_handle
        cached_len = handle.cached_len
        extend_len = req.input_len - cached_len
        # + page_size: the last partially filled page is allocated whole
        estimated_len = extend_len + self._reserve(req.output_len) + self.page_size

        # First check: does it fit at all?
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # Locking moves the matched prefix from "evictable" to "protected",
        # shrinking available_size, so check again afterwards.
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            # token ids for the model, KV slots for attention
            self.table_manager.token_pool[table_idx, :cached_len] = req.input_ids[:cached_len]
            self.table_manager.page_table[table_idx, :cached_len] = handle.get_matched_indices()
        self.stats.num_reqs += 1
        self.stats.prompt_tokens += req.input_len
        self.stats.cached_tokens += cached_len
        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        cls = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + self._reserve(pending_req.output_len) + self.page_size

        # Only token ids are written now; KV pages are allocated in
        # Scheduler._prepare_batch once the batch is final.
        _slice = slice(cached_len, cached_len + chunk_size)
        self.table_manager.token_pool[table_idx, _slice] = pending_req.input_ids[_slice]
        return cls(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            sampling_params=pending_req.sampling_params,
            cache_handle=cache_handle,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        if chunked := pending_req.chunked_req:  # continue a chunked prefill
            return self._add_one_req(
                pending_req, chunked.cache_handle, chunked.table_idx, chunked.cached_len
            )
        if resource := self._try_allocate_one(pending_req):
            handle, table_idx = resource
            return self._add_one_req(pending_req, handle, table_idx, handle.cached_len)
        return None


class PrefillManager:
    def __init__(
        self,
        cache_manager: CacheManager,
        table_manager: TableManager,
        decode_manager: DecodeManager,
    ) -> None:
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.decode_manager = decode_manager
        self.pending_list: List[PendingReq] = []
        # fraction of output_len reserved at admission; set by the scheduler
        # (1.0 = conservative, < 1 under optimistic admission)
        self.new_token_ratio = 1.0
        self.stats = PrefillStats()

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0

    def add_one_req(self, pending_req: PendingReq) -> None:
        self.pending_list.append(pending_req)

    def add_retracted(self, pending_reqs: List[PendingReq]) -> None:
        """Retracted requests go to the front of the queue (they are older than
        anything waiting), but behind unfinished chunked prefills: those hold
        KV pages and must be allowed to finish, or a retracted request that does
        not fit could block them forever."""
        n_chunked = 0
        while n_chunked < len(self.pending_list) and self.pending_list[n_chunked].chunked_req:
            n_chunked += 1
        retracted = sorted(pending_reqs, key=lambda p: p.uid)
        self.pending_list[n_chunked:n_chunked] = retracted

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None
        new_token_ratio = self.new_token_ratio
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.estimated_inflight_tokens(new_token_ratio),
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            stats=self.stats,
            new_token_ratio=new_token_ratio,
            spec_reserve=self.decode_manager.spec_reserve,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # FCFS: do not let later requests jump the queue
        if len(reqs) == 0:
            return None
        # unfinished chunked prefills go back to the front so they do not starve
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Tuple[bool, Req | None]:
        """Returns (found, req holding resources or None)."""
        for i, pending_req in enumerate(self.pending_list):
            if pending_req.uid == uid:
                self.pending_list.pop(i)
                return True, pending_req.chunked_req
        return False, None
