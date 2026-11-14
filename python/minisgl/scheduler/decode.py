"""The running set: requests that have finished prefill and keep generating."""

from __future__ import annotations

from typing import Iterable, Set

from ..core import Batch, Req


class DecodeManager:
    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.running_reqs: Set[Req] = set()

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    @property
    def inflight_tokens(self) -> int:
        """KV tokens running requests may still allocate (an upper bound)."""
        return sum(r.max_device_len - r.cached_len + self.page_size for r in self.running_reqs)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # sorted by uid: every TP rank builds the same batch in the same order
        return Batch(reqs=sorted(self.running_reqs, key=lambda r: r.uid), phase="decode")
