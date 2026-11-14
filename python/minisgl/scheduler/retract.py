"""Optimistic admission and decode retraction (preemption).

Conservative admission (Chapter 6) reserves ``output_len`` KV slots for every
admitted request, so decode can never run out of memory. But most requests stop
long before ``max_tokens`` (EOS), so most of that reservation is never used and
concurrency is far lower than the pool allows.

Optimistic admission reserves only ``output_len * new_token_ratio``. That can
over-commit the pool, so before every decode step the scheduler checks that the
step's new pages fit; if not, it *retracts* running requests until they do:

1. free the request's table row; insert its computed KV into the prefix cache
   (``cache_req``), where it becomes evictable instead of being thrown away;
2. put it back at the front of the waiting queue with prompt + generated
   tokens as its new "prompt" and its remaining budget as ``output_len``;
3. when re-admitted, the radix match usually hits all of it, so the
   re-prefill only recomputes the last token (unless the pages were evicted).

``new_token_ratio`` adapts like SGLang's ``NewTokenRatioTracker``: it decays
linearly toward ``min`` on every decode step that fits (be more optimistic),
and jumps up to an estimate from the surviving requests after a retraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..core import Req
from .config import SchedulerConfig


@dataclass
class NewTokenRatioTracker:
    init: float
    min: float
    decay: float
    current: float
    retract_decode_steps: int

    @classmethod
    def from_config(cls, config: SchedulerConfig) -> "NewTokenRatioTracker":
        if config.admission == "conservative":  # fixed at 1: reserve everything
            return cls(init=1.0, min=1.0, decay=0.0, current=1.0, retract_decode_steps=0)
        init = config.init_new_token_ratio
        min_ratio = min(init * config.min_new_token_ratio_factor, 1.0)
        decay = (init - min_ratio) / config.new_token_ratio_decay_steps
        return cls(init, min_ratio, decay, init, config.retract_decode_steps)

    def decay_step(self) -> None:
        self.current = max(self.current - self.decay, self.min)

    def reset(self) -> None:
        self.current = self.init

    def estimate_after_retract(self, survivors: Sequence[Req]) -> float:
        """SGLang ``estimate_new_token_ratio_after_retract``: assume each survivor
        needs what it already decoded plus ``retract_decode_steps`` more."""
        decoded = sum(len(r.output_ids) for r in survivors)
        budget = sum(r.output_len for r in survivors)
        estimate = (decoded + self.retract_decode_steps * len(survivors)) / (budget + 1)
        return min(1.0, max(self.min, estimate))


def reserved_output_tokens(req: Req, ratio: float) -> int:
    """KV tokens a *running* request is still expected to need.

    ``output_len`` is the budget the request was admitted with and
    ``output_ids`` what it generated since, so with ``ratio == 1`` this is
    exactly ``max_device_len - device_len`` (the conservative bound).
    """
    return max(0, math.ceil(req.output_len * ratio) - len(req.output_ids))


def admission_output_tokens(output_len: int, ratio: float) -> int:
    return math.ceil(output_len * ratio)


def retraction_order(reqs: Iterable[Req], prompt_lens: Dict[int, int]) -> List[Req]:
    """Most-preferred-to-keep first (SGLang ``_get_decode_retraction_order``).

    Keep requests that already generated the most tokens (their work is the
    most expensive to redo if their cached pages get evicted); among equals keep
    shorter prompts. ``uid`` makes the order identical on every TP rank.
    """

    def key(r: Req):
        prompt_len = prompt_lens[r.uid]
        return (len(r.input_ids) - prompt_len, -prompt_len, -r.uid)

    return sorted(reqs, key=key, reverse=True)
