"""Scheduler-only knobs (admission policy, overlap loop).

Kept separate from ``EngineConfig`` because none of them affect the model or
the KV pool layout, only *when* requests run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Tuple

Admission = Literal["conservative", "optimistic"]
SpecAlgorithm = Literal["none", "ngram", "draft_model"]


@dataclass
class SchedulerConfig:
    # --- admission (Chapter 6 + SGLang's new_token_ratio) ---
    # conservative: reserve every request's full output budget -> never retracts.
    # optimistic:   reserve output_len * new_token_ratio and retract decode
    #               requests when a decode step does not fit (preemption).
    admission: Admission = "conservative"
    # Same defaults as SGLang (SGLANG_INIT_NEW_TOKEN_RATIO, ..._MIN_..._FACTOR,
    # ..._DECAY_STEPS, SGLANG_RETRACT_DECODE_STEPS).
    init_new_token_ratio: float = 0.7
    min_new_token_ratio_factor: float = 0.14
    new_token_ratio_decay_steps: int = 600
    retract_decode_steps: int = 20
    # --- loop (Chapter 6 overlap_loop / SGLang event_loop_overlap) ---
    disable_overlap: bool = True
    # --- speculative decoding in the scheduler (docs/speculative-in-scheduler.md) ---
    # none | ngram (prompt lookup, no model) | draft_model (a smaller model with
    # its own KV pool that shares the target's page table and allocator).
    speculative_algorithm: SpecAlgorithm = "none"
    speculative_num_draft_tokens: int = 4  # k: proposals per request per round
    speculative_draft_model: str | None = None  # default: the target checkpoint
    speculative_draft_num_layers: int | None = None  # keep only the first N layers

    def __post_init__(self) -> None:
        assert self.admission in ("conservative", "optimistic"), self.admission
        assert 0 < self.init_new_token_ratio <= 1
        assert 0 < self.min_new_token_ratio_factor <= 1
        assert self.new_token_ratio_decay_steps > 0
        assert self.speculative_algorithm in ("none", "ngram", "draft_model"), self.speculative_algorithm
        assert self.speculative_num_draft_tokens >= 1

    @property
    def speculative(self) -> bool:
        return self.speculative_algorithm != "none"

    @property
    def spec_reserve_tokens(self) -> int:
        """Extra KV tokens a decoding request may hold *during* a round: the
        k proposals are written before we know how many are accepted."""
        return self.speculative_num_draft_tokens if self.speculative else 0

    @classmethod
    def split_kwargs(cls, kwargs: Dict[str, Any]) -> Tuple["SchedulerConfig", Dict[str, Any]]:
        """Pull scheduler fields out of a mixed ``**kwargs`` (used by ``LLM``)."""
        names = {f.name for f in fields(cls)}
        mine = {k: v for k, v in kwargs.items() if k in names}
        rest = {k: v for k, v in kwargs.items() if k not in names}
        return cls(**mine), rest
