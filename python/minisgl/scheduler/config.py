"""Scheduler-only knobs (admission policy, overlap loop).

Kept separate from ``EngineConfig`` because none of them affect the model or
the KV pool layout, only *when* requests run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Tuple

Admission = Literal["conservative", "optimistic"]


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

    def __post_init__(self) -> None:
        assert self.admission in ("conservative", "optimistic"), self.admission
        assert 0 < self.init_new_token_ratio <= 1
        assert 0 < self.min_new_token_ratio_factor <= 1
        assert self.new_token_ratio_decay_steps > 0

    @classmethod
    def split_kwargs(cls, kwargs: Dict[str, Any]) -> Tuple["SchedulerConfig", Dict[str, Any]]:
        """Pull scheduler fields out of a mixed ``**kwargs`` (used by ``LLM``)."""
        names = {f.name for f in fields(cls)}
        mine = {k: v for k, v in kwargs.items() if k in names}
        rest = {k: v for k, v in kwargs.items() if k not in names}
        return cls(**mine), rest
