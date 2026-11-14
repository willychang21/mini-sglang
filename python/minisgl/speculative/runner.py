"""Run one sequence on an Engine with an explicit, rewindable KV cache.

Speculative decoding needs two things the continuous-batching scheduler does
not: logits for *every* position of a step (verification rows) and the ability
to roll the cache back after a rejection. ``SequenceRunner`` reserves one
page-table row with a fixed contiguous slot range, so rolling back is just
moving ``length`` backwards: stale K/V beyond it is never read (attention only
looks at ``[0, device_len)``) and is overwritten by the next extend.
"""

from __future__ import annotations

from typing import List

import torch

from ..core import Batch, Req, SamplingParams
from ..engine import Engine
from ..kvcache.naive_cache import NaiveCacheHandle


class SequenceRunner:
    def __init__(self, engine: Engine, row: int = 0) -> None:
        self.engine = engine
        self.row = row
        n = engine.max_seq_len
        engine.page_table[row, :n] = torch.arange(n, device=engine.device)
        self.length = 0  # number of leading tokens whose K/V is valid

    def reset(self) -> None:
        self.length = 0

    def truncate(self, length: int) -> None:
        assert 0 <= length <= self.length
        self.length = length

    @torch.inference_mode()
    def extend(self, tokens: List[int], all_logits: bool = True) -> torch.Tensor:
        """Append ``tokens`` after the cached prefix; returns ``[len(tokens), V]``
        logits (or ``[1, V]`` for the last token only)."""
        assert len(tokens) > 0
        engine = self.engine
        start, n = self.length, len(tokens)
        if start + n > engine.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        req = Req(
            input_ids=torch.zeros(start + n, dtype=torch.int64),  # only lengths matter here
            table_idx=self.row,
            cached_len=start,
            output_len=0,
            uid=0,
            sampling_params=SamplingParams(),
            cache_handle=NaiveCacheHandle(engine.device),
        )
        batch = Batch(reqs=[req], phase="prefill", return_all_logits=all_logits)
        batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=engine.device)
        batch.positions = torch.arange(start, start + n, device=engine.device)
        batch.out_loc = engine.page_table[self.row, start : start + n]
        engine.attn_backend.prepare_metadata(batch)
        out = engine.forward_batch(batch, None)
        self.length = start + n
        return out.logits.float()
