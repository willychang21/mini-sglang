"""Chapter 3 baseline: autoregressive generation *without* a KV cache.

Every step feeds the entire sequence through the model again
(``cached_len = 0``), exactly like::

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = argmax(logits[-1])
        input_ids = append(input_ids, next_token)

The engine's KV pool is only used as scratch space for the current step, so the
work per step grows with the sequence length: O(n^2) over a generation. Chapter
4's KV cache makes each decode step process a single token instead.
"""

from __future__ import annotations

from typing import List

import torch

from ..core import Batch, Req, SamplingParams
from ..engine import Engine
from ..kvcache.naive_cache import NaiveCacheHandle


@torch.inference_mode()
def generate_without_kv_cache(
    engine: Engine, input_ids: List[int], max_new_tokens: int, eos_ids=()
) -> List[int]:
    tokens = list(input_ids)
    output: List[int] = []
    row = 0
    slots = torch.arange(engine.max_seq_len, device=engine.device)  # scratch slots
    for _ in range(max_new_tokens):
        req = Req(
            input_ids=torch.tensor(tokens),
            table_idx=row,
            cached_len=0,  # nothing reused: recompute the whole prefix
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(),
            cache_handle=NaiveCacheHandle(engine.device),
        )
        n = len(tokens)
        engine.page_table[row, :n] = slots[:n]
        batch = Batch(reqs=[req], phase="prefill")
        batch.input_ids = torch.tensor(tokens, device=engine.device)
        batch.positions = torch.arange(n, device=engine.device)
        batch.out_loc = engine.page_table[row, :n]
        engine.attn_backend.prepare_metadata(batch)
        out = engine.forward_batch(batch, engine.sampler.prepare(batch))
        token = int(out.next_tokens_cpu[0])
        output.append(token)
        tokens.append(token)
        if token in eos_ids:
            break
    return output
