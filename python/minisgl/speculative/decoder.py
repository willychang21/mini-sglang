"""Draft-then-verify generation with a separate (smaller) draft model.

Invariants between rounds (``seq`` = confirmed tokens):

* target cache holds ``seq[:-1]``: the newest confirmed token (a correction or
  bonus token) has never been run through the target, so its K/V cannot come
  from a rejected candidate (Chapter 10, Section 5.2).
* draft cache holds some prefix of ``seq``; it catches up at the next round.

One round::

    draft:   propose d1..dk one by one (k cheap forward passes)
    target:  extend([seq[-1], d1..dk]) -> k+1 rows in ONE forward pass
    commit:  accepted prefix + correction/bonus
    rewind:  truncate both caches to the confirmed path
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import torch

from ..core import SamplingParams
from ..engine import Engine
from .runner import SequenceRunner
from .verify import greedy_commit, logits_to_probs, rejection_commit


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    elapsed: float = 0.0
    accepted_per_round: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)

    @property
    def tokens_per_target_forward(self) -> float:
        return (self.accepted + self.rounds) / max(self.rounds, 1)


class SpeculativeDecoder:
    def __init__(self, target: Engine, draft: Engine, num_draft_tokens: int = 4, seed: int = 0) -> None:
        assert num_draft_tokens >= 1
        tv, dv = target.model_config.vocab_size, draft.model_config.vocab_size
        if tv != dv:
            raise ValueError(f"target/draft vocabularies differ ({tv} vs {dv})")
        self.target = SequenceRunner(target)
        self.draft = SequenceRunner(draft)
        self.k = num_draft_tokens
        self.generator = torch.Generator().manual_seed(seed)
        self.eos_ids = set(target.model_config.eos_token_ids)

    def _probs(self, logits: torch.Tensor, sp: SamplingParams) -> torch.Tensor:
        return logits_to_probs(logits, sp.temperature, sp.top_k, sp.top_p)

    def _pick(self, logits: torch.Tensor, sp: SamplingParams) -> int:
        if sp.is_greedy:
            return int(logits.argmax())
        return int(torch.multinomial(self._probs(logits, sp), 1, generator=self.generator))

    def generate(self, prompt: List[int], sp: SamplingParams) -> tuple[List[int], SpecStats]:
        stats = SpecStats()
        t0 = time.perf_counter()
        eos = set() if sp.ignore_eos else self.eos_ids
        self.target.reset()
        self.draft.reset()

        # Prefill: the first output token comes from the target alone.
        logits = self.target.extend(prompt, all_logits=False)
        stats.target_forwards += 1
        first = self._pick(logits[-1], sp)
        seq = list(prompt) + [first]
        output = [first]
        # target cache now holds `prompt` == seq[:-1]

        while len(output) < sp.max_tokens and output[-1] not in eos:
            remaining = sp.max_tokens - len(output)
            k = min(self.k, remaining)
            L = len(seq)

            # ---- draft k proposals (the draft first catches up on seq) ----
            proposals: List[int] = []
            draft_probs: List[torch.Tensor] = []
            feed = seq[self.draft.length :]
            for _ in range(k):
                row = self.draft.extend(feed, all_logits=False)[-1]
                stats.draft_forwards += 1
                if sp.is_greedy:
                    token = int(row.argmax())
                else:
                    q = self._probs(row, sp)
                    token = int(torch.multinomial(q, 1, generator=self.generator))
                    draft_probs.append(q)
                proposals.append(token)
                feed = [token]
                if token in eos:
                    break
            # draft cache now holds seq + proposals[:-1]

            # ---- verify all proposals with ONE target forward ----
            verification = self.target.extend([seq[-1]] + proposals, all_logits=True)
            stats.target_forwards += 1
            if sp.is_greedy:
                committed = greedy_commit(proposals, verification, remaining, eos)
            else:
                target_probs = self._probs(verification, sp)
                committed = rejection_commit(
                    proposals, torch.stack(draft_probs), target_probs, remaining, eos, self.generator
                )

            # number of proposals that were accepted as-is
            n_acc = 0
            while n_acc < len(proposals) and n_acc < len(committed) and committed[n_acc] == proposals[n_acc]:
                n_acc += 1
            n_acc = min(n_acc, len(committed))
            stats.rounds += 1
            stats.proposed += len(proposals)
            stats.accepted += n_acc
            stats.accepted_per_round.append(n_acc)

            seq.extend(committed)
            output.extend(committed)
            # ---- rewind both caches to the confirmed path ----
            self.target.truncate(min(L + n_acc, len(seq) - 1))
            self.draft.truncate(min(self.draft.length, len(seq) - 1, L + n_acc))

        stats.elapsed = time.perf_counter() - t0
        return output[: sp.max_tokens], stats


@torch.inference_mode()
def autoregressive_generate(engine: Engine, prompt: List[int], sp: SamplingParams, seed: int = 0) -> List[int]:
    """Plain target-only decoding with the same runner, for comparisons."""
    runner = SequenceRunner(engine)
    gen = torch.Generator().manual_seed(seed)
    eos = set() if sp.ignore_eos else set(engine.model_config.eos_token_ids)
    logits = runner.extend(prompt, all_logits=False)[-1]
    output: List[int] = []
    while True:
        if sp.is_greedy:
            token = int(logits.argmax())
        else:
            token = int(torch.multinomial(logits_to_probs(logits, sp.temperature, sp.top_k, sp.top_p), 1, generator=gen))
        output.append(token)
        if len(output) >= sp.max_tokens or token in eos:
            return output
        logits = runner.extend([token], all_logits=False)[-1]
