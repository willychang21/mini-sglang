"""Draft-then-verify generation (Chapter 10).

Invariants between rounds (``seq`` = confirmed tokens):

* the target cache holds ``seq[:-1]``: the newest confirmed token (a correction
  or bonus token) has never been run through the target, so its K/V cannot come
  from a rejected candidate (Chapter 10, Section 5.2).
* the drafter caches some prefix of ``seq``; it catches up at the next round.

One round::

    draft:   propose d1..dk
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
from .drafter import Drafter, ModelDrafter
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
        """Tokens committed per verification forward (1.0 = no gain)."""
        return (self.accepted + self.rounds) / max(self.rounds, 1)


class SpeculativeDecoder:
    def __init__(
        self, target: Engine, draft: Engine | Drafter, num_draft_tokens: int = 4, seed: int = 0
    ) -> None:
        assert num_draft_tokens >= 1
        drafter = ModelDrafter(draft) if isinstance(draft, Engine) else draft
        tv, dv = target.model_config.vocab_size, getattr(drafter, "vocab_size", None)
        if dv is not None and tv != dv:
            raise ValueError(f"target/draft vocabularies differ ({tv} vs {dv})")
        self.target = SequenceRunner(target)
        self.drafter = drafter
        self.k = num_draft_tokens
        self.generator = torch.Generator().manual_seed(seed)
        self.eos_ids = set(target.model_config.eos_token_ids)

    def _probs(self, logits: torch.Tensor, sp: SamplingParams) -> torch.Tensor:
        return logits_to_probs(logits, sp.temperature, sp.top_k, sp.top_p)

    def generate(self, prompt: List[int], sp: SamplingParams) -> tuple[List[int], SpecStats]:
        stats = SpecStats()
        t0 = time.perf_counter()
        eos = set() if sp.ignore_eos else self.eos_ids
        self.target.reset()
        self.drafter.reset()

        # Prefill: the first output token comes from the target alone.
        last = self.target.extend(prompt, all_logits=False)[-1]
        stats.target_forwards += 1
        if sp.is_greedy:
            first = int(last.argmax())
        else:
            first = int(torch.multinomial(self._probs(last, sp), 1, generator=self.generator))
        seq = list(prompt) + [first]
        output = [first]

        while len(output) < sp.max_tokens and output[-1] not in eos:
            remaining = sp.max_tokens - len(output)
            L = len(seq)
            proposals, draft_probs = self.drafter.propose(
                seq, min(self.k, remaining), sp, eos, self.generator
            )

            # one target forward verifies every proposal (+1 bonus row)
            verification = self.target.extend([seq[-1]] + proposals, all_logits=True)
            stats.target_forwards += 1
            if sp.is_greedy:
                committed = greedy_commit(proposals, verification, remaining, eos)
            else:
                q = torch.stack(draft_probs) if proposals else verification.new_zeros(0, verification.shape[-1])
                committed = rejection_commit(
                    proposals, q, self._probs(verification, sp), remaining, eos, self.generator
                )

            n_acc = 0  # proposals taken as-is
            while n_acc < min(len(proposals), len(committed)) and committed[n_acc] == proposals[n_acc]:
                n_acc += 1
            stats.rounds += 1
            stats.proposed += len(proposals)
            stats.accepted += n_acc
            stats.accepted_per_round.append(n_acc)

            seq.extend(committed)
            output.extend(committed)
            # rewind both caches to the confirmed path
            self.target.truncate(min(L + n_acc, len(seq) - 1))
            self.drafter.rollback(min(L + n_acc, len(seq) - 1))

        stats.draft_forwards = self.drafter.forwards
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
            probs = logits_to_probs(logits, sp.temperature, sp.top_k, sp.top_p)
            token = int(torch.multinomial(probs, 1, generator=gen))
        output.append(token)
        if len(output) >= sp.max_tokens or token in eos:
            return output
        logits = runner.extend([token], all_logits=False)[-1]
