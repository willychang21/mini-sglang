"""Where proposals come from.

* ``ModelDrafter``: a smaller (or truncated) model run autoregressively with its
  own rewindable KV cache.
* ``NgramDrafter``: "prompt lookup" - find the most recent earlier occurrence of
  the current suffix n-gram and propose the tokens that followed it. It costs
  (almost) nothing, so whatever it gets accepted is pure gain; it shines when the
  output copies from the context (summaries, code edits, quoting).

Both return the proposals and, for sampling, the draft distributions ``q``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, List, Tuple

import torch

from ..core import SamplingParams
from ..engine import Engine
from .runner import SequenceRunner
from .verify import logits_to_probs

Proposal = Tuple[List[int], List[torch.Tensor] | None]


class Drafter(ABC):
    forwards = 0  # draft model forward passes (0 for model-free drafters)

    def reset(self) -> None:
        pass

    @abstractmethod
    def propose(
        self, seq: List[int], k: int, sp: SamplingParams, eos: Collection[int], gen: torch.Generator
    ) -> Proposal: ...

    def rollback(self, valid_len: int) -> None:
        """Only the first ``valid_len`` tokens of the confirmed sequence are
        guaranteed to match what the drafter has cached."""


class ModelDrafter(Drafter):
    def __init__(self, engine: Engine) -> None:
        self.runner = SequenceRunner(engine)
        self.vocab_size = engine.model_config.vocab_size
        self.forwards = 0

    def reset(self) -> None:
        self.runner.reset()
        self.forwards = 0

    def propose(self, seq, k, sp, eos, gen) -> Proposal:
        proposals: List[int] = []
        probs: List[torch.Tensor] = []
        feed = seq[self.runner.length :]  # catch up on tokens confirmed last round
        for _ in range(k):
            row = self.runner.extend(feed, all_logits=False)[-1]
            self.forwards += 1
            if sp.is_greedy:
                token = int(row.argmax())
            else:
                q = logits_to_probs(row, sp.temperature, sp.top_k, sp.top_p)
                token = int(torch.multinomial(q, 1, generator=gen))
                probs.append(q)
            proposals.append(token)
            feed = [token]
            if token in eos:
                break
        # the runner now caches seq + proposals[:-1]
        return proposals, (None if sp.is_greedy else probs)

    def rollback(self, valid_len: int) -> None:
        self.runner.truncate(min(self.runner.length, valid_len))


class NgramDrafter(Drafter):
    def __init__(self, vocab_size: int, max_ngram: int = 4, min_ngram: int = 1) -> None:
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def propose(self, seq, k, sp, eos, gen) -> Proposal:
        proposals: List[int] = []
        for n in range(min(self.max_ngram, len(seq) - 1), self.min_ngram - 1, -1):
            suffix = seq[-n:]
            # most recent earlier occurrence of the suffix
            for start in range(len(seq) - n - 1, -1, -1):
                if seq[start : start + n] == suffix:
                    proposals = seq[start + n : start + n + k]
                    break
            if proposals:
                break
        for i, t in enumerate(proposals):
            if t in eos:
                proposals = proposals[: i + 1]
                break
        if sp.is_greedy:
            return proposals, None
        # a deterministic proposal is a one-hot draft distribution
        q = [torch.nn.functional.one_hot(torch.tensor(t), self.vocab_size).float() for t in proposals]
        return proposals, q
