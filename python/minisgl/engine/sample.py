"""Choosing the next token from logits (Chapters 2 & 3).

An all-greedy batch takes ``argmax`` directly (softmax preserves ranking, so no
probabilities are needed). Otherwise logits are scaled by temperature, filtered
by top-k / top-p and sampled. Greedy rows inside a mixed batch still use argmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None  # None => whole batch is greedy
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    greedy = temperatures <= 0
    greedy_tokens = torch.argmax(logits, dim=-1)
    temps = torch.where(greedy, torch.ones_like(temperatures), temperatures)
    logits = logits / temps.unsqueeze(1)

    vocab = logits.shape[-1]
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    ranks = torch.arange(vocab, device=logits.device).unsqueeze(0)
    k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab))
    remove = ranks >= k.unsqueeze(1)  # top-k
    probs = torch.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove |= (cum - probs) > top_p.unsqueeze(1)  # top-p: keep the smallest nucleus
    remove[:, 0] = False  # always keep the best token
    probs = torch.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
    choice = torch.multinomial(probs, 1, generator=generator)
    sampled = sorted_idx.gather(1, choice).squeeze(1)
    return torch.where(greedy, greedy_tokens, sampled)


class Sampler:
    def __init__(self, device: torch.device, seed: int = 0) -> None:
        self.device = device
        # Every TP rank uses the same seed and sees identical (all-gathered)
        # logits, so they all sample the same token without extra communication.
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)
        to = lambda xs, dt: torch.tensor(xs, dtype=dt, device=self.device)  # noqa: E731
        return BatchSamplingArgs(
            temperatures=to([0.0 if p.is_greedy else p.temperature for p in params], torch.float32),
            top_k=to([p.top_k for p in params], torch.int64),
            top_p=to([p.top_p for p in params], torch.float32),
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)
        assert args.top_k is not None and args.top_p is not None
        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p, self.generator)
