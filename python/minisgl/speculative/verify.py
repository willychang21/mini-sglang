"""Verification rules for speculative decoding (Chapter 10, Sections 4-5).

Alignment: with confirmed prefix length ``L`` and ``k`` proposals, the target's
rows ``L-1 .. L+k-2`` verify the proposals and row ``L+k-1`` gives the bonus
token after full acceptance. Callers pass exactly those ``k+1`` rows.
"""

from __future__ import annotations

from typing import Collection, List

import torch


def greedy_commit(
    proposals: List[int],
    verification: torch.Tensor,
    remaining: int,
    eos_ids: Collection[int] = (),
) -> List[int]:
    """Commit the accepted prefix plus one correction (or bonus) token.

    Every committed token is the target's argmax under the confirmed prefix, so
    the output equals target-only greedy decoding.
    """
    return greedy_commit_ids(proposals, verification.argmax(dim=-1).tolist(), remaining, eos_ids)


def greedy_commit_ids(
    proposals: List[int],
    target_ids: List[int],
    remaining: int,
    eos_ids: Collection[int] = (),
) -> List[int]:
    """``greedy_commit`` on precomputed ``argmax`` ids (``len(proposals) + 1``
    of them). The batched scheduler path takes one argmax over the whole
    verify batch and moves it to the host once, instead of once per row."""
    if remaining == 0:
        return []
    output: List[int] = []
    for i, proposed in enumerate(proposals):
        token = int(target_ids[i])
        output.append(token)
        if token in eos_ids or len(output) == remaining:
            return output
        if token != proposed:  # first rejection: later rows saw a wrong prefix
            return output
    output.append(int(target_ids[len(proposals)]))  # bonus token
    return output


def rejection_commit(
    proposals: List[int],
    draft_probs: torch.Tensor,  # [k, V]  q: the draft's *sampling* distributions
    target_probs: torch.Tensor,  # [k+1, V] p: the target's sampling distributions
    remaining: int,
    eos_ids: Collection[int] = (),
    generator: torch.Generator | None = None,
) -> List[int]:
    """Speculative sampling (Leviathan et al., Algorithm 1).

    Accept proposal ``y`` with probability ``min(1, p[y] / q[y])``; on the
    first rejection sample a correction from ``normalize(max(p - q, 0))``. The
    committed tokens are distributed exactly as if sampled from ``p``.
    """
    if remaining == 0:
        return []
    output: List[int] = []
    for i, y in enumerate(proposals):
        p, q = target_probs[i], draft_probs[i]
        accept_prob = min(1.0, float(p[y] / q[y])) if q[y] > 0 else 0.0
        if float(torch.rand((), generator=generator)) < accept_prob:
            token = y
        else:
            residual = (p - q).clamp_min(0)
            total = float(residual.sum())
            dist = residual / total if total > 1e-12 else p  # numerical fallback
            token = int(torch.multinomial(dist, 1, generator=generator))
        output.append(token)
        if token in eos_ids or len(output) == remaining or token != y:
            return output
    bonus = int(torch.multinomial(target_probs[len(proposals)], 1, generator=generator))
    output.append(bonus)
    return output


def logits_to_probs(logits: torch.Tensor, temperature: float, top_k: int = -1, top_p: float = 1.0) -> torch.Tensor:
    """The *processed* sampling distribution (temperature, top-k, top-p)."""
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
        remove = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs
