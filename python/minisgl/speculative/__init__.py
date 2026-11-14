"""Speculative decoding (Chapter 10)."""

from .decoder import SpecStats, SpeculativeDecoder, autoregressive_generate
from .runner import SequenceRunner
from .verify import greedy_commit, logits_to_probs, rejection_commit

__all__ = [
    "SequenceRunner",
    "SpecStats",
    "SpeculativeDecoder",
    "autoregressive_generate",
    "greedy_commit",
    "logits_to_probs",
    "rejection_commit",
]
