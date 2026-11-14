"""Vocabulary-parallel embedding and LM head (Chapters 3 & 9).

The vocabulary is split into ``div_ceil(V, tp)`` rows per rank. The embedding
looks up only ids in its local range and all-reduces; the LM head computes local
logits and all-gathers them into the full vocabulary.

During prefill the LM head keeps only the *last* hidden state of each request:
earlier positions would predict tokens that already exist (Chapter 3, Section 4.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import get_global_ctx
from ..distributed import get_comm, get_tp_info
from ..utils import div_ceil


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        tp = get_tp_info()
        self.num_embeddings = num_embeddings
        self.tp_size = tp.size
        self.chunk = div_ceil(num_embeddings, tp.size)
        start = self.chunk * tp.rank
        self.vocab_range = (start, min(start + self.chunk, num_embeddings))
        self.weight = nn.Parameter(torch.empty(self.chunk, embedding_dim), requires_grad=False)
        self._comm = get_comm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(x, self.weight)
        start, end = self.vocab_range
        mask = (x >= start) & (x < end)
        y = F.embedding((x - start).clamp(0, self.chunk - 1) * mask, self.weight)
        y = y * mask.unsqueeze(-1).to(y.dtype)
        return self._comm.all_reduce(y)


class ParallelLMHead(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tied_embedding: VocabParallelEmbedding | None = None,
    ) -> None:
        super().__init__()
        tp = get_tp_info()
        self.num_embeddings = num_embeddings
        self.tp_size = tp.size
        # A tied head shares the embedding's weight; keep it out of the module
        # tree so it is not loaded/saved twice.
        self._tied = [tied_embedding] if tied_embedding is not None else []
        if tied_embedding is None:
            chunk = div_ceil(num_embeddings, tp.size)
            self.weight = nn.Parameter(torch.empty(chunk, embedding_dim), requires_grad=False)
        self._comm = get_comm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill and not batch.return_all_logits:
            x = x[batch.attn_metadata.get_last_indices()].contiguous()
        weight = self._tied[0].weight if self._tied else self.weight
        logits = F.linear(x, weight)
        if self.tp_size == 1:
            return logits
        return self._comm.all_gather_last_dim(logits)[:, : self.num_embeddings]
