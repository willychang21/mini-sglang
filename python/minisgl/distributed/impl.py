"""Collective communication used inside the forward pass (Chapter 9).

Model layers only depend on :class:`DistributedCommunicator`; they never know
whether the backend is Gloo (CPU) or NCCL (GPU). With ``size == 1`` every
collective is the identity, so single-device execution pays nothing.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist

from .info import get_tp_info


class DistributedCommunicator:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if get_tp_info().size == 1:
            return x
        dist.all_reduce(x)
        return x

    def all_gather_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the per-rank shards of ``x`` along the last dimension."""
        size = get_tp_info().size
        if size == 1:
            return x
        x = x.contiguous()
        parts: List[torch.Tensor] = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(parts, x)
        return torch.cat(parts, dim=-1)


_COMM = DistributedCommunicator()


def get_comm() -> DistributedCommunicator:
    return _COMM
