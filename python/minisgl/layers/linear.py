"""Tensor-parallel linear layers (Chapter 9).

* Column parallel: split the *output* dim. Each rank computes a slice of the
  outputs, no communication needed (QKV, gate/up projections).
* Row parallel: split the *input* dim. Each rank computes a partial sum that is
  combined with all-reduce (o_proj, down_proj).

Column -> Row pairs therefore need exactly one all-reduce per block.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import get_comm, get_tp_info
from ..utils import div_even


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape), requires_grad=False)


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int], bias: bool = False) -> None:
        super().__init__()
        tp = get_tp_info().size
        self.output_size_per_partition = sum(div_even(o, tp) for o in output_sizes)
        self.weight = _param(self.output_size_per_partition, input_size)
        self.bias = _param(self.output_size_per_partition) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(ColumnParallelLinear):
    """Fused Q/K/V projection. KV heads are replicated when there are fewer KV
    heads than TP ranks (GQA)."""

    def __init__(
        self, hidden_size: int, head_dim: int, num_qo_heads: int, num_kv_heads: int, bias: bool = False
    ) -> None:
        tp = get_tp_info().size
        self.local_qo_heads = div_even(num_qo_heads, tp)
        self.local_kv_heads = div_even(num_kv_heads, tp, allow_replicate=True)
        self.head_dim = head_dim
        q = self.local_qo_heads * head_dim * tp
        kv = self.local_kv_heads * head_dim * tp
        super().__init__(hidden_size, [q, kv, kv], bias)


class RowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        tp = get_tp_info().size
        self.input_size_per_partition = div_even(input_size, tp)
        self.weight = _param(output_size, self.input_size_per_partition)
        self.bias = _param(output_size) if bias else None
        self._comm = get_comm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        y = self._comm.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y
