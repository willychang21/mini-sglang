"""RMSNorm, and the fused "add residual + RMSNorm" used between blocks (Chapter 3).

``RMSNormFused.forward(x, residual)``:
  * first call (``residual is None``): save ``x`` as the residual, return norm(x)
  * later calls: ``h = x + residual``; return ``(norm(h), h)``

That is why ``Qwen3DecoderLayer.forward`` never writes ``x + attention(x)``
explicitly: the residual addition is deferred into the next norm.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class RMSNormFused(RMSNorm):
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        return rms_norm(x, self.weight, self.eps), x
