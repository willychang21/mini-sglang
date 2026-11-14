"""Rotary position embedding (NeoX style, as used by Qwen3)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float, device=None) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim)
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_pos, D/2]
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        """``q``: ``[N, Hq, D]``, ``k``: ``[N, Hkv, D]``, ``positions``: ``[N]``."""
        cos = self.cos_cache[positions].unsqueeze(1)  # [N, 1, D/2]
        sin = self.sin_cache[positions].unsqueeze(1)
        return _apply(q, cos, sin), _apply(k, cos, sin)


def _apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
