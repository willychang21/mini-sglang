"""Attention and MLP blocks shared by decoder models (Chapter 3, Sections 3.3-3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..core import get_global_ctx
from ..distributed import get_tp_info
from ..layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding


class RopeAttn(nn.Module):
    """QKV projection -> (QK norm) -> RoPE -> paged attention -> output projection."""

    def __init__(
        self, config: ModelConfig, layer_id: int, rotary: RotaryEmbedding, has_qk_norm: bool = True
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, config.head_dim, config.num_qo_heads, config.num_kv_heads
        )
        self.num_qo_heads = self.qkv_proj.local_qo_heads
        self.num_kv_heads = self.qkv_proj.local_kv_heads
        self.o_proj = RowParallelLinear(config.num_qo_heads * config.head_dim, config.hidden_size)
        if has_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None
        self._rotary = [rotary]  # shared across layers, not a submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        qkv = self.qkv_proj(x)
        q_size = self.num_qo_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.reshape(-1, self.num_qo_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:  # Qwen3 normalizes each Q/K head; V is untouched
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self._rotary[0](batch.positions, q, k)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)
        return self.o_proj(o.reshape(-1, q_size))


class GatedMLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))`` with gate/up fused into one matmul."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size, [config.intermediate_size, config.intermediate_size]
        )
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def tp_size() -> int:
    return get_tp_info().size
