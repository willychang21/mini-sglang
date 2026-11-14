"""Qwen3 dense model (Chapter 3).

Hidden states use the *packed* layout ``[N, C]``: all tokens computed in this
step, from all requests, concatenated. Request boundaries live in the attention
metadata, not in the tensor shape.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..core import get_global_ctx
from ..layers.embedding import ParallelLMHead, VocabParallelEmbedding
from ..layers.norm import RMSNormFused
from ..layers.rotary import RotaryEmbedding
from .utils import GatedMLP, RopeAttn


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = RopeAttn(config, layer_id, rotary, has_qk_norm=True)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta, device=device
        )
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i, self.rotary_emb) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        return self.norm(x, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        output = self.model(get_global_ctx().batch.input_ids)
        return self.lm_head(output)
