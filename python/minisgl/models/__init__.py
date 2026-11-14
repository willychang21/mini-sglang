from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .qwen3 import Qwen3ForCausalLM
from .weight import load_model_weights

_MODELS = {"Qwen3ForCausalLM": Qwen3ForCausalLM}


def create_model(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> nn.Module:
    if config.architecture not in _MODELS:
        raise ValueError(f"unsupported architecture {config.architecture}; supported: {list(_MODELS)}")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = _MODELS[config.architecture](config, device)
    finally:
        torch.set_default_dtype(old_dtype)
    return model.eval()


__all__ = ["create_model", "load_model_weights", "Qwen3ForCausalLM"]
