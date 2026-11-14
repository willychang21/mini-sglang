"""Streaming, TP-aware checkpoint loading (Chapter 9, Section 3.3).

Order matters: **shard first, then merge**. Each rank slices its part of
q/k/v (or gate/up) and only then concatenates the local slices into
``qkv_proj`` (``gate_up_proj``). Concatenating the full tensors first and then
splitting evenly would pair the wrong heads/channels on each rank.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..distributed import get_tp_info
from ..utils import div_ceil

_SPLIT_DIM_0 = (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj")
_SPLIT_DIM_1 = (".o_proj", ".down_proj")

# original projection -> (merged projection, slot index)
_MERGE_GROUPS = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
_NUM_SLOTS = {"qkv_proj": 3, "gate_up_proj": 2}
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _shard_tensor(
    key: str, value: torch.Tensor, rank: int, n: int, config: ModelConfig
) -> torch.Tensor:
    if n == 1:
        return value
    if any(sub in key for sub in _SPLIT_DIM_0):
        is_kv = ".k_proj" in key or ".v_proj" in key
        if is_kv and config.num_kv_heads < n:
            # GQA with fewer KV heads than ranks: replicate whole heads.
            head_idx = rank * config.num_kv_heads // n
            d = config.head_dim
            return value[head_idx * d : (head_idx + 1) * d].clone()
        return value.chunk(n, dim=0)[rank].clone()
    if any(sub in key for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[rank].clone()
    if "lm_head" in key or "embed_tokens" in key:
        chunk = div_ceil(config.vocab_size, n)
        part = value[rank * chunk : (rank + 1) * chunk]
        if len(part) < chunk:  # pad the last rank's vocabulary slice
            part = F.pad(part, (0, 0, 0, chunk - len(part)))
        return part.clone()
    return value  # norms are replicated


def _get_merge_info(name: str) -> Tuple[str, int] | None:
    for proj, (merged, slot) in _MERGE_GROUPS.items():
        token = f".{proj}."
        if token in name:
            return name.replace(token, f".{merged}."), slot
    return None


def load_weight(model_path: str, config: ModelConfig) -> Iterator[Tuple[str, torch.Tensor]]:
    tp = get_tp_info()
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    assert files, f"no *.safetensors found in {model_path}"
    merge_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    for file in files:
        with safetensors.safe_open(file, framework="pt") as f:
            for name in f.keys():
                if (m := _LAYER_RE.match(name)) and int(m.group(1)) >= config.num_layers:
                    continue  # truncated (draft) model
                if name.startswith("lm_head") and config.tie_word_embeddings:
                    continue
                name_out = name.removeprefix("language_model.")
                raw = f.get_tensor(name)
                tensor = _shard_tensor(name_out, raw, tp.rank, tp.size, config)
                del raw

                if (info := _get_merge_info(name_out)) is None:
                    yield name_out, tensor
                    continue
                merged_key, slot = info
                slots = merge_buf.setdefault(merged_key, {})
                slots[slot] = tensor
                num_slots = _NUM_SLOTS[merged_key.split(".")[-2]]
                if len(slots) == num_slots:
                    del merge_buf[merged_key]
                    yield merged_key, torch.cat([slots[i] for i in range(num_slots)], dim=0)

    assert not merge_buf, f"incomplete merge groups: {list(merge_buf)}"


@torch.no_grad()
def load_model_weights(model: nn.Module, model_path: str, config: ModelConfig) -> None:
    params = dict(model.named_parameters())
    loaded = set()
    for name, tensor in load_weight(model_path, config):
        if name not in params:
            raise KeyError(f"unexpected checkpoint weight {name}")
        param = params[name]
        if param.shape != tensor.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(param.shape)} vs {tuple(tensor.shape)}")
        param.copy_(tensor.to(param.dtype))
        loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise KeyError(f"weights missing from checkpoint: {sorted(missing)}")
