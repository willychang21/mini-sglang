"""Model and engine configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from .distributed import DistributedInfo


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: Tuple[int, ...] = ()
    architecture: str = "Qwen3ForCausalLM"

    @classmethod
    def from_pretrained(cls, model_path: str, num_layers: int | None = None) -> "ModelConfig":
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        eos = cfg.get("eos_token_id")
        gen_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                eos = json.load(f).get("eos_token_id", eos)
        if eos is None:
            eos_ids: Tuple[int, ...] = ()
        elif isinstance(eos, int):
            eos_ids = (eos,)
        else:
            eos_ids = tuple(eos)
        num_heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // num_heads
        total_layers = cfg["num_hidden_layers"]
        if num_layers is not None:
            assert 0 < num_layers <= total_layers
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=num_layers or total_layers,
            num_qo_heads=num_heads,
            num_kv_heads=cfg.get("num_key_value_heads", num_heads),
            head_dim=head_dim,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 10000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            eos_token_ids=eos_ids,
            architecture=cfg.get("architectures", ["Qwen3ForCausalLM"])[0],
        )


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_dtype(device: str) -> torch.dtype:
    # bfloat16 matmuls are slow on most CPUs; float32 also keeps numerics exact
    # enough to compare against HuggingFace token by token.
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float": torch.float32,
}


@dataclass
class EngineConfig:
    model_path: str
    device: str = field(default_factory=_default_device)
    dtype: torch.dtype | None = None
    tp_info: DistributedInfo = field(default_factory=DistributedInfo)
    # --- KV cache (Chapters 4 & 7) ---
    page_size: int = 1
    num_pages: int | None = None  # None: derive from max_total_tokens
    max_total_tokens: int = 32768
    # --- scheduler (Chapter 6) ---
    max_running_req: int = 64
    max_seq_len: int = 4096
    prefill_budget: int = 2048  # max prefill tokens per step (chunked prefill)
    # --- prefix cache (Chapter 8) ---
    cache_type: str = "radix"  # "radix" | "naive"
    # --- GPU execution (Part III Ch1, Part II Ch4) ---
    attention_backend: str | None = None  # None: "torch" on CPU, "triton" on CUDA
    # CUDA Graph for decode: captured batch sizes. None = default buckets up to
    # ``cuda_graph_max_bs``; [] or max_bs == 0 disables. Ignored on CPU.
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    disable_cuda_graph: bool = False
    # --- misc ---
    num_layers: int | None = None  # load only the first N layers (draft models)
    seed: int = 0
    dist_init_addr: str = "tcp://127.0.0.1:29517"

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = _DTYPE_ALIASES.get(self.dtype, None) or getattr(torch, self.dtype)
        if self.dtype is None:
            self.dtype = _default_dtype(self.device)
        assert self.dtype in (torch.float32, torch.bfloat16, torch.float16), self.dtype
        assert self.page_size >= 1
        if self.attention_backend is None:
            from .attention import default_attention_backend

            self.attention_backend = default_attention_backend(self.device)

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig.from_pretrained(self.model_path, self.num_layers)
