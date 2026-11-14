"""Part III Ch1: attention backends are chosen by name."""

from __future__ import annotations

import pytest

from minisgl.attention import (
    SUPPORTED_ATTENTION_BACKENDS,
    TorchAttnBackend,
    default_attention_backend,
)
from minisgl.config import EngineConfig
from minisgl.engine import Engine
from minisgl.server.args import parse_args


def test_registry_names_and_defaults():
    names = SUPPORTED_ATTENTION_BACKENDS.names()
    assert "torch" in names and "triton" in names
    assert default_attention_backend("cpu") == "torch"
    with pytest.raises(KeyError):
        SUPPORTED_ATTENTION_BACKENDS["no-such-backend"]


def test_cli_flag_reaches_engine_config():
    args = parse_args(["--model", "m", "--attention-backend", "torch", "--disable-cuda-graph"])
    cfg = args.engine_config()
    assert cfg.attention_backend == "torch" and cfg.disable_cuda_graph
    with pytest.raises(SystemExit):
        parse_args(["--model", "m", "--attention-backend", "nope"])


def test_engine_uses_selected_backend(tiny_model):
    engine = Engine(EngineConfig(tiny_model, device="cpu", max_total_tokens=256, max_seq_len=128))
    assert engine.config.attention_backend == "torch"
    assert isinstance(engine.attn_backend, TorchAttnBackend)
