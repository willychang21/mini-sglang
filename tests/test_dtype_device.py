"""GPU readiness that can be checked on CPU: device resolution, dtype parsing,
and half-precision execution of both attention backends."""

from __future__ import annotations

import pytest
import torch

from minisgl.config import EngineConfig
from minisgl.engine import Engine
from minisgl.engine.engine import resolve_device
from minisgl.speculative import SequenceRunner


def test_resolve_device():
    assert resolve_device("cpu", 0, 1) == torch.device("cpu")
    assert resolve_device("cpu", 1, 2) == torch.device("cpu")
    # TP: rank r always owns cuda:r, whatever index was passed
    assert resolve_device("cuda", 1, 2) == torch.device("cuda", 1)
    assert resolve_device("cuda:0", 3, 4) == torch.device("cuda", 3)
    assert resolve_device("cuda:2", 0, 1) == torch.device("cuda", 2)


@pytest.mark.parametrize("name,dtype", [("bf16", torch.bfloat16), ("bfloat16", torch.bfloat16),
                                         ("fp16", torch.float16), ("float32", torch.float32)])
def test_dtype_aliases(tiny_model, name, dtype):
    assert EngineConfig(tiny_model, device="cpu", dtype=name).dtype == dtype


def test_default_dtype_cpu(tiny_model):
    assert EngineConfig(tiny_model, device="cpu").dtype == torch.float32


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_half_precision_close_to_fp32(tiny_model, dtype, backend):
    if backend == "triton":
        pytest.importorskip("triton")
    prompt = [3, 17, 99, 4, 250, 8, 77]
    kw = dict(device="cpu", max_total_tokens=256, max_seq_len=64)
    ref = SequenceRunner(Engine(EngineConfig(tiny_model, **kw))).extend(prompt)
    engine = Engine(EngineConfig(tiny_model, dtype=dtype, attention_backend=backend, **kw))
    assert engine.kv_cache.dtype == getattr(torch, dtype)
    assert all(p.dtype == getattr(torch, dtype) for p in engine.model.parameters())
    out = SequenceRunner(engine).extend(prompt).float()
    assert (out - ref).abs().max() < 0.05 * ref.abs().max()
    assert torch.equal(out.argmax(-1), ref.argmax(-1))
