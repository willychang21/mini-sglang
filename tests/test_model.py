"""Chapter 3: the forward pass matches HuggingFace; Chapter 4: the KV cache does
not change results."""

from __future__ import annotations

import random

import pytest
import torch
from conftest import hf_greedy

from minisgl.config import EngineConfig, ModelConfig
from minisgl.engine import Engine
from minisgl.llm import generate_without_kv_cache
from minisgl.speculative import SequenceRunner


def _engine(path, **kw):
    kw.setdefault("max_total_tokens", 2048)
    kw.setdefault("max_seq_len", 512)
    return Engine(EngineConfig(path, device="cpu", **kw))


@pytest.mark.parametrize("fixture", ["tiny_model", "tiny_model_untied"])
def test_prefill_logits_match_hf(request, fixture, hf_reference):
    path = request.getfixturevalue(fixture)
    engine = _engine(path)
    hf = hf_reference(path)
    prompt = [3, 17, 99, 4, 250, 8, 77]
    logits = SequenceRunner(engine).extend(prompt, all_logits=True)
    with torch.no_grad():
        ref = hf(torch.tensor([prompt])).logits[0]
    torch.testing.assert_close(logits, ref, atol=1e-4, rtol=1e-4)


def test_incremental_extend_matches_full_prefill(tiny_model):
    """K/V written in earlier steps must give the same logits as recomputing."""
    engine = _engine(tiny_model)
    prompt = list(range(10, 40))
    full = SequenceRunner(engine).extend(prompt)
    runner = SequenceRunner(engine, row=1)
    parts = [runner.extend(prompt[:7]), runner.extend(prompt[7:8]), runner.extend(prompt[8:])]
    torch.testing.assert_close(torch.cat(parts), full, atol=1e-5, rtol=1e-5)


def test_generate_without_kv_cache_matches_hf(tiny_model, hf_reference):
    engine = _engine(tiny_model)
    hf = hf_reference(tiny_model)
    random.seed(0)
    for _ in range(3):
        prompt = [random.randrange(500) for _ in range(random.randrange(1, 12))]
        assert generate_without_kv_cache(engine, prompt, 10) == hf_greedy(hf, prompt, 10)


def test_model_config_reads_hf_config(tiny_model):
    cfg = ModelConfig.from_pretrained(tiny_model)
    assert (cfg.num_layers, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (3, 4, 2, 16)
    assert cfg.tie_word_embeddings and cfg.eos_token_ids == (511,)
    assert ModelConfig.from_pretrained(tiny_model, num_layers=1).num_layers == 1
