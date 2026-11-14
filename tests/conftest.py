"""Shared fixtures: a tiny random Qwen3 checkpoint + HuggingFace reference."""

from __future__ import annotations

import os
import shutil

import pytest
import torch

REAL_MODEL = os.environ.get("MINISGL_TEST_MODEL", "/tmp/models/Qwen3-0.6B")


def make_tiny_qwen3(path: str, vocab_size: int = 512, seed: int = 0, **overrides) -> str:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = dict(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        eos_token_id=vocab_size - 1,
        bos_token_id=vocab_size - 2,
    )
    cfg.update(overrides)
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
    with torch.no_grad():  # random init is too tame; make logits peaky & diverse
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.2)
    model.save_pretrained(path, safe_serialization=True)
    gen = os.path.join(path, "generation_config.json")
    if os.path.exists(gen):
        os.remove(gen)
    return path


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> str:
    return make_tiny_qwen3(str(tmp_path_factory.mktemp("tiny-qwen3")))


@pytest.fixture(scope="session")
def tiny_model_untied(tmp_path_factory) -> str:
    return make_tiny_qwen3(
        str(tmp_path_factory.mktemp("tiny-qwen3-untied")), tie_word_embeddings=False, num_key_value_heads=1
    )


@pytest.fixture(scope="session")
def tiny_model_with_tokenizer(tmp_path_factory) -> str:
    """Tiny weights + the real Qwen3 tokenizer (for detokenizer / server tests)."""
    if not os.path.exists(os.path.join(REAL_MODEL, "tokenizer.json")):
        pytest.skip("real Qwen3 tokenizer not available")
    path = str(tmp_path_factory.mktemp("tiny-qwen3-tok"))
    make_tiny_qwen3(path, vocab_size=151936, eos_token_id=151645, bos_token_id=151643)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(os.path.join(REAL_MODEL, f), path)
    return path


@pytest.fixture(scope="session")
def hf_reference():
    """Returns a function(model_path) -> HF model, cached per path."""
    from transformers import AutoModelForCausalLM

    cache = {}

    def load(path: str):
        if path not in cache:
            cache[path] = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        return cache[path]

    return load


def hf_greedy(model, prompt, max_new_tokens, eos=None):
    """Reference greedy generation with HuggingFace (no sampling)."""
    ids = torch.tensor([prompt])
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids).logits[0, -1]
            tok = int(logits.argmax())
            out.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
            if eos is not None and tok in eos:
                break
    return out
