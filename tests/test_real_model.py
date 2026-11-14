"""End-to-end check with the real Qwen3-0.6B checkpoint (skipped if absent)."""

from __future__ import annotations

import os

import pytest
from conftest import REAL_MODEL, hf_greedy

from minisgl import SamplingParams
from minisgl.llm import LLM

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(os.path.join(REAL_MODEL, "config.json")), reason="no checkpoint"),
]


def test_qwen3_greedy_matches_huggingface(hf_reference):
    llm = LLM(REAL_MODEL, device="cpu", max_total_tokens=4096, max_seq_len=2048, page_size=4)
    prompts = [
        [{"role": "user", "content": "What is KV cache? Answer in one sentence."}],
        "The capital of France is",
        "用一句话介绍北京。",
    ]
    outs = llm.generate(prompts, SamplingParams(max_tokens=24))
    hf = hf_reference(REAL_MODEL)
    for prompt, out in zip(prompts, outs):
        ref = hf_greedy(hf, llm.tokenize(prompt), 24, eos={151645, 151643})
        assert out.token_ids == ref
    assert outs[1].text.startswith(" Paris")
