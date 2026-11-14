"""GPU-only checks (skipped without CUDA): the real Triton kernels, CUDA Graph
capture + replay, and bf16 serving. Run them with notebooks/gpu_benchmarks.ipynb.

Correctness bar: fp32 greedy output equals HuggingFace; graph replay equals
eager execution.
"""

from __future__ import annotations

import os
import random

import pytest
import torch
from conftest import REAL_MODEL, hf_greedy

from minisgl import SamplingParams
from minisgl.llm import LLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _prompts(n=6, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(500) for _ in range(rng.randrange(1, 30))] for _ in range(n)]


def _llm(path, **kw):
    kw.setdefault("max_total_tokens", 2048)
    kw.setdefault("max_seq_len", 512)
    kw.setdefault("max_running_req", 16)
    return LLM(path, load_tokenizer_=False, device="cuda", **kw)


@pytest.mark.parametrize("page_size", [1, 16])
def test_cuda_graph_fp32_matches_hf(tiny_model, hf_reference, page_size):
    pytest.importorskip("triton")
    hf = hf_reference(tiny_model)
    prompts = _prompts()
    lengths = [3, 17, 9, 25, 1, 12]  # batch shrinks 6 -> 1: replays pad to 8/4/2/1
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    llm = _llm(tiny_model, dtype="float32", attention_backend="triton", page_size=page_size)
    assert llm.engine.graph_runner is not None and llm.engine.graph_runner.graphs
    outs = llm.generate(prompts, sps)
    for out, prompt, n in zip(outs, prompts, lengths):
        assert out.token_ids == hf_greedy(hf, prompt, n)
    llm.shutdown()


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_eager_fp32_matches_hf(tiny_model, hf_reference, backend):
    hf = hf_reference(tiny_model)
    prompts = _prompts(seed=1)
    llm = _llm(tiny_model, dtype="float32", attention_backend=backend, disable_cuda_graph=True)
    outs = llm.generate(prompts, SamplingParams(max_tokens=10, ignore_eos=True))
    for out, prompt in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, prompt, 10)


def test_cuda_graph_replay_equals_eager_bf16(tiny_model):
    prompts = _prompts(seed=2)
    sp = SamplingParams(max_tokens=16, ignore_eos=True)
    graph = _llm(tiny_model, dtype="bfloat16", attention_backend="triton")
    eager = _llm(tiny_model, dtype="bfloat16", attention_backend="triton", disable_cuda_graph=True)
    a = [o.token_ids for o in graph.generate(prompts, sp)]
    b = [o.token_ids for o in eager.generate(prompts, sp)]
    assert a == b


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_MODEL, "config.json")), reason="no checkpoint")
def test_real_model_triton_cuda_graph_fp32(hf_reference):
    llm = LLM(REAL_MODEL, device="cuda", dtype="float32", attention_backend="triton",
              max_total_tokens=4096, max_seq_len=2048, max_running_req=8)
    prompts = ["The capital of France is", [{"role": "user", "content": "What is KV cache?"}]]
    outs = llm.generate(prompts, SamplingParams(max_tokens=16))
    hf = hf_reference(REAL_MODEL)
    for prompt, out in zip(prompts, outs):
        assert out.token_ids == hf_greedy(hf, llm.tokenize(prompt), 16, eos={151645, 151643})
