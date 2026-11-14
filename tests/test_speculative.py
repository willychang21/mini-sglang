"""Chapter 10: speculative decoding."""

from __future__ import annotations

import random

import pytest
import torch
from conftest import hf_greedy, make_tiny_qwen3

from minisgl import SamplingParams
from minisgl.config import EngineConfig
from minisgl.engine import Engine
from minisgl.speculative import SpeculativeDecoder, greedy_commit, rejection_commit


def _engine(path, **kw):
    return Engine(EngineConfig(path, device="cpu", max_total_tokens=1024, max_seq_len=512, **kw))


def test_greedy_commit_alignment():
    v = torch.zeros(4, 5)
    for row, tok in enumerate([1, 2, 3, 4]):
        v[row, tok] = 1.0
    assert greedy_commit([1, 2, 3], v, remaining=10) == [1, 2, 3, 4]  # all accepted + bonus
    assert greedy_commit([1, 1, 1], v, remaining=10) == [1, 2]  # stop at first rejection
    assert greedy_commit([1, 2, 3], v, remaining=2) == [1, 2]  # output budget
    assert greedy_commit([1, 2, 3], v, remaining=10, eos_ids={2}) == [1, 2]  # EOS
    assert greedy_commit([1, 2, 3], v, remaining=0) == []


def test_rejection_sampling_preserves_target_distribution():
    """Section 5.1's example: p=[0.4,0.6], q=[0.8,0.2] -> output ~ p."""
    p = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    q = torch.tensor([[0.8, 0.2]])
    gen = torch.Generator().manual_seed(0)
    counts = torch.zeros(2)
    n = 20000
    for _ in range(n):
        y = int(torch.multinomial(q[0], 1, generator=gen))
        first = rejection_commit([y], q, p, remaining=1, generator=gen)[0]
        counts[first] += 1
    freq = counts / n
    assert abs(freq[0] - 0.4) < 0.015 and abs(freq[1] - 0.6) < 0.015


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_speculation_equals_target_greedy(tiny_model, hf_reference, tmp_path, k):
    hf = hf_reference(tiny_model)
    target = _engine(tiny_model)
    unrelated = make_tiny_qwen3(str(tmp_path / "draft"), seed=123, num_hidden_layers=1)
    for draft in (_engine(tiny_model), _engine(unrelated), _engine(tiny_model, num_layers=1)):
        spec = SpeculativeDecoder(target, draft, num_draft_tokens=k)
        rng = random.Random(k)
        for _ in range(3):
            prompt = [rng.randrange(500) for _ in range(rng.randrange(1, 15))]
            n = rng.randrange(1, 30)
            out, stats = spec.generate(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
            assert out == hf_greedy(hf, prompt, n)
            assert stats.target_forwards <= n


def test_self_draft_accepts_everything(tiny_model):
    target = _engine(tiny_model)
    spec = SpeculativeDecoder(target, _engine(tiny_model), num_draft_tokens=4)
    out, stats = spec.generate([1, 2, 3], SamplingParams(max_tokens=21, ignore_eos=True))
    assert len(out) == 21
    assert stats.acceptance_rate == 1.0
    assert stats.target_forwards == 1 + 4  # prefill + 4 rounds of 5 tokens


def test_sampling_speculation_runs(tiny_model):
    target = _engine(tiny_model)
    spec = SpeculativeDecoder(target, _engine(tiny_model, num_layers=1), num_draft_tokens=3)
    sp = SamplingParams(max_tokens=25, temperature=0.8, top_k=50, ignore_eos=True)
    out, stats = spec.generate([5, 6, 7], sp)
    assert len(out) == 25 and stats.rounds > 0


def test_ngram_drafter_is_lossless_and_accepts_repeats(tiny_model, hf_reference):
    from minisgl.speculative import NgramDrafter

    hf = hf_reference(tiny_model)
    target = _engine(tiny_model)
    spec = SpeculativeDecoder(target, NgramDrafter(vocab_size=512), num_draft_tokens=5)
    rng = random.Random(0)
    for _ in range(4):
        prompt = [rng.randrange(500) for _ in range(rng.randrange(2, 20))]
        n = rng.randrange(5, 40)
        out, _ = spec.generate(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
        assert out == hf_greedy(hf, prompt, n)
    # greedy output of a random model tends to fall into cycles -> lookups hit
    out, stats = spec.generate([1, 2, 3], SamplingParams(max_tokens=120, ignore_eos=True))
    assert out == hf_greedy(hf, [1, 2, 3], 120)
    assert stats.target_forwards < 120


def test_ngram_drafter_proposals():
    from minisgl.speculative import NgramDrafter

    d = NgramDrafter(vocab_size=100, max_ngram=2)
    gen = torch.Generator()
    sp = SamplingParams()
    assert d.propose([5, 6, 7, 8, 5, 6], 3, sp, (), gen)[0] == [7, 8, 5]
    assert d.propose([1, 2, 3], 3, sp, (), gen)[0] == []
    assert d.propose([5, 6, 9, 5, 6], 3, sp, {9}, gen)[0] == [9]
    _, q = d.propose([5, 6, 7, 5, 6], 2, SamplingParams(temperature=1.0), (), gen)
    assert len(q) == 2 and q[0][7] == 1.0
