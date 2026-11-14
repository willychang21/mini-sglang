"""Chapters 6-8: continuous batching, paged KV, prefix caching, chunked prefill,
cancellation. Every configuration must reproduce HuggingFace greedy output."""

from __future__ import annotations

import random

import pytest
from conftest import hf_greedy

from minisgl import SamplingParams
from minisgl.llm import LLM


def _prompts(n=8, seed=0, shared=5):
    rng = random.Random(seed)
    base = [rng.randrange(500) for _ in range(shared)]
    return [base + [rng.randrange(500) for _ in range(rng.randrange(1, 30))] for _ in range(n)]


LOOP = "normal"


@pytest.fixture(autouse=True, params=["normal", "overlap"])
def _loop(request):
    """Every test runs under both scheduler loops."""
    global LOOP
    LOOP = request.param
    yield


def _llm(path, **kw):
    kw.setdefault("disable_overlap", LOOP == "normal")
    kw.setdefault("max_total_tokens", 1024)
    kw.setdefault("max_seq_len", 256)
    return LLM(path, load_tokenizer_=False, device="cpu", **kw)


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cache_type", ["radix", "naive"])
@pytest.mark.parametrize("max_running_req,prefill_budget,kv_tokens", [(16, 2048, 1024), (3, 16, 300)])
def test_continuous_batching_matches_hf(
    tiny_model, hf_reference, page_size, cache_type, max_running_req, prefill_budget, kv_tokens
):
    hf = hf_reference(tiny_model)
    prompts = _prompts()
    llm = _llm(
        tiny_model,
        page_size=page_size,
        cache_type=cache_type,
        max_running_req=max_running_req,
        prefill_budget=prefill_budget,
        max_total_tokens=kv_tokens,
    )
    lengths = [random.Random(i).randrange(1, 25) for i in range(len(prompts))]
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    for _ in range(2):  # second round exercises prefix-cache hits
        outs = llm.generate(prompts, sps)
        for out, prompt, n in zip(outs, prompts, lengths):
            assert out.token_ids == hf_greedy(hf, prompt, n)
            assert out.finish_reason == "length"
        llm.scheduler.cache_manager.check_integrity()
    stats = llm.scheduler.prefill_manager.stats
    if cache_type == "radix":
        assert stats.cached_tokens > 0
    else:
        assert stats.cached_tokens == 0
    if max_running_req == 3:  # requests had to wait for free slots
        assert max(llm.scheduler.stats.batch_sizes) <= 3


def test_prefix_cache_skips_shared_prefix(tiny_model):
    llm = _llm(tiny_model)
    system = list(range(100, 160))
    llm.generate([system + [1, 2, 3]], SamplingParams(max_tokens=4, ignore_eos=True))
    before = llm.scheduler.stats.prefill_tokens
    llm.generate([system + [7, 8]], SamplingParams(max_tokens=4, ignore_eos=True))
    computed = llm.scheduler.stats.prefill_tokens - before
    assert computed == 2  # only the unmatched suffix ran through the model


def test_full_prompt_hit_still_runs_last_token(tiny_model, hf_reference):
    llm = _llm(tiny_model)
    p = list(range(50, 80))
    a = llm.generate([p], SamplingParams(max_tokens=5, ignore_eos=True))[0]
    before = llm.scheduler.stats.prefill_tokens
    b = llm.generate([p], SamplingParams(max_tokens=5, ignore_eos=True))[0]
    assert llm.scheduler.stats.prefill_tokens - before == 1
    assert a.token_ids == b.token_ids == hf_greedy(hf_reference(tiny_model), p, 5)


def test_eviction_under_memory_pressure(tiny_model, hf_reference):
    hf = hf_reference(tiny_model)
    llm = _llm(tiny_model, max_total_tokens=120, max_running_req=4)
    rng = random.Random(3)
    for _ in range(6):  # each round leaves cached prefixes that must be evicted
        prompts = [[rng.randrange(500) for _ in range(rng.randrange(5, 30))] for _ in range(3)]
        outs = llm.generate(prompts, SamplingParams(max_tokens=10, ignore_eos=True))
        for o, p in zip(outs, prompts):
            assert o.token_ids == hf_greedy(hf, p, 10)
        llm.scheduler.cache_manager.check_integrity()


def test_eos_stops_generation(tiny_model, hf_reference):
    hf = hf_reference(tiny_model)
    llm = _llm(tiny_model)
    rng = random.Random(1)
    found = False
    for _ in range(40):
        p = [rng.randrange(500) for _ in range(5)]
        ref = hf_greedy(hf, p, 40)
        if 511 in ref:
            ref = ref[: ref.index(511) + 1]
            out = llm.generate([p], SamplingParams(max_tokens=40))[0]
            assert out.token_ids == ref and out.finish_reason == "stop"
            found = True
            break
    if not found:
        pytest.skip("random model never produced EOS")


def test_abort_frees_resources(tiny_model):
    llm = _llm(tiny_model)
    prompts = _prompts(4)
    outs = llm.generate(
        prompts, SamplingParams(max_tokens=50, ignore_eos=True), abort_after={1: 3, 2: 1}
    )
    assert outs[0].finish_reason == outs[3].finish_reason == "length"
    assert outs[1].finish_reason == outs[2].finish_reason == "abort"
    assert len(outs[1].token_ids) < 50
    assert llm.scheduler.table_manager.available_size == llm.config.max_running_req
    llm.scheduler.cache_manager.check_integrity()


def test_too_long_prompt_returns_error(tiny_model):
    llm = _llm(tiny_model, max_seq_len=32)
    out = llm.generate([list(range(40))], SamplingParams(max_tokens=4))[0]
    assert out.finish_reason == "error" and out.error


def test_max_tokens_clipped_to_max_seq_len(tiny_model):
    llm = _llm(tiny_model, max_seq_len=32)
    out = llm.generate([list(range(20))], SamplingParams(max_tokens=100, ignore_eos=True))[0]
    assert len(out.token_ids) == 12 and out.finish_reason == "length"


def test_sampling_is_seeded_and_respects_top_k(tiny_model):
    sp = SamplingParams(max_tokens=12, temperature=1.0, top_k=1, ignore_eos=True)
    greedy = SamplingParams(max_tokens=12, ignore_eos=True)
    llm = _llm(tiny_model)
    prompts = _prompts(3)
    assert [o.token_ids for o in llm.generate(prompts, sp)] == [
        o.token_ids for o in llm.generate(prompts, greedy)
    ]
    hot = SamplingParams(max_tokens=12, temperature=1.5, ignore_eos=True)
    a = [o.token_ids for o in _llm(tiny_model, seed=7).generate(prompts, hot)]
    b = [o.token_ids for o in _llm(tiny_model, seed=7).generate(prompts, hot)]
    assert a == b  # same seed -> same samples
    mixed = llm.generate(prompts, [hot, greedy, hot])
    ref = llm.generate([prompts[1]], greedy)[0]
    assert mixed[1].token_ids == ref.token_ids  # greedy rows stay greedy in mixed batches


def test_out_of_vocab_ids_are_rejected(tiny_model):
    """ISSUES.md #2: an id >= vocab_size used to raise IndexError in the loop."""
    llm = _llm(tiny_model)
    bad = llm.generate([[1, 600, 2], [-1, 3], [5, 6, 7]], SamplingParams(max_tokens=3, ignore_eos=True))
    assert [o.finish_reason for o in bad] == ["error", "error", "length"]
    assert "token ids" in bad[0].error and len(bad[2].token_ids) == 3


def test_hopeless_request_does_not_touch_radix_tree(tiny_model):
    """ISSUES.md #4 (scheduler side): no match_prefix while even a fully cached
    prompt could not fit, so waiting requests do not split nodes / bump LRU."""
    llm = _llm(tiny_model, max_total_tokens=120, max_running_req=4)
    sched = llm.scheduler
    calls = []
    orig = sched.cache_manager.match_req
    sched.cache_manager.match_req = lambda ids: calls.append(len(ids)) or orig(ids)
    # three requests that each reserve ~100 tokens: only one fits at a time
    outs = llm.generate([[i] * 4 for i in range(1, 4)], SamplingParams(max_tokens=90, ignore_eos=True))
    assert all(len(o.token_ids) == 90 for o in outs)
    assert len(calls) == 3  # one successful match each, no matching while waiting
