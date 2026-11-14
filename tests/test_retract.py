"""Optimistic admission + decode retraction (preemption).

Under a KV budget far too small for every request's full ``max_tokens``,
optimistic admission over-commits the pool and must retract running requests.
Outputs must still equal HuggingFace greedy token for token, and no page or
table row may leak.
"""

from __future__ import annotations

import random

import pytest
from conftest import hf_greedy

from minisgl import SamplingParams
from minisgl.llm import LLM
from minisgl.scheduler import SchedulerConfig
from minisgl.scheduler.retract import NewTokenRatioTracker


def _prompts(n=8, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(500) for _ in range(rng.randrange(5, 20))] for _ in range(n)]


LOOP = "normal"


@pytest.fixture(autouse=True, params=["normal", "overlap"])
def _loop(request):
    """Every test runs under both scheduler loops."""
    global LOOP
    LOOP = request.param
    yield


def _llm(path, **kw):
    kw.setdefault("disable_overlap", LOOP == "normal")
    kw.setdefault("max_total_tokens", 160)
    kw.setdefault("max_seq_len", 128)
    return LLM(path, load_tokenizer_=False, device="cpu", **kw)


def _check_idle(llm: LLM) -> None:
    sched = llm.scheduler
    assert sched.table_manager.available_size == llm.config.max_running_req
    assert not sched.prompt_lens
    sched.cache_manager.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cache_type", ["radix", "naive"])
@pytest.mark.parametrize("prefill_budget", [2048, 8])
def test_optimistic_retracts_and_matches_hf(tiny_model, hf_reference, page_size, cache_type, prefill_budget):
    hf = hf_reference(tiny_model)
    prompts = _prompts()
    lengths = [random.Random(i).randrange(20, 45) for i in range(len(prompts))]
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    llm = _llm(
        tiny_model,
        page_size=page_size,
        cache_type=cache_type,
        prefill_budget=prefill_budget,
        admission="optimistic",
        new_token_ratio_decay_steps=20,  # decay fast so retraction happens repeatedly
    )
    for _ in range(2):
        outs = llm.generate(prompts, sps)
        for out, prompt, n in zip(outs, prompts, lengths):
            assert out.token_ids == hf_greedy(hf, prompt, n)
            assert out.finish_reason == "length"
            assert out.prompt_tokens == len(prompt)  # not prompt + generated after a resume
        _check_idle(llm)
    assert llm.scheduler.stats.retracted_reqs > 0


def test_conservative_never_retracts_and_optimistic_runs_wider(tiny_model):
    prompts = _prompts()
    sp = SamplingParams(max_tokens=40, ignore_eos=True)
    peaks, outs = {}, {}
    for admission in ("conservative", "optimistic"):
        llm = _llm(tiny_model, admission=admission)
        outs[admission] = [o.token_ids for o in llm.generate(prompts, sp)]
        peaks[admission] = max(llm.scheduler.stats.batch_sizes)
        if admission == "conservative":
            assert llm.scheduler.stats.retracted_reqs == 0
        _check_idle(llm)
    assert outs["conservative"] == outs["optimistic"]
    assert peaks["optimistic"] > peaks["conservative"]


def test_resume_reuses_radix_cache(tiny_model, hf_reference):
    """A retracted request's KV is handed to the radix cache (minus the pages
    freed as headroom for the survivors), so its re-prefill matches a prefix
    instead of recomputing the whole prompt + output."""
    hf = hf_reference(tiny_model)
    rng = random.Random(0)
    prompts = [[rng.randrange(500) for _ in range(rng.randrange(40, 80))] for _ in range(12)]
    llm = _llm(tiny_model, admission="optimistic", max_total_tokens=700, max_seq_len=256)
    sched = llm.scheduler
    resumed = []  # (cached_len, extend_len) of re-admitted requests
    orig = sched.prefill_manager.schedule_next_batch

    def spy(*a, **kw):
        retracted = {p.uid for p in sched.prefill_manager.pending_list if p.retracted}
        batch = orig(*a, **kw)
        if batch is not None:
            resumed.extend((r.cached_len, r.extend_len) for r in batch.reqs if r.uid in retracted)
        return batch

    sched.prefill_manager.schedule_next_batch = spy
    outs = llm.generate(prompts, SamplingParams(max_tokens=60, ignore_eos=True))
    for out, p in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, p, 60)
    assert sched.stats.retracted_reqs > 0 and resumed
    assert all(cached > 0 for cached, _ in resumed)
    _check_idle(llm)


def test_abort_while_retracted(tiny_model, hf_reference):
    hf = hf_reference(tiny_model)
    llm = _llm(tiny_model, admission="optimistic")
    sched = llm.scheduler
    prompts = _prompts()
    sp = SamplingParams(max_tokens=40, ignore_eos=True)
    aborted = []
    orig = sched._retract_one

    def retract_then_abort(req, short):
        pending = orig(req, short)
        if not aborted:  # abort the first retracted request while it waits
            aborted.append(req.uid)
            from minisgl.message import AbortBackendMsg

            llm.io.put(AbortBackendMsg(uid=req.uid))
        return pending

    sched._retract_one = retract_then_abort
    uid0 = llm._uid + 1
    outs = llm.generate(prompts, sp)
    assert aborted
    for i, (out, p) in enumerate(zip(outs, prompts)):
        if uid0 + i == aborted[0]:
            assert out.finish_reason == "abort"
            assert out.token_ids == hf_greedy(hf, p, len(out.token_ids))
        else:
            assert out.token_ids == hf_greedy(hf, p, 40)
    _check_idle(llm)


def test_eos_mix_under_optimistic(tiny_model, hf_reference):
    """Without ignore_eos some requests stop early: the case optimism is for."""
    hf = hf_reference(tiny_model)
    prompts = _prompts(12, seed=5)
    llm = _llm(tiny_model, admission="optimistic", max_total_tokens=200)
    outs = llm.generate(prompts, SamplingParams(max_tokens=40))
    for out, p in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, p, 40, eos={511})
    _check_idle(llm)


def test_new_token_ratio_tracker():
    cfg = SchedulerConfig(admission="optimistic", init_new_token_ratio=0.7,
                          min_new_token_ratio_factor=0.1, new_token_ratio_decay_steps=10)
    t = NewTokenRatioTracker.from_config(cfg)
    assert t.current == pytest.approx(0.7) and t.min == pytest.approx(0.07)
    for _ in range(100):
        t.decay_step()
    assert t.current == pytest.approx(0.07)  # clamps at min
    t.reset()
    assert t.current == pytest.approx(0.7)
    cons = NewTokenRatioTracker.from_config(SchedulerConfig())
    cons.decay_step()
    assert cons.current == 1.0  # conservative admission never decays
