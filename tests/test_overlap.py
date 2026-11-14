"""Overlap scheduling: launch batch N+1 before processing batch N.

The scheduler/test_scheduler.py and test_retract.py suites already run under
both loops; these target the hazards specific to overlap.
"""

from __future__ import annotations

import random
import threading
import time

import pytest
from conftest import hf_greedy

from minisgl import SamplingParams
from minisgl.llm import LLM
from minisgl.message import AbortBackendMsg

EOS = 511


def _llm(path, **kw):
    kw.setdefault("max_total_tokens", 1024)
    kw.setdefault("max_seq_len", 256)
    return LLM(path, load_tokenizer_=False, device="cpu", disable_overlap=False, **kw)


def _eos_prompts(hf, n, max_tokens=40, seed=1):
    """Prompts whose greedy continuation hits EOS before max_tokens."""
    rng = random.Random(seed)
    found = []
    for _ in range(400):
        p = [rng.randrange(500) for _ in range(rng.randrange(3, 12))]
        ref = hf_greedy(hf, p, max_tokens, eos={EOS})
        if ref[-1] == EOS and len(ref) < max_tokens:
            found.append((p, ref))
            if len(found) == n:
                return found
    pytest.skip("random model rarely produces EOS")


def _check_idle(llm):
    s = llm.scheduler
    assert s.table_manager.available_size == llm.config.max_running_req
    assert not s.finished_reqs and not s._graveyard and s._ready is None and s._done is None
    s.cache_manager.check_integrity()


def test_scheduling_runs_beside_the_forward(tiny_model):
    """The forward stays on the calling thread; processing / scheduling for the
    next batch runs on the CPU thread at the same time."""
    llm = _llm(tiny_model)
    sched = llm.scheduler
    fwd_threads, cpu_threads = set(), set()
    orig_fwd, orig_proc = llm.engine.forward_batch, sched._process_results

    def fwd(*a, **kw):
        fwd_threads.add(threading.current_thread().name)
        return orig_fwd(*a, **kw)

    def proc(*a, **kw):
        cpu_threads.add(threading.current_thread().name)
        return orig_proc(*a, **kw)

    llm.engine.forward_batch, sched._process_results = fwd, proc
    llm.generate([[1, 2, 3], [4, 5]], SamplingParams(max_tokens=8, ignore_eos=True))
    assert fwd_threads == {threading.current_thread().name}
    assert any(t.startswith("minisgl-sched-cpu") for t in cpu_threads)
    llm.shutdown()


def test_eos_in_flight_token_is_discarded(tiny_model, hf_reference):
    """A request that hits EOS in batch N was already scheduled into N+1 (EOS
    was unknown then); that extra token must be dropped, not sent."""
    hf = hf_reference(tiny_model)
    cases = _eos_prompts(hf, 4)
    llm = _llm(tiny_model)
    outs = llm.generate([p for p, _ in cases], SamplingParams(max_tokens=40))
    for out, (_, ref) in zip(outs, cases):
        assert out.token_ids == ref and out.finish_reason == "stop"
    assert llm.scheduler.stats.discarded_tokens > 0  # the guard was exercised
    _check_idle(llm)


def test_fast_row_reuse_keeps_token_pool_consistent(tiny_model, hf_reference):
    """The token_pool write-ordering hazard.

    With one table row: request A hits EOS in batch N, but was already
    scheduled into N+1. If A's row were freed when N is processed, the next
    admission B would write its (longer) prompt into ``token_pool[row]`` while
    N+1 is still running, and N+1 would then scatter A's discarded token over
    B's prompt. Slowing the worker down makes that interleaving certain;
    deferring the row release until N+1 is processed must keep B intact.
    """
    hf = hf_reference(tiny_model)
    short = _eos_prompts(hf, 3, seed=2)
    rng = random.Random(9)
    long_prompts = [[rng.randrange(500) for _ in range(60)] for _ in range(3)]
    prompts, refs = [], []
    for (p, ref), q in zip(short, long_prompts):
        prompts += [p, q]
        refs += [ref, hf_greedy(hf, q, 8, eos={EOS})]
    for page_size in (1, 4):
        llm = _llm(tiny_model, max_running_req=1, page_size=page_size)
        orig = llm.engine.forward_batch

        def slow(*a, **kw):
            time.sleep(0.02)  # the CPU thread schedules the next batch meanwhile
            return orig(*a, **kw)

        llm.engine.forward_batch = slow
        outs = llm.generate(prompts, [SamplingParams(max_tokens=40), SamplingParams(max_tokens=8)] * 3)
        for out, ref in zip(outs, refs):
            assert out.token_ids == ref
        assert llm.scheduler.stats.discarded_tokens > 0
        _check_idle(llm)


def test_abort_mid_flight(tiny_model, hf_reference):
    """Abort requests while a batch containing them is running."""
    hf = hf_reference(tiny_model)
    rng = random.Random(4)
    prompts = [[rng.randrange(500) for _ in range(rng.randrange(4, 20))] for _ in range(6)]
    llm = _llm(tiny_model)
    sched = llm.scheduler
    victims = {}
    orig = sched._mark_launched

    def launch_then_abort(forward_input):
        inflight = orig(forward_input)
        # once decoding, abort the first two running requests right after
        # their batch was launched: the abort is processed while it runs
        if forward_input.batch.is_decode and not victims:
            for req in forward_input.batch.reqs[:2]:
                victims[req.uid] = None
                llm.io.put(AbortBackendMsg(uid=req.uid))
        return inflight

    sched._mark_launched = launch_then_abort
    outs = llm.generate(prompts, SamplingParams(max_tokens=30, ignore_eos=True))
    uid0 = llm._uid - len(prompts) + 1
    assert victims
    for i, (out, p) in enumerate(zip(outs, prompts)):
        ref = hf_greedy(hf, p, 30)
        if uid0 + i in victims:
            assert out.finish_reason == "abort" and out.token_ids == ref[: len(out.token_ids)]
            assert len(out.token_ids) < 30
        else:
            assert out.token_ids == ref and out.finish_reason == "length"
    assert sched.stats.discarded_tokens >= 2  # their in-flight tokens were dropped
    _check_idle(llm)


def test_sampling_matches_normal_loop(tiny_model):
    """Same seed, same batches -> the overlap loop samples the same tokens."""
    prompts = [[5, 6, 7], [8, 9, 10, 11], [12]]
    sp = SamplingParams(max_tokens=10, temperature=1.0, ignore_eos=True)
    a = [o.token_ids for o in _llm(tiny_model, seed=3).generate(prompts, sp)]
    b = [o.token_ids for o in LLM(tiny_model, load_tokenizer_=False, device="cpu", max_total_tokens=1024,
                                  max_seq_len=256, seed=3, disable_overlap=True).generate(prompts, sp)]
    assert a == b
