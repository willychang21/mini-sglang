"""Speculative decoding inside the continuous-batching scheduler.

Greedy speculation is lossless, so every configuration must reproduce
HuggingFace greedy output token for token, and leave no page or table row
behind (``check_integrity``). See docs/speculative-in-scheduler.md.
"""

from __future__ import annotations

import random

import pytest
from conftest import hf_greedy, make_tiny_qwen3

from minisgl import SamplingParams
from minisgl.llm import LLM

LOOP = "normal"


@pytest.fixture(autouse=True, params=["normal", "overlap"])
def _loop(request):
    """Every test runs under both loop settings; with speculation on, the
    overlap request must be overridden by the normal loop (enforced)."""
    global LOOP
    LOOP = request.param
    yield


# name -> scheduler kwargs (draft models: the target truncated to 1 layer,
# the full target = every proposal accepted)
ALGOS = {
    "ngram": dict(speculative_algorithm="ngram"),
    "draft-1layer": dict(speculative_algorithm="draft_model", speculative_draft_num_layers=1),
    "draft-self": dict(speculative_algorithm="draft_model"),
}


def _llm(path, algo="ngram", **kw):
    kw.setdefault("disable_overlap", LOOP == "normal")
    kw.setdefault("max_total_tokens", 1024)
    kw.setdefault("max_seq_len", 256)
    kw.setdefault("speculative_num_draft_tokens", 4)
    llm = LLM(path, load_tokenizer_=False, device="cpu", **ALGOS[algo], **kw)
    assert not llm.scheduler.overlap  # speculation forces the normal loop
    return llm


def _prompts(n=8, seed=0, shared=5):
    rng = random.Random(seed)
    base = [rng.randrange(500) for _ in range(shared)]
    out = []
    for _ in range(n):
        p = base + [rng.randrange(500) for _ in range(rng.randrange(1, 20))]
        if rng.random() < 0.5:  # copy-heavy: the prompt repeats itself -> n-gram hits
            p = p + p[: rng.randrange(3, 10)]
        out.append(p)
    return out


def _check_idle(llm: LLM) -> None:
    sched = llm.scheduler
    assert sched.table_manager.available_size == llm.config.max_running_req
    assert not sched.prompt_lens and not sched.decode_manager.running_reqs
    sched.cache_manager.check_integrity()


@pytest.mark.parametrize("algo", list(ALGOS))
@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cache_type", ["radix", "naive"])
def test_spec_batch_matches_hf(tiny_model, hf_reference, algo, page_size, cache_type):
    hf = hf_reference(tiny_model)
    prompts = _prompts()
    # max_tokens spread so requests finish in different rounds, many mid-round
    lengths = [random.Random(i).randrange(1, 40) for i in range(len(prompts))]
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    llm = _llm(tiny_model, algo, page_size=page_size, cache_type=cache_type)
    for _ in range(2):  # second round: prefix-cache hits (draft K/V recomputed)
        outs = llm.generate(prompts, sps)
        for out, prompt, n in zip(outs, prompts, lengths):
            assert out.token_ids == hf_greedy(hf, prompt, n)
            assert out.finish_reason == "length"
        _check_idle(llm)
    st = llm.scheduler.spec.stats
    assert st.verify_steps > 0 and st.committed > 0
    assert st.committed <= st.accepted + st.req_rounds  # accepted + at most one correction/bonus
    if algo == "ngram":
        assert st.accepted > 0  # copy-heavy prompts give the lookup something to find
    if algo == "draft-self":
        assert st.acceptance_rate == 1.0
        assert st.accept_length > 2.0  # k=4: most rounds commit 5 tokens


@pytest.mark.parametrize("k", [1, 2, 7])
def test_num_draft_tokens(tiny_model, hf_reference, k):
    hf = hf_reference(tiny_model)
    prompts = _prompts(4, seed=k)
    llm = _llm(tiny_model, "draft-self", speculative_num_draft_tokens=k)
    outs = llm.generate(prompts, SamplingParams(max_tokens=23, ignore_eos=True))
    for out, prompt in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, prompt, 23)
    st = llm.scheduler.spec.stats
    # first token from prefill, then rounds of up to k+1 (budget-clipped)
    assert st.verify_steps <= -(-22 // (k + 1)) + 1
    _check_idle(llm)


def test_unrelated_draft_model_is_lossless(tiny_model, hf_reference, tmp_path):
    hf = hf_reference(tiny_model)
    draft = make_tiny_qwen3(str(tmp_path / "draft"), seed=123, num_hidden_layers=1)
    prompts = _prompts(5, seed=3)
    llm = _llm(tiny_model, "draft-1layer", speculative_draft_model=draft, page_size=4)
    outs = llm.generate(prompts, SamplingParams(max_tokens=30, ignore_eos=True))
    for out, prompt in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, prompt, 30)
    _check_idle(llm)


@pytest.mark.parametrize("algo", ["ngram", "draft-self"])
@pytest.mark.parametrize("page_size", [1, 4])
def test_tight_kv_retracts_and_matches_hf(tiny_model, hf_reference, algo, page_size):
    """Optimistic admission over-commits a tiny pool: rounds get trimmed to the
    free pages and requests get retracted between rounds."""
    hf = hf_reference(tiny_model)
    prompts = _prompts(8, seed=1)
    lengths = [random.Random(i).randrange(20, 45) for i in range(len(prompts))]
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    llm = _llm(
        tiny_model,
        algo,
        page_size=page_size,
        max_total_tokens=200,
        max_seq_len=128,
        admission="optimistic",
        new_token_ratio_decay_steps=20,
    )
    for _ in range(2):
        outs = llm.generate(prompts, sps)
        for out, prompt, n in zip(outs, prompts, lengths):
            assert out.token_ids == hf_greedy(hf, prompt, n)
            assert out.prompt_tokens == len(prompt)
        _check_idle(llm)
    assert llm.scheduler.stats.retracted_reqs > 0


def test_conservative_tight_kv_never_retracts(tiny_model, hf_reference):
    """Admission reserves k extra tokens per request, so a conservative
    scheduler never runs short during a round."""
    hf = hf_reference(tiny_model)
    prompts = _prompts(6, seed=2)
    llm = _llm(tiny_model, "draft-self", max_total_tokens=160, max_seq_len=128, max_running_req=8)
    outs = llm.generate(prompts, SamplingParams(max_tokens=40, ignore_eos=True))
    for out, prompt in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, prompt, 40)
    assert llm.scheduler.stats.retracted_reqs == 0
    assert llm.scheduler.spec.stats.trimmed == 0
    _check_idle(llm)


@pytest.mark.parametrize("algo", ["ngram", "draft-1layer"])
def test_chunked_prefill_with_spec(tiny_model, hf_reference, algo):
    hf = hf_reference(tiny_model)
    prompts = _prompts(6, seed=4)
    llm = _llm(tiny_model, algo, prefill_budget=7, max_running_req=3, page_size=4)
    outs = llm.generate(prompts, SamplingParams(max_tokens=25, ignore_eos=True))
    for out, prompt in zip(outs, prompts):
        assert out.token_ids == hf_greedy(hf, prompt, 25)
    _check_idle(llm)


@pytest.mark.parametrize("algo", list(ALGOS))
def test_eos_mid_round(tiny_model, hf_reference, algo):
    """A request whose EOS lands inside a round stops exactly there; others in
    the same batch keep going."""
    hf = hf_reference(tiny_model)
    eos = 511
    rng = random.Random(1)
    with_eos = []
    for _ in range(200):
        p = [rng.randrange(500) for _ in range(5)]
        ref = hf_greedy(hf, p, 40)
        if eos in ref and ref.index(eos) > 1:
            with_eos.append((p, ref[: ref.index(eos) + 1]))
        if len(with_eos) == 2:
            break
    if not with_eos:
        pytest.skip("random model never produced EOS")
    others = _prompts(3, seed=9)
    llm = _llm(tiny_model, algo)
    prompts = [p for p, _ in with_eos] + others
    outs = llm.generate(prompts, SamplingParams(max_tokens=40))
    for out, (_, ref) in zip(outs, with_eos):
        assert out.token_ids == ref and out.finish_reason == "stop"
    for out, p in zip(outs[len(with_eos) :], others):
        ref = hf_greedy(hf, p, 40, eos={eos})
        assert out.token_ids == ref
    _check_idle(llm)


@pytest.mark.parametrize("algo", list(ALGOS))
def test_abort_mid_round(tiny_model, hf_reference, algo):
    hf = hf_reference(tiny_model)
    prompts = _prompts(5, seed=5)
    llm = _llm(tiny_model, algo, page_size=4)
    sp = SamplingParams(max_tokens=50, ignore_eos=True)
    outs = llm.generate(prompts, sp, abort_after={1: 3, 2: 1, 4: 12})
    for i, (out, prompt) in enumerate(zip(outs, prompts)):
        ref = hf_greedy(hf, prompt, 50)
        if i in (1, 2, 4):
            assert out.finish_reason == "abort"
            assert out.token_ids == ref[: len(out.token_ids)] and len(out.token_ids) < 50
        else:
            assert out.finish_reason == "length" and out.token_ids == ref
    _check_idle(llm)
    # the pool is still sound for the next batch
    outs = llm.generate(prompts[:2], sp)
    assert outs[0].token_ids == hf_greedy(hf, prompts[0], 50)
    _check_idle(llm)


@pytest.mark.parametrize("algo", list(ALGOS))
def test_sampling_is_seeded(tiny_model, hf_reference, algo):
    hf = hf_reference(tiny_model)
    prompts = _prompts(4, seed=6)
    hot = SamplingParams(max_tokens=20, temperature=0.9, top_k=50, ignore_eos=True)
    greedy = SamplingParams(max_tokens=20, ignore_eos=True)
    a = _llm(tiny_model, algo, seed=7).generate(prompts, [hot, greedy, hot, hot])
    b = _llm(tiny_model, algo, seed=7).generate(prompts, [hot, greedy, hot, hot])
    c = _llm(tiny_model, algo, seed=8).generate(prompts, [hot, greedy, hot, hot])
    assert [o.token_ids for o in a] == [o.token_ids for o in b]  # same seed -> same samples
    assert [o.token_ids for o in a] != [o.token_ids for o in c]
    assert all(len(o.token_ids) == 20 for o in a)
    # a greedy row in a mixed batch is still exactly greedy
    assert a[1].token_ids == hf_greedy(hf, prompts[1], 20)


def test_top_k_1_sampling_equals_greedy(tiny_model):
    prompts = _prompts(3, seed=8)
    llm = _llm(tiny_model, "draft-1layer")
    sp = SamplingParams(max_tokens=15, temperature=1.0, top_k=1, ignore_eos=True)
    greedy = SamplingParams(max_tokens=15, ignore_eos=True)
    assert [o.token_ids for o in llm.generate(prompts, sp)] == [
        o.token_ids for o in llm.generate(prompts, greedy)
    ]


def test_overlap_is_disabled_with_speculation(tiny_model):
    llm = LLM(tiny_model, load_tokenizer_=False, device="cpu", max_total_tokens=512, max_seq_len=128,
              disable_overlap=False, speculative_algorithm="ngram")
    assert not llm.scheduler.overlap
    plain = LLM(tiny_model, load_tokenizer_=False, device="cpu", max_total_tokens=512, max_seq_len=128,
                disable_overlap=False)
    assert plain.scheduler.overlap


def test_server_args_wire_speculation():
    from minisgl.server.args import parse_args

    args = parse_args(["--model", "m", "--speculative-algorithm", "draft_model",
                       "--speculative-num-draft-tokens", "3", "--speculative-draft-model", "d",
                       "--speculative-draft-num-layers", "2"])
    sc = args.scheduler_config()
    assert (sc.speculative_algorithm, sc.speculative_num_draft_tokens) == ("draft_model", 3)
    assert (sc.speculative_draft_model, sc.speculative_draft_num_layers) == ("d", 2)
    assert sc.spec_reserve_tokens == 3
    with pytest.raises(SystemExit):
        parse_args(["--model", "m", "--speculative-algorithm", "ngram", "--enable-overlap"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "m", "--speculative-algorithm", "draft_model", "--tp-size", "2"])
