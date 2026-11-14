"""Part II Ch4 / reference engine/graph.py: CUDA-graph decode padding.

The capture itself needs a GPU and is NOT tested here. Everything around it is:
batch-size buckets, the dummy page / dummy row, static-buffer padding, and a
full generation where every decode step runs through ``GraphRunner.replay`` in
its eager CPU stand-in (same static buffers, same padding).
"""

from __future__ import annotations

import random

import pytest
import torch
from conftest import hf_greedy

from minisgl import SamplingParams
from minisgl.config import EngineConfig
from minisgl.core import Batch
from minisgl.engine import Engine
from minisgl.engine.graph import GraphInputBuffers, GraphRunner, determine_graph_bs, pad_size
from minisgl.llm import LLM


def test_graph_bs_buckets():
    assert determine_graph_bs(None, None, 64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert determine_graph_bs(None, 20, 256) == [1, 2, 4, 8, 16]
    assert determine_graph_bs(None, 0, 256) == []
    assert determine_graph_bs([8, 1, 3, 3, 500], None, 64) == [1, 3, 8]
    buckets = [1, 2, 4, 8, 16]
    assert [pad_size(b, buckets) for b in (1, 2, 3, 5, 8, 9, 16)] == [1, 2, 4, 8, 8, 16, 16]
    assert pad_size(17, buckets) is None


def test_dummy_page_and_row(tiny_model):
    cfg = EngineConfig(tiny_model, device="cpu", page_size=4, max_total_tokens=64, max_running_req=3,
                       max_seq_len=32)
    engine = Engine(cfg)
    assert engine.num_pages == 16
    assert engine.kv_cache.k_cache(0).shape[0] == 17  # +1 dummy page
    assert engine.dummy_row == 3 and engine.page_table.shape[0] == 4
    assert engine.dummy_slot == 64
    assert bool((engine.page_table[3] == 64).all())
    assert engine.graph_runner is None  # CPU: no CUDA graphs


def test_buffers_pad_with_dummy_request():
    buf = GraphInputBuffers.create(8, torch.device("cpu"), dummy_slot=99)
    batch = Batch(reqs=[object()] * 3, phase="decode")
    batch.input_ids = torch.tensor([5, 6, 7])
    batch.positions = torch.tensor([10, 3, 7])
    batch.out_loc = torch.tensor([40, 41, 42])
    buf.out_loc.fill_(-1)  # stale values from a previous, larger step
    buf.copy_from(batch, padded_bs=4, dummy_slot=99)
    assert buf.input_ids[:4].tolist() == [5, 6, 7, 0]
    assert buf.positions[:4].tolist() == [10, 3, 7, 0]
    assert buf.out_loc[:4].tolist() == [40, 41, 42, 99]


@pytest.mark.parametrize("page_size", [1, 4])
def test_padded_decode_matches_hf(tiny_model, hf_reference, page_size):
    pytest.importorskip("triton")
    from minisgl.attention.triton_backend import TritonAttnBackend

    hf = hf_reference(tiny_model)
    # On a GPU machine run the real kernels (fp32 so the output can match HF
    # token for token). Mixing the Triton interpreter with compiled kernels in
    # one process does not work, so the CPU path is only taken without a GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = LLM(tiny_model, load_tokenizer_=False, device=device, dtype="float32",
              attention_backend="triton", page_size=page_size, max_total_tokens=512,
              max_seq_len=128, max_running_req=8)
    engine = llm.engine
    runner = GraphRunner(engine.model, engine.ctx, [1, 2, 4, 8], engine.dummy_row,
                         engine.dummy_slot, engine.device, use_cuda_graph=False)
    engine.graph_runner = runner
    cap = engine.attn_backend
    assert isinstance(cap, TritonAttnBackend) and cap.capture is not None

    replayed = []
    orig_replay = runner.replay

    def spy(batch):
        replayed.append((batch.size, pad_size(batch.size, runner.graph_bs)))
        return orig_replay(batch)

    runner.replay = spy
    rng = random.Random(0)
    prompts = [[rng.randrange(500) for _ in range(rng.randrange(1, 15))] for _ in range(5)]
    lengths = [3, 9, 5, 12, 7]  # requests finish at different steps -> bs 5, 4, 3, 2, 1
    outs = llm.generate(prompts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths])
    for out, prompt, n in zip(outs, prompts, lengths):
        assert out.token_ids == hf_greedy(hf, prompt, n)

    sizes = {bs for bs, _ in replayed}
    assert {5, 3, 1} <= sizes and all(p >= bs for bs, p in replayed)
    assert any(p > bs for bs, p in replayed)  # some steps really were padded
    # padding rows wrote into the dummy page only, at its first slot
    dummy_k = engine.kv_cache.flat_k_cache(0)[engine.dummy_slot]
    assert dummy_k.abs().sum() > 0
    # after replay the static metadata rows past bs belong to the dummy request
    bs, padded = replayed[-1]
    assert cap.capture.rows[bs:padded].tolist() == [engine.dummy_row] * (padded - bs)
    assert cap.capture.seq_lens[bs:padded].tolist() == [1] * (padded - bs)


def test_torch_backend_refuses_graphs(tiny_model):
    engine = Engine(EngineConfig(tiny_model, device="cpu", max_total_tokens=256, max_seq_len=64))
    with pytest.raises(AssertionError):
        GraphRunner(engine.model, engine.ctx, [1, 2], engine.dummy_row, engine.dummy_slot,
                    engine.device, use_cuda_graph=False)
