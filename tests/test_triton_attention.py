"""Part III Ch1: Triton paged attention == the gather + SDPA torch backend.

On machines without a GPU the kernels run in the Triton interpreter
(``TRITON_INTERPRET=1``, set by conftest.py), so the shapes are kept small.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("triton")

from minisgl.attention import TorchAttnBackend  # noqa: E402
from minisgl.attention.triton_backend import TritonAttnBackend  # noqa: E402
from minisgl.config import ModelConfig  # noqa: E402
from minisgl.core import Batch  # noqa: E402
from minisgl.kvcache import MHAKVCache  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FakeReq:  # the only fields attention backends read
    table_idx: int
    cached_len: int
    device_len: int

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


def _model_config(hq: int, hkv: int, d: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=8, hidden_size=hq * d, intermediate_size=8, num_layers=1, num_qo_heads=hq,
        num_kv_heads=hkv, head_dim=d, rms_norm_eps=1e-6, rope_theta=1e4,
        max_position_embeddings=1024, tie_word_embeddings=True,
    )


def _setup(lens, hq, hkv, d, page_size, dtype, seed=0):
    """lens: list of (cached_len, extend_len). Pages are handed out in a random
    order so every request's K/V is scattered across the pool."""
    g = torch.Generator().manual_seed(seed)
    random.seed(seed)
    max_len = max(c + e for c, e in lens)
    pages_per_req = [-(-(c + e) // page_size) for c, e in lens]
    num_pages = sum(pages_per_req) + 3
    cache = MHAKVCache(hkv, 1, d, num_pages, page_size, dtype, torch.device(DEVICE))
    rows = len(lens) + 2
    page_table = torch.zeros((rows, max_len + 5), dtype=torch.int64, device=DEVICE)
    free = list(range(num_pages))
    random.shuffle(free)
    reqs, out_loc = [], []
    for i, (c, e) in enumerate(lens):
        row = rows - 1 - i  # rows deliberately not 0..B-1
        pages = [free.pop() for _ in range(pages_per_req[i])]
        slots = torch.tensor([p * page_size + o for p in pages for o in range(page_size)])
        page_table[row, : c + e] = slots[: c + e].to(DEVICE)
        reqs.append(FakeReq(row, c, c + e))
        out_loc.append(slots[c : c + e])
    # random history everywhere (stale data outside a request's range must not leak in)
    cache._kv_buffer.copy_(torch.randn(cache._kv_buffer.shape, generator=g).to(dtype))
    n = sum(e for _, e in lens)
    q = torch.randn(n, hq, d, generator=g).to(DEVICE, dtype)
    k = torch.randn(n, hkv, d, generator=g).to(DEVICE, dtype)
    v = torch.randn(n, hkv, d, generator=g).to(DEVICE, dtype)
    phase = "decode" if all(e == 1 for _, e in lens) else "prefill"
    batch = Batch(reqs=reqs, phase=phase)
    batch.out_loc = torch.cat(out_loc).to(DEVICE)
    return cache, page_table, batch, q, k, v, _model_config(hq, hkv, d)


def _compare(lens, hq=4, hkv=2, d=16, page_size=1, dtype=torch.float32, splits=1):
    cache, pt, batch, q, k, v, cfg = _setup(lens, hq, hkv, d, page_size, dtype)
    ref_backend = TorchAttnBackend(cache, pt)
    ref_backend.prepare_metadata(batch)
    ref = ref_backend.forward(q, k, v, 0, batch)
    tri_backend = TritonAttnBackend(cache, pt, cfg, num_kv_splits=splits)
    tri_backend.prepare_metadata(batch)
    out = tri_backend.forward(q, k, v, 0, batch)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
    assert torch.equal(batch.attn_metadata.get_last_indices().cpu(),
                       torch.cumsum(torch.tensor([e for _, e in lens]), 0) - 1)


DECODE_LENS = [(0, 1), (6, 1), (37, 1), (100, 1), (64, 1)]  # cached_len, 1 new token


@pytest.mark.parametrize("splits", [1, 3, 8])
@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_decode_matches_torch(splits, page_size):
    _compare(DECODE_LENS, page_size=page_size, splits=splits)


@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2), (6, 1)])
def test_decode_gqa_groups(hq, hkv):
    _compare(DECODE_LENS, hq=hq, hkv=hkv, page_size=4, splits=4)


def test_decode_more_splits_than_tokens():
    # empty splits must get zero weight in the LSE merge
    _compare([(0, 1), (2, 1)], splits=8)


def test_decode_odd_head_dim():
    _compare(DECODE_LENS, d=24, splits=2)  # BLOCK_D > HEAD_DIM: masked lanes


EXTEND_LENS = [(0, 7), (13, 5), (32, 40), (5, 1), (0, 33)]  # prefill, extend, chunk, 1-token


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_extend_matches_torch(page_size):
    _compare(EXTEND_LENS, page_size=page_size)


@pytest.mark.parametrize("hq,hkv,d", [(8, 2, 16), (6, 1, 24), (4, 4, 32)])
def test_extend_gqa_and_head_dims(hq, hkv, d):
    _compare(EXTEND_LENS, hq=hq, hkv=hkv, d=d, page_size=2)


@pytest.mark.parametrize("kind", ["decode", "extend"])
def test_bf16(kind):
    lens = DECODE_LENS if kind == "decode" else EXTEND_LENS
    _compare(lens, dtype=torch.bfloat16, page_size=4, splits=4)


# ------------------------------------------------------------- end to end
from conftest import hf_greedy  # noqa: E402

from minisgl import SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402


@pytest.mark.parametrize(
    "fixture,page_size,splits,max_running_req,prefill_budget",
    [
        ("tiny_model", 1, 1, 16, 2048),
        ("tiny_model", 4, 4, 3, 16),  # paged + split-KV + chunked prefill + queueing
        ("tiny_model_untied", 2, 8, 8, 2048),  # GQA group 4 (1 KV head)
    ],
)
def test_triton_backend_greedy_matches_hf(
    request, hf_reference, monkeypatch, fixture, page_size, splits, max_running_req, prefill_budget
):
    path = request.getfixturevalue(fixture)
    monkeypatch.setenv("MINISGL_TRITON_KV_SPLITS", str(splits))
    hf = hf_reference(path)
    rng = random.Random(1)
    shared = [rng.randrange(500) for _ in range(6)]
    prompts = [shared + [rng.randrange(500) for _ in range(rng.randrange(1, 20))] for _ in range(5)]
    lengths = [rng.randrange(1, 12) for _ in prompts]
    llm = LLM(
        path, load_tokenizer_=False, device=DEVICE, dtype="float32", attention_backend="triton",
        page_size=page_size, max_running_req=max_running_req, prefill_budget=prefill_budget,
        max_total_tokens=1024, max_seq_len=256,
    )
    assert isinstance(llm.engine.attn_backend, TritonAttnBackend)
    sps = [SamplingParams(max_tokens=n, ignore_eos=True) for n in lengths]
    for _ in range(2):  # second round: radix-cache hits -> extend with a prefix
        outs = llm.generate(prompts, sps)
        for out, prompt, n in zip(outs, prompts, lengths):
            assert out.token_ids == hf_greedy(hf, prompt, n)
    assert llm.scheduler.prefill_manager.stats.cached_tokens > 0
