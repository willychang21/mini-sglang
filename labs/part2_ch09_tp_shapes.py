"""Part II Ch9 lab: what tensor parallelism does to every weight and to the KV pool.

What it shows
  1. Per-rank parameter shapes for tp = 1/2/4 (Qwen3-0.6B config, built on the
     ``meta`` device: no weights, no process group): qkv_proj, o_proj,
     gate_up_proj, down_proj, embed / lm_head (vocab range), KV-pool heads.
  2. KV-head replication when num_kv_heads < tp (tiny model: 2 KV heads at tp=4;
     Qwen3-0.6B: 8 KV heads at tp=16).
  3. Memory per rank vs tp (weights + KV bytes per token): why 2x TP does not
     halve memory.
  4. Shard-then-merge vs merge-then-shard for q/k/v (models/weight.py docstring):
     the real loader's shards reproduce the full attention output, the wrong
     order does not.
  5. (opt-in, --run-tp) a real TP=2 run over Gloo on CPU vs TP=1 (two processes,
     tests/tp_worker.py), comparing logits and generated tokens.

    python labs/part2_ch09_tp_shapes.py [--run-tp]
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import replace

from _common import REAL_MODEL, human_bytes, lab_parser, make_tiny_model, quiet_logs, section, table

QWEN3_06B_FALLBACK = dict(  # used only if /tmp/models/Qwen3-0.6B/config.json is missing
    vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_layers=28, num_qo_heads=16,
    num_kv_heads=8, head_dim=128, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=40960,
    tie_word_embeddings=True,
)


def _qwen_config():
    from minisgl.config import ModelConfig

    if os.path.exists(os.path.join(REAL_MODEL, "config.json")):
        return ModelConfig.from_pretrained(REAL_MODEL)
    return ModelConfig(**QWEN3_06B_FALLBACK)


def build_meta(config, rank: int, size: int):
    """Build the model (and a 1-page KV pool) exactly as rank ``rank`` of ``size`` would."""
    import torch

    from minisgl.distributed import set_tp_info
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.models import create_model

    set_tp_info(rank, size)
    try:
        model = create_model(config, torch.device("meta"), torch.bfloat16)
        pool = create_kvcache_pool(config, num_pages=1, page_size=1, dtype=torch.bfloat16, device=torch.device("meta"))
    finally:
        set_tp_info(0, 1)
    return model, pool


def kv_heads_held(config, rank: int, size: int):
    """Which global KV heads rank ``rank`` holds (mirrors models/weight.py:_shard_tensor)."""
    n = config.num_kv_heads
    if n < size:
        return [rank * n // size]  # replicated
    per = n // size
    return list(range(rank * per, (rank + 1) * per))


def shape_table(config, sizes, ranks_for=lambda tp: range(tp)):
    rows = []
    for tp in sizes:
        for r in ranks_for(tp):
            m, pool = build_meta(config, r, tp)
            l0 = m.model.layers[0]
            qkv = l0.self_attn.qkv_proj
            lm = m.lm_head
            lm_shape = "tied" if lm._tied else tuple(lm.weight.shape)
            rows.append([
                tp, r, tuple(qkv.weight.shape), f"{qkv.local_qo_heads}/{qkv.local_kv_heads}",
                tuple(l0.self_attn.o_proj.weight.shape), tuple(l0.mlp.gate_up_proj.weight.shape),
                tuple(l0.mlp.down_proj.weight.shape), tuple(m.model.embed_tokens.weight.shape),
                m.model.embed_tokens.vocab_range, lm_shape, pool._kv_buffer.shape[-2],
                kv_heads_held(config, r, tp),
            ])
    return rows


SHAPE_HEADERS = ["tp", "rank", "qkv_proj", "q/kv heads", "o_proj", "gate_up_proj", "down_proj",
                 "embed", "vocab_range", "lm_head", "KV pool heads", "KV heads held"]


def mini_attention(x, w_qkv, w_o, nq: int, nkv: int, d: int):
    """Causal GQA attention for one sequence (no RoPE / QK-norm: only the sharding matters)."""
    import torch

    q, k, v = (x @ w_qkv.T).split([nq * d, nkv * d, nkv * d], dim=-1)
    T = x.shape[0]
    q, k, v = q.view(T, nq, d), k.view(T, nkv, d), v.view(T, nkv, d)
    group = nq // nkv
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    att = torch.einsum("thd,shd->hts", q, k) / d**0.5
    att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")).softmax(-1)
    o = torch.einsum("hts,shd->thd", att, v).reshape(T, nq * d)
    return o @ w_o.T


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_tp(model: str, size: int, prompt, out: str, max_tokens: int = 8) -> dict:
    import torch

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "tp_worker.py")
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, worker, str(r), str(size), model, str(port), out, str(max_tokens),
             ",".join(map(str, prompt))],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        for r in range(size)
    ]
    for p in procs:
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err.decode()[-2000:]
    return torch.load(out)


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--run-tp", action="store_true", help="also run a real TP=2 forward over Gloo (~10-20 s)")
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    from minisgl.config import ModelConfig
    from minisgl.distributed import get_tp_info, set_tp_info
    from minisgl.models.weight import load_weight

    qcfg = _qwen_config()
    tiny_path = make_tiny_model()
    tcfg = ModelConfig.from_pretrained(tiny_path)

    # ------------------------------------------------------------------ 1
    section("1. per-rank shapes, Qwen3-0.6B (layer 0; meta device, bf16)")
    print(f"config: hidden={qcfg.hidden_size} q_heads={qcfg.num_qo_heads} kv_heads={qcfg.num_kv_heads} "
          f"head_dim={qcfg.head_dim} intermediate={qcfg.intermediate_size} vocab={qcfg.vocab_size} "
          f"layers={qcfg.num_layers} tied={qcfg.tie_word_embeddings}")
    rows = shape_table(qcfg, [1, 2, 4])
    print(table(rows, SHAPE_HEADERS))
    H, D, I, V = qcfg.hidden_size, qcfg.head_dim, qcfg.intermediate_size, qcfg.vocab_size
    for row in rows:
        tp = row[0]
        assert row[2] == ((qcfg.num_qo_heads + 2 * qcfg.num_kv_heads) * D // tp, H)  # column parallel
        assert row[4] == (H, qcfg.num_qo_heads * D // tp)  # row parallel: input dim split
        assert row[5] == (2 * I // tp, H) and row[6] == (H, I // tp)
        assert row[7] == (-(-V // tp), H)
    assert get_tp_info().size == 1
    print("-> column parallel (qkv, gate_up) splits dim 0 = outputs; row parallel (o, down) splits dim 1 =")
    print("   inputs and all-reduces; only vocab_range differs between ranks of the same tp.")
    untied = replace(qcfg, tie_word_embeddings=False)
    m, _ = build_meta(untied, 1, 2)
    print(f"untied variant: lm_head weight on rank 1/2 = {tuple(m.lm_head.weight.shape)} "
          "(local logits, then all-gather along the vocab dim)")

    # ------------------------------------------------------------------ 2
    section("2. KV-head replication when num_kv_heads < tp")
    rows = shape_table(tcfg, [1, 2, 4])
    print(f"tiny model: q_heads={tcfg.num_qo_heads} kv_heads={tcfg.num_kv_heads} head_dim={tcfg.head_dim}")
    print(table(rows, SHAPE_HEADERS))
    tp4 = [r for r in rows if r[0] == 4]
    assert [r[11] for r in tp4] == [[0], [0], [1], [1]] and all(r[10] == 1 for r in tp4)
    print("-> tp=4: every rank holds 1 KV head; ranks 0,1 share head 0 and ranks 2,3 share head 1,")
    print("   matching GQA (q head r uses kv head r // 2). The KV pool is NOT split 4 ways.")
    rows16 = shape_table(qcfg, [8, 16], ranks_for=lambda tp: [0, tp - 1])
    print(table([[r[0], r[1], r[3], r[10], r[11]] for r in rows16], ["tp", "rank", "q/kv heads", "KV pool heads", "KV heads held"]))
    assert rows16[-1][10] == 1 and rows16[-1][11] == [7]
    try:
        build_meta(qcfg, 0, 32)
        raise AssertionError("expected tp=32 to fail")
    except AssertionError as e:
        if "expected" in str(e):
            raise
        print(f"tp=32 is rejected at construction: {e}")

    # ------------------------------------------------------------------ 3
    section("3. memory per rank vs tp (Qwen3-0.6B, bf16 weights + KV per token)")
    rows, base = [], None
    for tp in [1, 2, 4, 8, 16]:
        m, pool = build_meta(qcfg, 0, tp)
        total = sum(p.numel() for p in m.parameters()) * 2
        repl = sum(p.numel() for n, p in m.named_parameters() if "norm" in n) * 2
        bufs = sum(b.numel() * b.element_size() for b in m.buffers())
        kv_tok = 2 * qcfg.num_layers * pool._kv_buffer.shape[-2] * qcfg.head_dim * 2
        base = base or (total, kv_tok)
        rows.append([tp, human_bytes(total), human_bytes(repl), human_bytes(bufs), f"{base[0] / total:.2f}x",
                     human_bytes(kv_tok), f"{base[1] / kv_tok:.2f}x", human_bytes(kv_tok * tp),
                     human_bytes(H * 2)])
    print(table(rows, ["tp", "weights/rank", "of which replicated (norms)", "rope cache/rank",
                       "weight shrink", "KV/token/rank", "KV shrink", "KV/token all ranks",
                       "hidden state/token/rank"]))
    assert rows[3][6] == "8.00x" and rows[4][6] == "8.00x"  # KV stops shrinking at tp=16
    m16, _ = build_meta(qcfg, 0, 16)
    print(f"tp=16 qkv_proj/rank = {tuple(m16.model.layers[0].self_attn.qkv_proj.weight.shape)} (not 4096/16=256 rows):")
    print("   k_proj/v_proj weights are replicated together with the KV heads -> 14.5x, not 16x.")
    print("-> weights shrink ~tp x while kv_heads >= tp (norms are tiny), but: per-rank replicated buffers (RoPE cache,")
    print("   page table, CUDA graphs on GPU) and the full-width hidden state [N, hidden] on every rank do")
    print("   not shrink; once kv_heads < tp, KV/token/rank stops shrinking and the total KV is duplicated;")
    print("   and each layer adds 2 all-reduces (o_proj, down_proj) to every forward pass.")

    # ------------------------------------------------------------------ 4
    section("4. shard-then-merge (the loader) vs merge-then-shard (wrong), tiny model layer 0")
    torch.manual_seed(0)
    x = torch.randn(5, tcfg.hidden_size)
    full = dict(load_weight(tiny_path, tcfg))  # tp=1: merged qkv_proj, full o_proj
    w_qkv, w_o = full["model.layers.0.self_attn.qkv_proj.weight"], full["model.layers.0.self_attn.o_proj.weight"]
    nq, nkv, d = tcfg.num_qo_heads, tcfg.num_kv_heads, tcfg.head_dim
    ref = mini_attention(x, w_qkv, w_o, nq, nkv, d)
    rows = []
    for tp in [2, 4]:
        good = torch.zeros_like(ref)
        bad = torch.zeros_like(ref)
        for r in range(tp):
            set_tp_info(r, tp)
            try:
                shard = dict(load_weight(tiny_path, tcfg))
            finally:
                set_tp_info(0, 1)
            lq = nq // tp
            lkv = max(nkv // tp, 1)
            wq_r, wo_r = shard["model.layers.0.self_attn.qkv_proj.weight"], shard["model.layers.0.self_attn.o_proj.weight"]
            good += mini_attention(x, wq_r, wo_r, lq, lkv, d)  # the all-reduce of o_proj
            if tp == 2:  # merge-then-shard: cut the merged [q;k;v] rows evenly
                bad += mini_attention(x, w_qkv.chunk(tp, dim=0)[r], wo_r, lq, lkv, d)
        err_good = (good - ref).abs().max().item()
        rows.append([tp, "shard q,k,v then cat (models/weight.py)", f"{err_good:.2e}"])
        assert err_good < 1e-4
        if tp == 2:
            err_bad = (bad - ref).abs().max().item()
            rows.append([tp, "cat q,k,v then chunk", f"{err_bad:.2e}"])
            assert err_bad > 1e-2
    print(table(rows, ["tp", "order", "max |sum_r out_r - full out|"]))
    q_rows, kv_rows = nq * d, nkv * d
    print(f"why: merged qkv has {q_rows}+{kv_rows}+{kv_rows} rows. chunk(2) gives rank 0 rows 0-63 = q heads 0-3,")
    print("   which it then splits as q=[q0,q1], k=[q2], v=[q3]; rank 1 gets only k/v rows. The correct")
    print(f"   rank 0 shard is cat(q[0:32], k[0:16], v[0:16]): q heads 0,1 with their own kv head 0.")

    # ------------------------------------------------------------------ 5
    section("5. real TP=2 over Gloo vs TP=1 (opt-in: --run-tp)")
    if not args.run_tp:
        print("skipped (pass --run-tp; tests/test_tp.py runs the same check in the test suite)")
        return
    prompt = [3, 17, 99, 4, 250, 8, 77, 5]
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        a = run_tp(tiny_path, 1, prompt, os.path.join(tmp, "tp1.pt"))
        b = run_tp(tiny_path, 2, prompt, os.path.join(tmp, "tp2.pt"))
        dt = time.perf_counter() - t0
    diff = (a["logits"] - b["logits"]).abs().max().item()
    print(f"prompt logits max |tp1 - tp2| = {diff:.2e}; tokens tp1={a['tokens']} tp2={b['tokens']} ({dt:.1f} s)")
    assert diff < 1e-4 and a["tokens"] == b["tokens"]


if __name__ == "__main__":
    main()
