"""Part II Ch3 lab: tensor shapes, residuals, causality and sampling in one forward pass.

What it shows
  1. Forward hooks on a *packed* prefill batch of 2 requests (lengths 5 and 3):
     embedding ``[N, C]`` -> fused QKV ``[N, (Hq+2*Hkv)*D]`` -> q ``[N, Hq, D]``,
     k/v ``[N, Hkv, D]`` -> attention ``[N, Hq, D]`` -> o_proj / MLP ``[N, C]`` ->
     final norm ``[N, C]`` -> LM head ``[B, V]`` (only the last token of each request).
  2. ``attn_metadata.get_last_indices()`` and that the ``[B, V]`` logits equal the
     rows of the all-position logits at those indices.
  3. Where the residual addition happens: inside ``RMSNormFused`` (``h = x + residual``).
  4. Causal mask: changing a future token leaves earlier positions' logits unchanged.
     A small manual re-implementation run *without* the mask shows that earlier
     positions then change too.
  5. ``Hkv = 2`` vs ``Hkv = 1`` (``make_tiny_model(num_key_value_heads=1)``): which shapes change.
  6. Sampling with ``sample_impl``: greedy vs temperature / top-k / top-p on fixed logits.

    python labs/part2_ch03_forward_shapes.py [--model tiny|real]
"""

from __future__ import annotations

from _common import lab_parser, make_engine, make_tiny_model, quiet_logs, resolve_model, section, table

LENS = (5, 3)


def packed_prefill(engine, seqs, all_logits=False):
    """Build a prefill Batch of several requests by hand (what Scheduler._prepare_batch does)."""
    import torch

    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.kvcache.naive_cache import NaiveCacheHandle

    reqs, slot = [], 0
    for i, toks in enumerate(seqs):
        n = len(toks)
        engine.page_table[i, :n] = torch.arange(slot, slot + n)
        slot += n
        reqs.append(Req(input_ids=torch.tensor(toks), table_idx=i, cached_len=0, output_len=1, uid=i,
                        sampling_params=SamplingParams(), cache_handle=NaiveCacheHandle(engine.device)))
    batch = Batch(reqs=reqs, phase="prefill", return_all_logits=all_logits)
    batch.input_ids = torch.tensor([t for s in seqs for t in s], device=engine.device)
    batch.positions = torch.cat([torch.arange(len(s)) for s in seqs]).to(engine.device)
    batch.out_loc = torch.cat([engine.page_table[i, : len(s)] for i, s in enumerate(seqs)])
    engine.attn_backend.prepare_metadata(batch)
    return batch


def traced_forward(engine, seqs):
    """Run one packed prefill with hooks on layer 0; return (logits, [(stage, shape)], captured)."""
    rows, cap, handles = [], {}, []
    m = engine.model
    layer = m.model.layers[0]

    def hook(name, idx=None):
        def fn(mod, inp, out):
            o = out[idx] if idx is not None else out
            rows.append((name, tuple(o.shape)))
            cap[name] = o.detach().clone()
        return fn

    def rot_hook(mod, inp, out):
        rows.append(("q after QK-norm + RoPE", tuple(out[0].shape)))
        rows.append(("k after QK-norm + RoPE", tuple(out[1].shape)))

    handles.append(m.model.embed_tokens.register_forward_hook(hook("embedding")))
    handles.append(layer.input_layernorm.register_forward_hook(hook("L0 input_layernorm", 0)))
    handles.append(layer.self_attn.qkv_proj.register_forward_hook(hook("L0 qkv_proj (fused)")))
    handles.append(m.model.rotary_emb.register_forward_hook(rot_hook))
    orig_attn = engine.attn_backend.forward

    def attn(q, k, v, layer_id, batch):
        o = orig_attn(q, k, v, layer_id, batch)
        if layer_id == 0:
            rows.append(("v (no RoPE)", tuple(v.shape)))
            rows.append(("attention backend out", tuple(o.shape)))
        return o

    engine.attn_backend.forward = attn
    handles.append(layer.self_attn.o_proj.register_forward_hook(hook("L0 o_proj")))
    handles.append(layer.self_attn.register_forward_hook(hook("L0 self_attn out")))
    handles.append(layer.post_attention_layernorm.register_forward_hook(hook("L0 post_attn_layernorm", 0)))
    handles.append(layer.mlp.gate_up_proj.register_forward_hook(hook("L0 mlp.gate_up_proj")))
    handles.append(layer.mlp.register_forward_hook(hook("L0 mlp out")))
    handles.append(m.model.norm.register_forward_hook(hook("final norm", 0)))
    handles.append(m.lm_head.register_forward_hook(hook("lm_head (logits)")))
    try:
        batch = packed_prefill(engine, seqs)
        out = engine.forward_batch(batch, None)
    finally:
        for h in handles:
            h.remove()
        engine.attn_backend.forward = orig_attn
    # keep the first occurrence of each stage (layer-0 hooks fire once; rotary fires per layer)
    seen, uniq = set(), []
    for name, shape in rows:
        if name not in seen:
            seen.add(name)
            uniq.append((name, shape))
    return out.logits, uniq, cap, batch


def manual_forward(engine, tokens, causal=True):
    """Plain re-implementation of Qwen3 (one sequence) used to switch the causal mask off."""
    import torch
    import torch.nn.functional as F

    from minisgl.layers.norm import rms_norm

    m = engine.model.model
    cfg = engine.model_config
    Hq, Hkv, D = cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim
    ids = torch.tensor(tokens, device=engine.device)
    pos = torch.arange(len(tokens), device=engine.device)
    x = F.embedding(ids, m.embed_tokens.weight)
    for L in m.layers:
        a = L.self_attn
        h = rms_norm(x, L.input_layernorm.weight, L.input_layernorm.eps)
        q, k, v = F.linear(h, a.qkv_proj.weight).split([Hq * D, Hkv * D, Hkv * D], dim=-1)
        q = a.q_norm(q.reshape(-1, Hq, D))
        k = a.k_norm(k.reshape(-1, Hkv, D))
        v = v.reshape(-1, Hkv, D)
        q, k = m.rotary_emb(pos, q, k)
        k, v = k.repeat_interleave(Hq // Hkv, dim=1), v.repeat_interleave(Hq // Hkv, dim=1)  # GQA
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                           is_causal=causal).transpose(0, 1)
        x = x + F.linear(o.reshape(len(tokens), -1), a.o_proj.weight)
        h = rms_norm(x, L.post_attention_layernorm.weight, L.post_attention_layernorm.eps)
        gate, up = F.linear(h, L.mlp.gate_up_proj.weight).chunk(2, dim=-1)
        x = x + F.linear(F.silu(gate) * up, L.mlp.down_proj.weight)
    x = rms_norm(x, m.norm.weight, m.norm.eps)
    head = engine.model.lm_head
    return F.linear(x, head._tied[0].weight if head._tied else head.weight)


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    from minisgl.engine.sample import sample_impl
    from minisgl.speculative import SequenceRunner

    path = resolve_model(args.model)
    eng = make_engine(path, max_total_tokens=256, max_seq_len=128)
    cfg = eng.model_config
    C, Hq, Hkv, D, V = cfg.hidden_size, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.vocab_size
    seqs = [list(range(10, 10 + LENS[0])), list(range(40, 40 + LENS[1]))]
    N, B = sum(LENS), len(LENS)

    # ------------------------------------------------------------------ 1
    section(f"1. shapes through one packed prefill (B={B}, lengths {LENS}, N={N})")
    print(f"config: C={C} Hq={Hq} Hkv={Hkv} D={D} V={V} layers={cfg.num_layers}  (note Hq*D={Hq * D}"
          f"{' == C' if Hq * D == C else ' != C'})")
    logits, shapes, cap, batch = traced_forward(eng, seqs)
    print(table(shapes, ["stage (layer 0 unless noted)", "shape"]))
    s = dict(shapes)
    assert s["embedding"] == (N, C)
    assert s["L0 qkv_proj (fused)"] == (N, (Hq + 2 * Hkv) * D)
    assert s["q after QK-norm + RoPE"] == (N, Hq, D) and s["k after QK-norm + RoPE"] == (N, Hkv, D)
    assert s["v (no RoPE)"] == (N, Hkv, D) and s["attention backend out"] == (N, Hq, D)
    assert s["L0 o_proj"] == s["L0 mlp out"] == s["final norm"] == (N, C)
    assert s["L0 mlp.gate_up_proj"] == (N, 2 * cfg.intermediate_size)
    assert s["lm_head (logits)"] == (B, V) and tuple(logits.shape) == (B, V)
    print("-> no batch or time axis: the B requests are packed into one N-token axis;")
    print("   request boundaries live in batch.attn_metadata, not in the tensor shape.")

    # ------------------------------------------------------------------ 2
    section("2. LM head keeps only the last position of each request")
    last = batch.attn_metadata.get_last_indices().tolist()
    full = eng.forward_batch(packed_prefill(eng, seqs, all_logits=True), None).logits
    print(f"get_last_indices() = {last}   (cumsum(extend_lens) - 1 = cumsum({list(LENS)}) - 1)")
    print(f"return_all_logits=False -> {tuple(logits.shape)};  return_all_logits=True -> {tuple(full.shape)}")
    same = torch.allclose(full[last], logits, atol=1e-5)
    print(f"all_logits[last_indices] == default logits: {same}")
    print(f"LM-head output rows: {N} -> {B} ({N / B:.1f}x less vocabulary-projection work)")
    assert last == [LENS[0] - 1, sum(LENS) - 1] and tuple(full.shape) == (N, V) and same

    # ------------------------------------------------------------------ 3
    section("3. residual addition lives inside RMSNormFused")
    L0 = eng.model.model.layers[0]
    calls = []

    def norm_hook(name):
        def fn(mod, inp, out):
            x, res = inp[0], (inp[1] if len(inp) > 1 else None)
            calls.append((name, res is None, x.detach().clone(), None if res is None else res.detach().clone(),
                          out[1].detach().clone()))
        return fn

    hs = [L0.input_layernorm.register_forward_hook(norm_hook("L0.input_layernorm")),
          L0.post_attention_layernorm.register_forward_hook(norm_hook("L0.post_attention_layernorm"))]
    if cfg.num_layers > 1:
        hs.append(eng.model.model.layers[1].input_layernorm.register_forward_hook(norm_hook("L1.input_layernorm")))
    try:
        eng.forward_batch(packed_prefill(eng, seqs), None)
    finally:
        for h in hs:
            h.remove()
    rows = []
    for name, first, x, res, h in calls:
        ok = torch.equal(h, x) if first else torch.allclose(h, x + res)
        rows.append([name, "None (first call)" if first else "given", "x" if first else "x + residual", ok])
        assert ok
    print(table(rows, ["RMSNormFused call", "residual arg", "returned residual h", "verified"]))
    emb, attn_out = cap["embedding"], cap["L0 self_attn out"]
    assert torch.allclose(calls[1][4], emb + attn_out)
    print("-> L0.post_attention_layernorm returns h = embedding + attention(norm(embedding)); the MLP output")
    print("   is added later by L1.input_layernorm (or by the final model.norm after the last layer).")

    # ------------------------------------------------------------------ 4
    section("4. causal mask: change a future token, earlier positions do not move")
    T, j = 6, 3
    a = list(range(100, 100 + T))
    b = list(a)
    b[j] = 7  # change token at position j
    runner = SequenceRunner(eng)
    la = runner.extend(a, all_logits=True)
    runner.reset()
    lb = runner.extend(b, all_logits=True)
    runner.reset()
    diff_engine = (la - lb).abs().amax(-1)
    ma, mb = manual_forward(eng, a, causal=True), manual_forward(eng, b, causal=True)
    assert torch.allclose(ma, la, atol=1e-4), "manual re-implementation must match the engine"
    na, nb = manual_forward(eng, a, causal=False), manual_forward(eng, b, causal=False)
    diff_nomask = (na - nb).abs().amax(-1)
    rows = [[t, "changed" if t == j else "", f"{diff_engine[t]:.2e}", f"{diff_nomask[t]:.2e}"] for t in range(T)]
    print(f"sequence A = {a}, B = A with position {j} set to {b[j]}  (manual == engine: True)")
    print(table(rows, ["position", "", "max|logit A - B| engine (causal)", "manual, NO mask"]))
    assert torch.all(diff_engine[:j] == 0) and torch.all(diff_engine[j:] > 1e-4)
    assert torch.all(diff_nomask[:j] > 1e-4)
    print(f"-> with the mask positions 0..{j - 1} are bit-identical; without it every position changes.")

    # ------------------------------------------------------------------ 5
    section("5. Hkv = 2 vs Hkv = 1 (engine packed layout + tutorial's [B,T] example)")
    rows_h = {}
    for hkv in (Hkv, 1) if args.model == "tiny" else (Hkv,):
        e = eng if hkv == Hkv else make_engine(make_tiny_model(num_key_value_heads=hkv), max_total_tokens=256,
                                               max_seq_len=128)
        _, sh, _, _ = traced_forward(e, seqs)
        d = dict(sh)
        d["KV pool"] = tuple(e.kv_cache._kv_buffer.shape)
        rows_h[hkv] = d
    keys = ["L0 qkv_proj (fused)", "q after QK-norm + RoPE", "k after QK-norm + RoPE", "v (no RoPE)",
            "attention backend out", "L0 o_proj", "lm_head (logits)", "KV pool"]
    hk = list(rows_h)
    print(table([[k] + [rows_h[h][k] for h in hk] for k in keys], ["stage"] + [f"Hkv={h}" for h in hk]))
    if len(hk) == 2:
        changed = [k for k in keys if rows_h[hk[0]][k] != rows_h[hk[1]][k]]
        print(f"changed: {changed}")
        assert changed == ["L0 qkv_proj (fused)", "k after QK-norm + RoPE", "v (no RoPE)", "KV pool"]
    # tutorial Section 5 example, teaching layout: B=1, T=3, C=32, Hq=4, D=8, V=64
    trows = []
    for hkv in (2, 1):
        q, k = torch.randn(1, 4, 3, 8), torch.randn(1, hkv, 3, 8)
        kk = k.repeat_interleave(4 // hkv, dim=1)
        scores = q @ kk.transpose(-1, -2)
        trows.append([hkv, (1, 3), (1, 3, 32), tuple(q.shape), tuple(k.shape), tuple(scores.shape), (1, 3, 64), (1, 64)])
    print(table(trows, ["Hkv", "input", "embedding", "Q", "K and V", "scores", "all logits", "next-token"]))
    assert trows[0][5] == trows[1][5] == (1, 4, 3, 3)
    print("-> only K/V (and the fused QKV width / KV cache) shrink; scores keep Hq heads because")
    print("   each K/V head is shared by Hq/Hkv query heads.")

    # ------------------------------------------------------------------ 6
    section("6. sampling on fixed logits with sample_impl (4000 draws each)")
    base = torch.tensor([3.0, 2.2, 1.9, 1.0, 0.0, -1.0, -2.0, -4.0])
    R = 4000
    gen = torch.Generator().manual_seed(0)

    def draw(temp, k=-1, p=1.0):
        x = base.repeat(R, 1)
        toks = sample_impl(x, torch.full((R,), temp), torch.full((R,), k, dtype=torch.int64),
                           torch.full((R,), p), generator=gen)
        return torch.bincount(toks, minlength=len(base)).float() / R

    srt = torch.softmax(base, -1)
    cum_before = torch.cumsum(srt, 0) - srt
    nucleus = int((cum_before < 0.8).sum())  # smallest prefix whose mass reaches 0.8
    cases = [("greedy (T=0)", 0.0, -1, 1.0), ("T=0.5", 0.5, -1, 1.0), ("T=1.0", 1.0, -1, 1.0),
             ("T=2.0", 2.0, -1, 1.0), ("T=1, top_k=2", 1.0, 2, 1.0), ("T=1, top_p=0.8", 1.0, -1, 0.8)]
    rows, res = [], {}
    for name, t, k, pp in cases:
        f = draw(t, k, pp)
        res[name] = f
        rows.append([name] + [f"{v:.2f}" for v in f.tolist()] + [int((f > 0).sum())])
    print("logits  = " + str([round(v, 2) for v in base.tolist()]))
    print("softmax = " + str([round(v, 3) for v in srt.tolist()]))
    print(table(rows, ["policy"] + [f"tok{i}" for i in range(len(base))] + ["support"]))
    assert res["greedy (T=0)"][0] == 1.0
    assert res["T=0.5"][0] > res["T=1.0"][0] > res["T=2.0"][0]  # lower T = sharper
    assert int((res["T=1, top_k=2"] > 0).sum()) == 2
    assert int((res["T=1, top_p=0.8"] > 0).sum()) == nucleus
    assert abs(res["T=1.0"][0] - srt[0]) < 0.03
    mixed = sample_impl(base.repeat(2, 1), torch.tensor([0.0, 5.0]), torch.tensor([-1, -1]), torch.tensor([1.0, 1.0]),
                        generator=gen)
    assert int(mixed[0]) == 0
    print(f"-> top_p=0.8 keeps the {nucleus} most likely tokens (cumulative mass before each < 0.8);")
    print("   a temperature-0 row in a mixed batch still takes argmax (torch.where(greedy, ...)).")


if __name__ == "__main__":
    main()
