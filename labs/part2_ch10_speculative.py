"""Part II Ch10 lab: speculative decoding, round by round.

What it shows
  1. Per-round traces of ``SpeculativeDecoder`` (draft -> one target forward ->
     commit -> rewind): L, proposals, target argmax per verification row,
     accepted count, committed tokens, cache lengths after truncate; and the
     check output == target-only greedy (``autoregressive_generate``) for a
     truncated 1-layer draft, an n-gram drafter and the target itself as draft.
  2. Row alignment for prefix length 5 and draft length 3: which full-prefix rows
     verify which proposal and which row gives the bonus.
  3. p=[0.3,0.7], q=[0.6,0.4] (concept exercise 3): exact acceptance probs, the
     residual, the final distribution (analytic + Monte Carlo through
     ``rejection_commit``), and the wrong "resample from p" variant.
  4. ``greedy_commit`` on hand-made logits: rejected EOS, accepted EOS,
     max_tokens cutting off the bonus.
  5. The cost model of concept exercise 6.

    python labs/part2_ch10_speculative.py [--model tiny|real] [--k 3] [--max-tokens 12]
"""

from __future__ import annotations

from _common import lab_parser, make_engine, quiet_logs, resolve_model, section, table


def trace_generate(spec, prompt, sp, mod):
    """Run ``spec.generate`` while recording every round (monkeypatches the instance only)."""
    rounds, cur = [], {}
    orig_propose, orig_extend = spec.drafter.propose, spec.target.extend
    orig_truncate, orig_commit = spec.target.truncate, mod.greedy_commit

    def propose(seq, k, *a):
        props, q = orig_propose(seq, k, *a)
        cur.update(L=len(seq), proposals=list(props), target_cached=spec.target.length)
        return props, q

    def extend(tokens, all_logits=True):
        out = orig_extend(tokens, all_logits)
        if all_logits and cur:
            cur["rows"] = out.argmax(-1).tolist()
        return out

    def commit(proposals, verification, remaining, eos):
        out = orig_commit(proposals, verification, remaining, eos)
        cur.update(committed=list(out), remaining=remaining)
        return out

    def truncate(n):
        orig_truncate(n)
        cur["truncated_to"] = n
        rounds.append(dict(cur))
        cur.clear()

    spec.drafter.propose, spec.target.extend, spec.target.truncate = propose, extend, truncate
    mod.greedy_commit = commit
    try:
        out, stats = spec.generate(prompt, sp)
    finally:
        del spec.drafter.propose, spec.target.extend, spec.target.truncate
        mod.greedy_commit = orig_commit
    return out, stats, rounds


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=3, help="draft tokens per round")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--samples", type=int, default=20000, help="Monte Carlo samples for section 3")
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    import minisgl.speculative.decoder as decoder_mod
    from minisgl import SamplingParams
    from minisgl.speculative import (NgramDrafter, SequenceRunner, SpeculativeDecoder, autoregressive_generate,
                                     greedy_commit, rejection_commit)

    path = resolve_model(args.model)
    kw = dict(max_total_tokens=1024, max_seq_len=512)
    target = make_engine(path, **kw)
    vocab = target.model_config.vocab_size
    nl = target.model_config.num_layers
    sp = SamplingParams(max_tokens=args.max_tokens, ignore_eos=True)

    # ------------------------------------------------------------------ 1
    section("1. per-round traces (greedy), checked against target-only greedy")
    if args.model == "tiny":
        prompt = [3, 17, 99, 4, 250, 8, 77, 5]  # same prompt as tests/test_tp.py
    else:
        from minisgl.tokenizer import TokenizeManager, load_tokenizer

        tok = TokenizeManager(load_tokenizer(path))
        prompt = tok.tokenize_one("Repeat exactly: the cat sat on the mat. the cat sat on the mat. the cat")
    ref = autoregressive_generate(target, prompt, sp)
    print(f"prompt ({len(prompt)} tokens) = {prompt}")
    print(f"target-only greedy output        = {ref}")
    dl = nl - 1 if nl <= 3 else nl // 2  # truncated draft: tiny 2 of 3 layers, Qwen3-0.6B 14 of 28
    drafts = [
        (f"model: first {dl} of {nl} layers", lambda: make_engine(path, num_layers=dl, **kw)),
        ("n-gram prompt lookup", lambda: NgramDrafter(vocab)),
        ("model: target itself", lambda: make_engine(path, **kw)),
    ]
    summary = []
    for name, make in drafts:
        spec = SpeculativeDecoder(target, make(), num_draft_tokens=args.k)
        out, st, rounds = trace_generate(spec, prompt, sp, decoder_mod)
        print(f"\n-- draft = {name}")
        rows = []
        for i, r in enumerate(rounds):
            n_acc = st.accepted_per_round[i]
            rows.append([i + 1, r["L"], r["target_cached"], r["proposals"], r["rows"], n_acc, r["committed"],
                         r["L"] + len(r["committed"]), r["truncated_to"]])
            # every committed token is the target's argmax under the confirmed prefix
            assert r["committed"] == r["rows"][: len(r["committed"])]
            # the target cache never holds the newest confirmed token (correction / bonus)
            assert r["truncated_to"] == r["L"] + len(r["committed"]) - 1
        print(table(rows, ["round", "L", "tgt cache", "proposals", "target argmax rows", "accepted", "committed",
                           "new L", "tgt cache after"]))
        print(f"output == target-only greedy: {out == ref}")
        assert out == ref
        summary.append([name, st.rounds, st.proposed, st.accepted, f"{st.acceptance_rate:.0%}",
                        f"{st.tokens_per_target_forward:.2f}", st.target_forwards, st.draft_forwards])
    print()
    print(table(summary, ["draft", "rounds", "proposed", "accepted", "acceptance", "tokens/target fwd",
                          "target fwds", "draft fwds"]))
    assert summary[-1][4] == "100%"
    print(f"-> target-only needs {len(ref)} target forwards. 'tgt cache' = L-1 at the start of every round: the")
    print("   newest confirmed token is fed as the first verification input, never taken from a rejected draft.")
    print("   A round with no proposals (n-gram found no match) is just a plain decode step.")
    ng = NgramDrafter(vocab)
    demo = [1, 2, 3, 4, 9, 9, 1, 2, 3]
    props, _ = ng.propose(demo, 3, sp, set(), None)
    print(f"   n-gram on {demo}: suffix [1, 2, 3] occurred earlier -> proposes {props}")
    assert props == [4, 9, 9]

    # ------------------------------------------------------------------ 2
    section("2. row alignment: prefix length L=5, draft length k=3")
    seq = prompt[:5]
    g = autoregressive_generate(target, seq, SamplingParams(max_tokens=3, ignore_eos=True))
    proposals = [g[0], g[1], (g[2] + 1) % vocab]  # d1, d2 match the target, d3 is wrong
    runner = SequenceRunner(target, row=1)
    full = runner.extend(seq + proposals, all_logits=True)  # 8 rows, rows 0..7
    runner.reset()
    runner.extend(seq[:-1], all_logits=False)  # the decoder's invariant: cache holds L-1 tokens
    ver = runner.extend([seq[-1]] + proposals, all_logits=True)  # k+1 rows
    rows = []
    for j in range(4):
        purpose = f"verify proposal d{j + 1}={proposals[j]}" if j < 3 else "bonus token after full acceptance"
        sees = "x" + "".join(f"+d{i + 1}" for i in range(j))
        rows.append([4 + j, j, sees, purpose, int(full[4 + j].argmax())])
    print(f"x = seq[:5] = {seq}, proposals d1..d3 = {proposals}")
    print(table(rows, ["full-prefix row (L-1+j)", "verification row j", "conditioned on", "purpose", "argmax"]))
    assert torch.allclose(full[4:], ver, atol=1e-4)
    committed = greedy_commit(proposals, ver, remaining=10)
    print(f"greedy_commit -> {committed}: d1, d2 accepted, d3 rejected and replaced by row 6's argmax;")
    print("   row 7 (bonus) is discarded because it was conditioned on the rejected d3.")
    assert committed == g[:3]
    print("-> rows 4,5,6 verify d1,d2,d3 and row 7 gives the bonus; extend([seq[-1]] + proposals) on a cache")
    print("   holding seq[:-1] produces exactly those 4 rows (allclose), in ONE forward pass.")

    # ------------------------------------------------------------------ 3
    section("3. p=[0.3,0.7], q=[0.6,0.4]: rejection sampling preserves p")
    P, Q = torch.tensor([0.3, 0.7]), torch.tensor([0.6, 0.4])
    acc = torch.minimum(torch.ones(2), P / Q)
    rej = float((Q * (1 - acc)).sum())
    resid = (P - Q).clamp_min(0)
    resid = resid / resid.sum()
    final = Q * acc + rej * resid
    wrong = Q * acc + rej * P
    print(table([["A", 0.3, 0.6, float(acc[0]), float(Q[0] * acc[0]), float(resid[0]), float(final[0]), float(wrong[0])],
                 ["B", 0.7, 0.4, float(acc[1]), float(Q[1] * acc[1]), float(resid[1]), float(final[1]), float(wrong[1])]],
                ["token", "p", "q", "accept prob", "q*accept", "residual", "final", "resample-from-p (wrong)"]))
    print(f"P(reject) = {rej:.2f}; P(accept) = sum min(p,q) = {1 - rej:.2f}")
    assert torch.allclose(final, P) and torch.allclose(wrong, torch.tensor([0.39, 0.61]))
    g = torch.Generator().manual_seed(0)
    counts, accepted = torch.zeros(2), 0
    target_rows = torch.stack([P, P])  # verification row + bonus row
    for _ in range(args.samples):
        y = int(torch.multinomial(Q, 1, generator=g))
        tok = rejection_commit([y], Q[None], target_rows, remaining=1, generator=g)[0]
        counts[tok] += 1
        accepted += tok == y
    mc = counts / args.samples
    print(f"Monte Carlo through rejection_commit ({args.samples} draws): final = [{mc[0]:.3f}, {mc[1]:.3f}], "
          f"acceptance = {accepted / args.samples:.3f}")
    assert torch.allclose(mc, P, atol=0.02)
    P2, Q2 = torch.tensor([0.4, 0.6]), torch.tensor([0.8, 0.2])
    a2 = torch.minimum(torch.ones(2), P2 / Q2)
    wrong2 = Q2 * a2 + float((Q2 * (1 - a2)).sum()) * P2
    print(f"code exercise 3 (p=[0.4,0.6], q=[0.8,0.2]) resampling from p gives {[round(float(x), 2) for x in wrong2]}"
          " instead of [0.4, 0.6]")
    assert torch.allclose(wrong2, torch.tensor([0.56, 0.44]))

    # ------------------------------------------------------------------ 4
    section("4. greedy_commit edge cases on hand-made logits (vocab 10, EOS=9)")

    def rows_for(*argmaxes):
        return torch.nn.functional.one_hot(torch.tensor(argmaxes), 10).float() * 5

    cases = [
        ("rejected EOS", [5, 9, 7], rows_for(5, 3, 1, 2), 10),
        ("accepted EOS", [5, 9, 7], rows_for(5, 9, 7, 2), 10),
        ("all accepted, budget 4", [5, 6, 7], rows_for(5, 6, 7, 8), 4),
        ("all accepted, budget 3", [5, 6, 7], rows_for(5, 6, 7, 8), 3),
        ("budget 0", [5], rows_for(5, 6), 0),
    ]
    rows, results = [], {}
    for name, props, ver, remaining in cases:
        out = greedy_commit(props, ver, remaining, eos_ids={9})
        results[name] = out
        rows.append([name, props, ver.argmax(-1).tolist(), remaining, out, bool(out) and out[-1] == 9])
    print(table(rows, ["case", "proposals", "target argmax rows", "remaining", "committed", "ends request (EOS)"]))
    assert results["rejected EOS"] == [5, 3]
    assert results["accepted EOS"] == [5, 9]
    assert results["all accepted, budget 4"] == [5, 6, 7, 8]
    assert results["all accepted, budget 3"] == [5, 6, 7]
    assert results["budget 0"] == []
    print("-> a rejected EOS is replaced by the correction and generation continues; only a committed EOS")
    print("   stops. The bonus row is only used when budget remains after all k proposals.")

    # ------------------------------------------------------------------ 5
    section("5. cost model (concept exercise 6): 8 ms/step plain decode vs 20 ms/round")
    rows = []
    for m in [1.2, 2, 2.5, 4]:
        per_tok = 20 / m
        rows.append([m, f"{per_tok:.1f}", f"{8 / per_tok:.2f}x", "faster" if per_tok < 8 else "slower"])
    print(table(rows, ["E[committed per round]", "ms / token", "decode speedup", "verdict"]))
    assert 20 / 2 > 8 and 20 / 4 < 8
    print("-> break-even at 20/8 = 2.5 tokens per round. This is decode-only: prefill, queueing, draft")
    print("   memory (fewer concurrent requests) and busy GPUs at high concurrency are not in it.")


if __name__ == "__main__":
    main()
