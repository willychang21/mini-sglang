"""Part II Ch2 lab: follow one request (by uid) through the offline pipeline.

What it shows
  1. Tokenize: ``TokenizeMsg(uid, text)`` -> ``TokenizeManager`` -> ``UserMsg(uid, input_ids)``,
     exactly what ``tokenize_worker`` does, then the message goes into the
     scheduler's inbox (``OfflineIO`` stands in for ZMQ).
  2. One ``Scheduler.normal_loop`` per step: receive -> schedule -> forward ->
     ``_process_last_data``, which emits one ``DetokenizeMsg`` per request per step;
     ``DetokenizeManager`` turns each into a ``UserReply`` (incremental text).
  3. Where a request finishes: the length limit (``not req.can_decode``) and EOS
     (``token in eos_token_ids`` unless ``ignore_eos``) in ``_process_last_data``.
  4. Why the detokenizer keeps ``surr_offset`` / ``read_offset``: with the real
     Qwen3 tokenizer the emoji in "KV cache 🚀 ok" is split over three tokens.
  5. Cancellation: ``LLM.generate(..., abort_after={0: 2})`` -> ``AbortBackendMsg`` ->
     ``finish_reason="abort"``, and every resource comes back.

The engine uses tiny random weights with the real Qwen3 vocabulary, so text in /
text out works (the generated text is gibberish) and every step is fast.

    python labs/part2_ch02_request_path.py [--max-tokens 5]
"""

from __future__ import annotations

from _common import REAL_MODEL, lab_parser, make_llm, make_tiny_model, quiet_logs, section, table


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--prompt", default="What is KV Cache?")
    args = p.parse_args(argv)
    quiet_logs()

    from minisgl import SamplingParams
    from minisgl.message import DetokenizeMsg, TokenizeMsg, UserMsg, UserReply
    from minisgl.tokenizer import DetokenizeManager, TokenizeManager, load_tokenizer

    if args.model == "tiny":  # tiny weights + real tokenizer
        path = make_tiny_model(with_tokenizer=True)
    else:
        from _common import resolve_model

        path = resolve_model(args.model)
    llm = make_llm(path, cache_type="naive", max_total_tokens=512)
    sched, io = llm.scheduler, llm.io
    tok = llm.tokenizer
    assert tok is not None
    rows0, pages0 = sched.table_manager.available_size, sched.cache_manager.available_size

    # ------------------------------------------------------------------ 1
    section("1. API server -> tokenizer -> scheduler inbox (same uid)")
    uid = 42  # the API server's FrontendManager.new_user() would assign this
    chat = [{"role": "user", "content": args.prompt}]
    tmsg = TokenizeMsg(uid=uid, text=chat, sampling_params=SamplingParams(max_tokens=args.max_tokens, ignore_eos=True),
                       chat_template_kwargs={"enable_thinking": False})
    tm = TokenizeManager(tok)
    templated = tm.apply_chat_template(tmsg.text, **tmsg.chat_template_kwargs)
    (ids,) = tm.tokenize([tmsg])
    umsg = UserMsg(uid=tmsg.uid, input_ids=ids, sampling_params=tmsg.sampling_params)  # as tokenize_worker
    print(f"TokenizeMsg(uid={tmsg.uid}, text={tmsg.text!r})")
    print(f"chat template -> {templated!r}")
    print(f"UserMsg(uid={umsg.uid}, input_ids=<{len(ids)} ids> {ids[:6]}..., max_tokens={umsg.sampling_params.max_tokens})")
    assert umsg.uid == tmsg.uid and ids == tok.encode(templated, add_special_tokens=False)
    io.put(umsg)  # OfflineIO.inbox plays the role of the ZMQ socket into scheduler rank 0

    # ------------------------------------------------------------------ 2
    section("2. scheduler steps: DetokenizeMsg -> DetokenizeManager -> UserReply")
    detok = DetokenizeManager(tok)
    rows, replies_all = [], []
    step = 0
    while True:
        n_prefill = sched.stats.prefill_steps
        sched.normal_loop()  # receive -> _schedule_next_batch -> _forward -> _process_last_data
        out, io.outbox = io.outbox, []
        replies = detok.detokenize(out)
        phase = "prefill" if sched.stats.prefill_steps > n_prefill else "decode"
        for m, r in zip(out, replies):
            assert isinstance(m, DetokenizeMsg) and isinstance(r, UserReply) and m.uid == r.uid == uid
            rows.append([step, phase, m.uid, m.next_token, m.finished, m.finish_reason,
                         repr(r.incremental_output), r.finished])
            replies_all.append(r)
        step += 1
        if replies and replies[-1].finished:
            break
    print(table(rows, ["step", "phase", "uid", "next_token", "finished", "finish_reason",
                       "UserReply.incremental_output", "reply.finished"]))
    text = "".join(r.incremental_output for r in replies_all)
    tokens = [r[3] for r in rows]
    print(f"concatenated text == tokenizer.decode(all tokens): {text == tok.decode(tokens, skip_special_tokens=True)}")
    assert rows[0][1] == "prefill" and all(r[1] == "decode" for r in rows[1:])
    assert len(rows) == args.max_tokens and rows[-1][5] == "length"
    assert text == tok.decode(tokens, skip_special_tokens=True)
    assert uid not in detok.status, "detokenizer must drop the uid when finished"
    print("-> one DetokenizeMsg per step; finish_reason='length' set when not req.can_decode")
    print("   (remain_len = max_device_len - device_len reached 0) in Scheduler._process_last_data.")

    # ------------------------------------------------------------------ 3
    section("3. EOS vs ignore_eos (pretend the 3rd generated token is EOS)")
    fake_eos = tokens[2]
    saved = sched.eos_token_ids
    sched.eos_token_ids = {fake_eos}  # lab-only monkeypatch of the scheduler's EOS set
    try:
        res = []
        for ignore in (False, True):
            o = llm.generate([ids], SamplingParams(max_tokens=args.max_tokens, ignore_eos=ignore))[0]
            res.append([ignore, len(o.token_ids), o.finish_reason, o.token_ids])
    finally:
        sched.eos_token_ids = saved
    print(table(res, ["ignore_eos", "tokens", "finish_reason", "token_ids"]))
    assert res[0][1] == 3 and res[0][2] == "stop" and res[0][3][-1] == fake_eos  # EOS is included in the output
    assert res[1][1] == args.max_tokens and res[1][2] == "length"
    print("-> the EOS check `token in self.eos_token_ids` is skipped when ignore_eos=True.")

    # ------------------------------------------------------------------ 4
    section("4. incremental detokenization: surr_offset / read_offset")
    real_tok = load_tokenizer(REAL_MODEL)
    s = "KV cache 🚀 ok"
    sids = real_tok.encode(s, add_special_tokens=False)
    dm = DetokenizeManager(real_tok)
    trace, emitted = [], []
    for i, t in enumerate(sids):
        fin = i == len(sids) - 1
        st = dm.status.get(7)
        before = (st.surr_offset, st.read_offset) if st else (0, 0)
        (r,) = dm.detokenize([DetokenizeMsg(uid=7, next_token=t, finished=fin)])
        emitted.append(r.incremental_output)
        st = dm.status.get(7)
        after = (st.surr_offset, st.read_offset) if st else "(dropped)"
        trace.append([t, repr(real_tok.convert_ids_to_tokens(t)), repr(real_tok.decode([t])), before, after,
                      repr(r.incremental_output)])
    print(table(trace, ["token", "BPE piece", "decode([t]) alone", "(surr,read) before", "after", "emitted"]))
    naive = "".join(real_tok.decode([t]) for t in sids)
    inc = "".join(emitted)
    print(f"per-token decode : {naive!r}")
    print(f"incremental      : {inc!r}")
    assert "�" in naive and inc == s and "�" not in inc
    assert emitted.count("") == 2, "the two partial-emoji tokens are held back"
    print("-> the 3 tokens of 🚀 each decode to '�' alone; the manager holds text that ends in '�'")
    print("   until the last byte arrives, then emits ' 🚀' at once.")

    # ------------------------------------------------------------------ 5
    section("5. abort: LLM.generate(..., abort_after={0: 2})")
    outs = llm.generate([ids, ids[:-1]], SamplingParams(max_tokens=8, ignore_eos=True), abort_after={0: 2})
    print(table([[i, len(o.token_ids), o.finish_reason, repr(o.text)] for i, o in enumerate(outs)],
                ["prompt", "tokens", "finish_reason", "text"]))
    print(f"aborted_reqs={sched.stats.aborted_reqs}; free table rows {sched.table_manager.available_size}/{rows0}; "
          f"free pages {sched.cache_manager.available_size}/{pages0}")
    assert outs[0].finish_reason == "abort" and len(outs[0].token_ids) == 2
    assert outs[1].finish_reason == "length" and len(outs[1].token_ids) == 8
    assert sched.stats.aborted_reqs == 1
    assert sched.table_manager.available_size == rows0 and sched.cache_manager.available_size == pages0
    sched.cache_manager.check_integrity()
    print("-> AbortBackendMsg: PrefillManager.abort_req / DecodeManager.abort_req, _free_req_resources,")
    print("   then a final DetokenizeMsg(finished=True, finish_reason='abort') so the detokenizer drops the uid.")


if __name__ == "__main__":
    main()
