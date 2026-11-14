"""Part II Ch4 lab: make the KV cache observable.

What it shows
  1. The physical pool layout ``[2, layers, pages, page_size, kv_heads, head_dim]``.
  2. ``_page_to_token``: one free-page entry expands into ``page_size`` token slots.
  3. A per-step trace of ``cached_len / device_len / extend_len``, ``positions``
     and ``out_loc`` for one request (prefill, then decode steps).
  4. Why ``allocate_paged`` must run *before* ``out_loc = page_table[input_mapping]``.
  5. KV cache vs. no cache: identical tokens, far fewer computed positions.

    python labs/part2_ch04_kv_cache.py [--model tiny|real] [--page-size 4]
"""

from __future__ import annotations

import time

from _common import lab_parser, make_llm, quiet_logs, resolve_model, section, table


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--page-size", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=4)
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    from minisgl import SamplingParams
    from minisgl.llm import generate_without_kv_cache

    path = resolve_model(args.model)
    llm = make_llm(path, page_size=args.page_size, max_total_tokens=256, cache_type="naive", load_tokenizer_=False)
    eng, sched = llm.engine, llm.scheduler
    cfg = eng.model_config

    # ------------------------------------------------------------------ 1
    section("1. physical KV pool (MHAKVCache._kv_buffer)")
    buf = eng.kv_cache._kv_buffer
    names = ["K/V", "layer", "page", "slot in page", "kv head", "head_dim"]
    print(table([[n, s] for n, s in zip(names, buf.shape)], ["dim meaning", "size"]))
    print(f"flat per-layer view used by store_kv: {tuple(eng.kv_cache.flat_k_cache(0).shape)}"
          "  = [num_pages*page_size, kv_heads, head_dim]")
    assert tuple(buf.shape) == (2, cfg.num_layers, eng.num_pages, args.page_size, cfg.num_kv_heads, cfg.head_dim)
    per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * buf.element_size()
    print(f"bytes per cached token = 2 x {cfg.num_layers} layers x {cfg.num_kv_heads} heads x "
          f"{cfg.head_dim} dim x {buf.element_size()} B = {per_token} B")
    print(f"pool total = {eng.kv_cache.num_bytes} B = per_token x {eng.num_pages * args.page_size} slots")
    assert eng.kv_cache.num_bytes == per_token * eng.num_pages * args.page_size

    # ------------------------------------------------------------------ 2
    section("2. free_slots holds the first slot of each free page")
    cm = sched.cache_manager
    print(f"page_size={cm.page_size}; free_slots[:5] = {cm.free_slots[:5].tolist()}")
    if cm.page_size > 1:
        slot = 2 * cm.page_size
        expanded = cm._page_to_token(torch.tensor([slot])).tolist()
        print(f"free_slots value {slot} -> page {slot // cm.page_size} -> token slots {expanded}")
        assert expanded == list(range(slot, slot + cm.page_size))

    # ------------------------------------------------------------------ 3
    section("3. per-step trace of one request")
    trace = []
    orig_prepare = sched._prepare_batch

    def traced_prepare(batch):
        before = {r.uid: (r.cached_len, r.device_len, r.extend_len) for r in batch.reqs}
        fi = orig_prepare(batch)  # allocate_paged + positions + out_loc + metadata
        for r in batch.reqs:
            c, d, e = before[r.uid]
            row = eng.page_table[r.table_idx, : r.device_len].tolist()
            trace.append([batch.phase, c, d, e, fi.batch.positions.tolist(), fi.batch.out_loc.tolist(), row,
                          len(sched.cache_manager.free_slots)])
        return fi

    sched._prepare_batch = traced_prepare
    prompt = list(range(10, 10 + args.prompt_len))
    out = llm.generate([prompt], SamplingParams(max_tokens=args.max_tokens, ignore_eos=True))[0]
    sched._prepare_batch = orig_prepare
    print(table(trace, ["phase", "cached_len", "device_len", "extend_len", "positions", "out_loc", "page_table row",
                        "free pages"]))
    print(f"generated tokens: {out.token_ids}")
    # checks: prefill computes the whole prompt, each decode step exactly one token
    assert trace[0][0] == "prefill" and trace[0][3] == args.prompt_len
    for i, row in enumerate(trace[1:], start=1):
        assert row[0] == "decode" and row[3] == 1, row
        assert row[1] == args.prompt_len + i - 1  # cached_len grows by one per step
        assert row[4] == [row[1]]  # the one new token sits at position cached_len
    assert len(trace) == args.max_tokens  # 1 prefill + (max_tokens - 1) decodes
    for prev, cur in zip(trace, trace[1:]):  # a page is consumed only on a page boundary
        crossed = (cur[2] - 1) // args.page_size != (prev[2] - 1) // args.page_size
        assert (prev[7] - cur[7]) == (1 if crossed else 0), (prev, cur)
    print("-> prefill: extend_len = prompt length; decode: extend_len = 1 and positions = [cached_len].")
    print("-> a new page is taken from free_slots only when device_len crosses a page boundary.")
    cm.check_integrity()

    # ------------------------------------------------------------------ 4
    section("4. why allocate_paged must come before out_loc = page_table[input_mapping]")
    free_row = sched.table_manager._free_rows[0]
    eng.page_table[free_row].zero_()  # a fresh row: nothing allocated yet
    stale = eng.page_table[free_row, 0:args.prompt_len].tolist()
    print(f"unallocated page_table row {free_row}, positions 0..{args.prompt_len - 1} -> out_loc = {stale}")
    print("-> every token's K/V would be scattered into slot 0 (or into whatever slots an earlier,")
    print("   finished request left in the row), silently corrupting another request's cache.")
    assert set(stale) == {0}

    # ------------------------------------------------------------------ 5
    section("5. KV cache vs. recomputing everything (Chapter 3 baseline)")
    n_new = args.max_tokens * 4
    t0 = time.perf_counter()
    ref = generate_without_kv_cache(eng, prompt, n_new)
    t_nocache = time.perf_counter() - t0
    t0 = time.perf_counter()
    cached = llm.generate([prompt], SamplingParams(max_tokens=n_new, ignore_eos=True))[0].token_ids
    t_cache = time.perf_counter() - t0
    computed_nocache = sum(args.prompt_len + i for i in range(n_new))
    computed_cache = args.prompt_len + (n_new - 1)
    print(table(
        [["no cache (llm/naive.py)", computed_nocache, f"{t_nocache * 1e3:.1f}"],
         ["KV cache (scheduler)", computed_cache, f"{t_cache * 1e3:.1f}"]],
        ["mode", "token positions run through the model", "wall ms"],
    ))
    print(f"identical output: {ref == cached}")
    assert ref == cached
    print("-> without a cache step t recomputes all prompt_len+t positions: sum is O(n^2) tokens;")
    print("   with it each decode step computes 1 new position (attention still reads all n).")


if __name__ == "__main__":
    main()
