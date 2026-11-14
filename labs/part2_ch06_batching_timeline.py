"""Part II Ch6 lab: watch continuous batching step by step.

What it shows (every step = one ``Scheduler.normal_loop`` iteration)
  1. The A/B/C example of Ch6 (concept §3.2 / code §5.2): two request slots,
     outputs 2/5/2, C arrives during step 0. The real scheduler's timeline is
     compared row by row with the tutorial's table.
  2. The five-request A..E example of concept §5.4 (2 slots, prefill budget 6).
  3. Concept exercise 3 / code exercise 4 variants: A generates 1 token; C needs
     two prefill chunks.
  4. Fixed (static) batching vs continuous batching on the same workload:
     step counts, first-token step, identical tokens.
  5. Chunked prefill with a small ``prefill_budget``: a ``ChunkedReq`` never
     commits a token, goes back to the *front* of the waiting queue, and the
     running request B is paused until the whole prompt is done.
  6. Why decode runs while the waiting queue is non-empty: admission blocked by
     request slots (part 1) or by KV capacity (here).
  7. Concept exercise 5: the KV-capacity inequality, evaluated by the real
     ``PrefillAdder._try_allocate_one``; the chapter's formula vs this repo's
     ``+ page_size`` variant.

    python labs/part2_ch06_batching_timeline.py [--model tiny|real]
"""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from _common import lab_parser, make_llm, quiet_logs, resolve_model, section, table

# arrivals: {step at whose start the scheduler receives the request: [(name, prompt_len, max_tokens)]}
# "arrives during step k" in the tutorial == received at the start of step k+1 here.
Arrivals = Dict[int, List[Tuple[str, int, int]]]


def run_timeline(path: str, arrivals: Arrivals, *, static: bool = False, **engine_kwargs):
    """Drive ``Scheduler.normal_loop`` by hand and record one row per step."""
    import torch  # noqa: F401  (engine import side effects)

    from minisgl import SamplingParams
    from minisgl.core import ChunkedReq
    from minisgl.message import UserMsg

    engine_kwargs.setdefault("cache_type", "naive")  # the chapter assumes no prefix hits
    llm = make_llm(path, load_tokenizer_=False, **engine_kwargs)
    sched, io = llm.scheduler, llm.io
    pm, dm = sched.prefill_manager, sched.decode_manager
    vocab = llm.engine.model_config.vocab_size

    name_of: Dict[int, str] = {}
    snap: dict = {}

    # --- instrumentation: wrap methods on *this instance* only -------------
    orig_schedule = sched._schedule_next_batch

    def traced_schedule():
        snap["waiting"] = [name_of[p.uid] + ("*" if p.chunked_req else "") for p in pm.pending_list]
        snap["running"] = sorted(name_of[r.uid] for r in dm.running_reqs)
        snap["free_slots"] = sched.table_manager.available_size
        snap["kv_available"] = sched.cache_manager.available_size
        snap["reserved"] = dm.inflight_tokens
        snap["ref_reserved"] = sum(r.remain_len + sched.page_size - 1 for r in dm.running_reqs)
        fi = orig_schedule()  # PrefillManager first, else DecodeManager, then _prepare_batch
        if fi is None:
            snap["phase"], snap["members"] = "idle", []
        else:
            snap["phase"] = fi.batch.phase
            snap["members"] = [
                (name_of[r.uid], r.cached_len, r.device_len, isinstance(r, ChunkedReq)) for r in fi.batch.reqs
            ]
        return fi

    sched._schedule_next_batch = traced_schedule

    # Record every KV/slot admission check the real PrefillAdder makes (class-level
    # patch, restored before returning; this cache is "naive" so cached_len = 0).
    from minisgl.scheduler.prefill import PrefillAdder

    orig_try = PrefillAdder._try_allocate_one

    def traced_try(adder, req):
        est = req.input_len + req.output_len + adder.page_size
        before = (adder.reserved_size, adder.cache_manager.available_size, adder.table_manager.available_size)
        result = orig_try(adder, req)
        snap.setdefault("checks", []).append((name_of[req.uid], est, *before, result is not None))
        return result

    PrefillAdder._try_allocate_one = traced_try

    if static:  # fixed batching: admit a new group only when the previous one is done
        orig_pf = pm.schedule_next_batch

        def gated(budget):
            if dm.running_reqs and not any(p.chunked_req for p in pm.pending_list):
                return None
            return orig_pf(budget)

        pm.schedule_next_batch = gated

    rows, produced, tokens = [], defaultdict(int), defaultdict(list)
    first_step, finish_step = {}, {}
    names = [n for v in arrivals.values() for n, _, _ in v]
    uid, step, last_arrival = 0, 0, max(arrivals)
    while True:
        for name, plen, max_tokens in arrivals.get(step, []):
            uid += 1
            name_of[uid] = name
            prompt = [(17 * uid + 5 * i + 3) % (vocab - 8) + 4 for i in range(plen)]
            io.put(UserMsg(uid=uid, input_ids=prompt,
                           sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True)))
        snap.clear()
        sched.normal_loop()
        replies, io.outbox = io.outbox, []
        events = []
        for m in replies:
            n = name_of[m.uid]
            if m.next_token >= 0:
                produced[n] += 1
                tokens[n].append(m.next_token)
                first_step.setdefault(n, step)
            if m.finished:
                finish_step[n] = step
                events.append(f"{n} done")
        rows.append(dict(step=step, outputs={n: produced[n] for n in names}, events=events, **snap))
        step += 1
        if step > last_arrival and not sched.has_work and not io.has_pending():
            break
        assert step < 500, "runaway timeline"
    PrefillAdder._try_allocate_one = orig_try
    sched.cache_manager.check_integrity()  # every page is back once idle
    assert sched.table_manager.available_size == llm.config.max_running_req
    stats = sched.stats
    llm.shutdown()
    del llm, sched, pm, dm, io, orig_schedule, traced_schedule
    gc.collect()  # one engine at a time: matters with --model real (2.4 GB of fp32 weights)
    return dict(rows=rows, tokens=dict(tokens), first=first_step, finish=finish_step, names=names,
                stats=stats)


def fmt_batch(row) -> str:
    if row.get("phase", "idle") == "idle":
        return "idle"
    parts = [f"{n}[{c}->{d}){'*' if ch else ''}" for n, c, d, ch in row["members"]]
    return f"{row['phase']}: " + " ".join(parts)


def fmt_checks(row) -> str:
    out = []
    for name, est, reserved, avail, slots, ok in row.get("checks", []):
        if slots == 0:
            out.append(f"{name}: no free slot")
        else:
            out.append(f"{name}: {est}+{reserved}={est + reserved} {'<=' if ok else '>'} {avail}")
    return "; ".join(out)


def short(row) -> Tuple[str, List[str]]:
    return row.get("phase", "idle"), [m[0] for m in row.get("members", [])]


def print_timeline(res, *, extra=()) -> None:
    names = res["names"]
    out = []
    for r in res["rows"]:
        line = [r["step"], " ".join(r.get("waiting", [])) or "-", " ".join(r.get("running", [])) or "-",
                r.get("free_slots", "-"), fmt_batch(r), "/".join(str(r["outputs"][n]) for n in names),
                ", ".join(r["events"]) or ""]
        line += [f(r) for _, f in extra]
        out.append(line)
    headers = ["step", "waiting", "running", "free slots", "batch: req[cached->device) *=chunked",
               "outputs " + "/".join(names), "events"] + [h for h, _ in extra]
    print(table(out, headers))


def compare(res, expected: Sequence[Tuple[str, List[str], Sequence[int] | None]]) -> None:
    """Row-by-row comparison with a tutorial table: (phase, members, cumulative outputs)."""
    names = res["names"]
    out, ok = [], True
    for i, (phase, members, outs) in enumerate(expected):
        got_phase, got_members = short(res["rows"][i]) if i < len(res["rows"]) else ("-", [])
        got_outs = [res["rows"][i]["outputs"][n] for n in names] if i < len(res["rows"]) else []
        match = (got_phase, got_members) == (phase, members) and (outs is None or list(outs) == got_outs)
        ok &= match
        tut = f"{phase}[{','.join(members)}]" + (f" {'/'.join(map(str, outs))}" if outs else "")
        got = f"{got_phase}[{','.join(got_members)}]" + (f" {'/'.join(map(str, got_outs))}" if outs else "")
        out.append([i, tut, got, "yes" if match else "NO"])
    print(table(out, ["step", "tutorial", "mini-sglang", "match"]))
    assert ok and len(res["rows"]) == len(expected), "timeline differs from the tutorial table"


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    quiet_logs()
    path = resolve_model(args.model)

    # ------------------------------------------------------------------ 1
    section("1. A/B/C (concept §3.2 = code §5.2): 2 slots, outputs 2/5/2")
    abc = {0: [("A", 3, 2), ("B", 3, 5)], 1: [("C", 3, 2)]}  # C arrives during step 0
    res = run_timeline(path, abc, max_running_req=2)
    print_timeline(res, extra=[("admission checks", fmt_checks)])
    expected_abc = [
        ("prefill", ["A", "B"], (1, 1, 0)),
        ("decode", ["A", "B"], (2, 2, 0)),
        ("prefill", ["C"], (2, 2, 1)),
        ("decode", ["B", "C"], (2, 3, 2)),
        ("decode", ["B"], (2, 4, 2)),
        ("decode", ["B"], (2, 5, 2)),
    ]
    compare(res, expected_abc)
    r1 = res["rows"][1]
    assert r1["waiting"] == ["C"] and r1["free_slots"] == 0 and r1["phase"] == "decode"
    assert r1["checks"] == [("C", 3 + 2 + 1, r1["reserved"], r1["kv_available"], 0, False)]
    r2 = res["rows"][2]
    assert "B" in r2["running"] and r2["phase"] == "prefill"  # B paused, not gone
    print("-> step 1: C is waiting but free slots = 0, so no prefill batch forms and decode runs.")
    print("-> step 2: B stays in the running set while the batch is C's prefill (prefill-first).")
    abc_tokens = res["tokens"]

    # ------------------------------------------------------------------ 2
    section("2. five requests A..E (concept §5.4): 2 slots, prefill budget 6")
    five = {0: [("A", 3, 4)], 1: [("B", 2, 2)], 2: [("C", 3, 3)], 4: [("D", 4, 2)], 6: [("E", 2, 4)]}
    res5 = run_timeline(path, five, max_running_req=2, prefill_budget=6)
    print_timeline(res5)
    compare(res5, [
        ("prefill", ["A"], None), ("prefill", ["B"], None), ("decode", ["A", "B"], None),
        ("prefill", ["C"], None), ("decode", ["A", "C"], None), ("decode", ["A", "C"], None),
        ("prefill", ["D", "E"], None), ("decode", ["D", "E"], None), ("decode", ["E"], None),
        ("decode", ["E"], None),
    ])
    assert sum(d - c for _, c, d, _ in res5["rows"][6]["members"]) == 6
    print("-> t6: D (4 tokens) + E (2 tokens) use exactly the budget of 6, so they prefill together.")

    # ------------------------------------------------------------------ 3
    section("3. concept ex.3 / code ex.4: what if A generates 1 token? what if C needs 2 chunks?")
    res_a1 = run_timeline(path, {0: [("A", 3, 1), ("B", 3, 5)], 1: [("C", 3, 2)]}, max_running_req=2)
    print("(a) A max_tokens=1")
    print_timeline(res_a1)
    assert res_a1["finish"]["A"] == 0 and short(res_a1["rows"][1]) == ("prefill", ["C"])
    print(f"-> A finishes at the end of its prefill (step 0); C prefills at step {res_a1['first']['C']}"
          f" (vs step {res['first']['C']} before).")
    res_c2 = run_timeline(path, {0: [("A", 3, 2), ("B", 3, 5)], 1: [("C", 10, 2)]},
                          max_running_req=2, prefill_budget=6)
    print("\n(b) prefill_budget=6, C's prompt has 10 tokens -> chunks of 6 and 4")
    print_timeline(res_c2)
    paused = [r["step"] for r in res_c2["rows"] if "B" in r.get("running", []) and "B" not in short(r)[1]]
    print(f"-> B sits in the running set but is not executed at steps {paused}: one step per C chunk.")
    assert paused == [2, 3] and len(res_c2["rows"]) == 7

    # ------------------------------------------------------------------ 4
    section("4. fixed (static) batching vs continuous batching")
    static_abc = run_timeline(path, abc, static=True, max_running_req=2)
    print("A/B/C with a static batch (C may join only after the whole group A,B is done):")
    print_timeline(static_abc)
    assert static_abc["first"]["C"] == 5 and len(static_abc["rows"]) == 7
    assert static_abc["tokens"] == abc_tokens, "batch composition must not change greedy tokens"

    outs = [2, 9, 3, 8, 2, 7, 4, 6]
    workload = {0: [(f"r{i}", 4 + i % 3, o) for i, o in enumerate(outs)]}
    cont = run_timeline(path, workload, max_running_req=4)
    stat = run_timeline(path, workload, static=True, max_running_req=4)
    rows = []
    for name, r in [("static batches of 4", stat), ("continuous, 4 slots", cont)]:
        dec = [len(x["members"]) for x in r["rows"] if x.get("phase") == "decode"]
        rows.append([name, len(r["rows"]), r["stats"].prefill_steps, r["stats"].decode_steps,
                     f"{sum(dec) / len(dec):.2f}", f"{sum(r['first'].values()) / len(outs):.2f}",
                     f"{sum(r['finish'].values()) / len(outs):.2f}"])
    print(f"\n8 requests, output lengths {outs}, all arrive at step 0:")
    print(table(rows, ["mode", "steps", "prefill steps", "decode steps", "mean decode batch",
                       "mean first-token step", "mean finish step"]))
    expect_static = (1 + max(outs[:4]) - 1) + (1 + max(outs[4:]) - 1)
    assert len(stat["rows"]) == expect_static, (len(stat["rows"]), expect_static)
    assert len(cont["rows"]) < len(stat["rows"])
    assert cont["tokens"] == stat["tokens"]
    print(f"-> static = sum over groups of (1 prefill + longest output - 1 decodes) = {expect_static} steps;")
    print("   continuous refills a slot as soon as a request ends, so fewer steps and bigger decode")
    print("   batches. Same tokens either way. (README: bench_batching.py, 16 requests, Qwen3-0.6B:")
    print("   one-at-a-time 14.4 tok/s, mean TTFT 22.93 s vs continuous 49.9 tok/s, 2.01 s.)")

    # ------------------------------------------------------------------ 5
    section("5. chunked prefill: prefill_budget=4, C has a 10-token prompt")
    chunk = {0: [("B", 3, 6)], 1: [("C", 10, 2), ("D", 2, 2)]}
    resc = run_timeline(path, chunk, max_running_req=4, prefill_budget=4)
    print_timeline(resc)
    rws = resc["rows"]
    assert fmt_batch(rws[1]) == "prefill: C[0->4)*" and rws[1]["outputs"]["C"] == 0
    assert fmt_batch(rws[2]) == "prefill: C[4->8)*" and rws[2]["outputs"]["C"] == 0
    assert rws[2]["waiting"] == ["C*", "D"]  # the unfinished chunk is back at the front
    assert rws[2]["running"] == ["B"]  # a ChunkedReq never enters the decode running set
    assert fmt_batch(rws[3]) == "prefill: C[8->10) D[0->2)" and rws[3]["outputs"]["C"] == 1
    assert [r["step"] for r in rws if "B" in short(r)[1]][:2] == [0, 4]
    print("-> steps 1-2: C is a ChunkedReq: no token is sampled/committed (outputs stay 0), its")
    print("   logits are those of an intermediate prompt position, not a prediction of output 1.")
    print("-> waiting at step 2 is 'C* D': chunked_list goes to the front of pending_list.")
    print("-> step 3: C's last 2 tokens leave 2 budget tokens, so D joins the same prefill batch.")
    print("-> B is paused at steps 1-3: prefill-first never interleaves B's decode between chunks.")

    # ------------------------------------------------------------------ 6
    section("6. decode runs while the waiting queue is non-empty: KV capacity blocks admission")
    kv = {0: [("A", 4, 16), ("B", 4, 12)]}
    resk = run_timeline(path, kv, max_running_req=4, max_total_tokens=32, page_size=1)
    print_timeline(resk, extra=[("admission check: est + reserved <= available?", fmt_checks)])
    blocked = [r for r in resk["rows"] if r.get("phase") == "decode" and r["waiting"] == ["B"]]
    assert blocked and all(r["free_slots"] > 0 for r in blocked)
    for r in blocked + [resk["rows"][0]]:  # step 0: B fails behind A's fresh reservation
        _, est, reserved, avail, _, ok = r["checks"][-1]
        assert not ok and est + reserved > avail
    assert resk["rows"][0]["checks"][-1][2] == 4 + 16 + 1  # A's reservation, added in the same step
    print(f"-> {len(blocked)} decode steps ran with B waiting and 3 free slots: est(B)+reserved > available.")
    print("   At step 0 the reservation (21) is A's, added by PrefillAdder._add_one_req a moment earlier.")
    print("   A's reservation (max_device_len - cached_len + page_size) shrinks by 1 per step, but so")
    print("   does the free pool, so B is admitted only after A finishes and frees its pages.")
    r = blocked[0]
    print(f"   e.g. step {r['step']}: this repo reserves {r['reserved']} for A; the reference's"
          f" remain_len + page_size - 1 = {r['ref_reserved']}.")

    # ------------------------------------------------------------------ 7
    section("7. concept ex.5: KV admission inequality via PrefillAdder._try_allocate_one")
    import torch

    from minisgl import SamplingParams
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.prefill import PendingReq, PrefillAdder
    from minisgl.scheduler.table import TableManager

    def real_check(available: int, page_size: int) -> bool:
        page_table = torch.zeros((5, 1024), dtype=torch.int64)
        cm = CacheManager(torch.device("cpu"), available // page_size, page_size, page_table, "naive")
        tm = TableManager(4, page_table)
        adder = PrefillAdder(token_budget=2048, reserved_size=500, cache_manager=cm, table_manager=tm)
        req = PendingReq(uid=1, input_ids=torch.arange(600), sampling_params=SamplingParams(max_tokens=200),
                         output_len=200)
        return adder._try_allocate_one(req) is not None

    rows = []
    for available, ps in [(1200, 1), (1300, 1), (1301, 1), (1312, 16), (1328, 16)]:
        chapter = 600 + 200 + 500 <= available
        repo = real_check(available, ps)
        assert repo == (600 + 200 + ps + 500 <= available)
        rows.append([available, ps, f"{600 + 200 + 500} <= {available}: {chapter}",
                     f"{600 + 200 + ps + 500} <= {available}: {repo}"])
    print("new request: 600 uncached input + 200 max output; others reserve 500")
    print(table(rows, ["available", "page_size", "chapter: extend+output+reserved", "this repo (+page_size)"]))
    assert not real_check(1200, 1)
    print("-> exercise answer: 1300 > 1200, so it waits (both formulas). The extra page_size in this")
    print("   repo covers rounding the last partial page up; it only matters at the boundary.")


if __name__ == "__main__":
    main()
