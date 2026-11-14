"""Part IV Ch2: capture a torch.profiler trace of mini-sglang serving a workload.

The scheduler path is exactly the one the HTTP server uses (``Scheduler.normal_loop``);
only the IO is in-process (``OfflineIO``), so the trace contains no HTTP/ZMQ noise.

Without touching ``python/minisgl`` the script wraps the scheduler's phases in
``torch.profiler.record_function`` ranges, so the timeline shows::

    step[prefill|decode]                one normal_loop iteration
      sched.recv                        drain the inbox
      sched.schedule                    PrefillManager / DecodeManager pick a batch
        sched.prefill_admission         PrefillAdder (radix match, budget checks)
        sched.prepare_batch             allocate pages, positions, out_loc, page-table gather
          attn.prepare_metadata         build masks / kv_indices for the attention backend
          sampler.prepare               sampling tensors
      sched.forward                     token gather + engine.forward_batch + token write-back
        model.forward                   28 decoder layers + LM head (aten ops underneath)
          attn.forward                  per-layer attention (store_kv, gather, SDPA)
        sampler.sample                  argmax / top-k/top-p
      sched.process_results             append tokens, finish/free, cache_req (radix insert)

Outputs (default ``benchmarks/traces/``):
  * ``minisgl_trace.json``   Chrome/Perfetto trace (open in https://ui.perfetto.dev)
  * ``profile_summary.md``   per-phase breakdown computed from the same run

    python benchmarks/profile.py --model /tmp/models/Qwen3-0.6B
    python benchmarks/profile.py --model /tmp/models/Qwen3-0.6B --num-requests 16 --output-len 64
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import os
import random
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from minisgl import SamplingParams
from minisgl.llm import LLM


# ----------------------------------------------------------------------------- instrumentation
def _wrap(obj, attr: str, label: str | None = None, label_fn=None) -> None:
    fn = getattr(obj, attr)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        name = label_fn(*a, **kw) if label_fn else label
        with record_function(name):
            return fn(*a, **kw)

    setattr(obj, attr, wrapper)


class StepTimer:
    """Wall-clock timings per phase (independent of the profiler, low overhead)."""

    def __init__(self) -> None:
        self.cur: dict | None = None
        self.steps: list[dict] = []

    @contextlib.contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cur is not None:
                self.cur[name] = self.cur.get(name, 0.0) + time.perf_counter() - t0


def instrument(llm: LLM, timer: StepTimer) -> None:
    sched = llm.scheduler
    eng = sched.engine

    def timed(obj, attr, name):
        fn = getattr(obj, attr)

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with record_function(name), timer.phase(name):
                return fn(*a, **kw)

        setattr(obj, attr, wrapper)

    # the loop itself: label each iteration by the phase of the batch it ran
    orig_loop = sched.normal_loop

    def loop():
        timer.cur = {}
        t0 = time.perf_counter()
        # messages still in the inbox will become a prefill batch in this iteration
        busy_prefill = sched.prefill_manager.runnable or sched.io.has_pending()
        phase = "prefill" if busy_prefill else "decode"
        with record_function(f"step[{phase}]"):
            orig_loop()
        timer.cur["step"] = time.perf_counter() - t0
        timer.cur["kind"] = timer.cur.get("_kind", phase)
        timer.steps.append(timer.cur)
        timer.cur = None

    sched.normal_loop = loop

    orig_sched = sched._schedule_next_batch

    def schedule():
        with record_function("sched.schedule"), timer.phase("sched.schedule"):
            fi = orig_sched()
        if fi is not None and timer.cur is not None:
            timer.cur["_kind"] = fi.batch.phase
            timer.cur["bs"] = fi.batch.size
            timer.cur["tokens"] = fi.batch.num_tokens
        return fi

    sched._schedule_next_batch = schedule
    timed(sched.io, "receive_msg", "sched.recv")
    timed(sched.prefill_manager, "schedule_next_batch", "sched.prefill_admission")
    timed(sched, "_prepare_batch", "sched.prepare_batch")
    timed(sched.cache_manager, "allocate_paged", "kv.allocate_paged")
    timed(eng.attn_backend, "prepare_metadata", "attn.prepare_metadata")
    timed(eng.sampler, "prepare", "sampler.prepare")
    timed(sched, "_forward", "sched.forward")
    timed(eng.model, "forward", "model.forward")
    _wrap(eng.attn_backend, "forward", "attn.forward")
    timed(eng.sampler, "sample", "sampler.sample")
    timed(sched, "_process_last_data", "sched.process_results")
    timed(sched.cache_manager, "cache_req", "kv.cache_req")


# ----------------------------------------------------------------------------- workload
def make_workload(n: int, shared: int, unique_lo: int, unique_hi: int, seed: int):
    rng = random.Random(seed)
    system = [rng.randrange(100, 20000) for _ in range(shared)]
    return [system + [rng.randrange(100, 20000) for _ in range(rng.randrange(unique_lo, unique_hi))] for _ in range(n)]


# ----------------------------------------------------------------------------- analysis
def summarize(timer: StepTimer, trace_path: str, prof, out_md: str, args, wall: float, outputs) -> str:
    lines = []
    w = lines.append
    steps = [s for s in timer.steps if "_kind" in s]
    idle = len(timer.steps) - len(steps)
    w(f"# mini-sglang profile summary\n")
    w(f"workload: {args.num_requests} requests, shared prefix {args.shared_prefix} tokens + "
      f"{args.unique_min}-{args.unique_max} unique, output {args.output_len} tokens, "
      f"prefill_budget={args.prefill_budget}, cache={args.cache}, device=cpu, threads={torch.get_num_threads()}\n")
    total_out = sum(len(o.token_ids) for o in outputs)
    st = args._prefill_stats
    w(f"prefix cache: {st.prompt_tokens} prompt tokens admitted, {st.cached_tokens} reused from the cache "
      f"({100 * st.cached_tokens / max(1, st.prompt_tokens):.0f}%), {args._sched_stats.prefill_tokens} computed in prefill\n")
    w(f"wall time (profiled run) {wall:.2f} s, {len(steps)} busy steps (+{idle} idle), "
      f"{total_out} output tokens, {total_out / wall:.1f} tok/s under the profiler\n")
    phases = ["sched.recv", "sched.prefill_admission", "kv.allocate_paged", "attn.prepare_metadata",
              "sampler.prepare", "sched.prepare_batch", "sched.schedule", "model.forward", "sampler.sample",
              "sched.forward", "kv.cache_req", "sched.process_results", "step"]
    for kind in ("prefill", "decode"):
        ks = [s for s in steps if s["_kind"] == kind]
        if not ks:
            continue
        tot = {p: sum(s.get(p, 0.0) for s in ks) for p in phases}
        w(f"\n## {kind}: {len(ks)} steps, batch size mean {statistics.mean(s['bs'] for s in ks):.1f}, "
          f"tokens/step mean {statistics.mean(s['tokens'] for s in ks):.0f}\n")
        w("| phase | total ms | per step ms | % of step |")
        w("|---|---:|---:|---:|")
        for p in phases:
            w(f"| {p} | {1e3 * tot[p]:.1f} | {1e3 * tot[p] / len(ks):.2f} | {100 * tot[p] / tot['step']:.1f}% |")
        cpu_overhead = tot["step"] - tot["model.forward"] - tot["sampler.sample"]
        w(f"\nnon-model time (scheduler + bookkeeping) = {1e3 * cpu_overhead / len(ks):.2f} ms/step "
          f"= {100 * cpu_overhead / tot['step']:.1f}% of {kind} step time")
    # decode step time vs batch size
    dec = [s for s in steps if s["_kind"] == "decode"]
    if dec:
        by_bs = collections.defaultdict(list)
        for s in dec:
            by_bs[s["bs"]].append(s)
        w("\n## decode step time by batch size\n")
        w("| batch size | steps | step ms | model.forward ms | non-model ms | attn.prepare_metadata ms |")
        w("|---:|---:|---:|---:|---:|---:|")
        for bs in sorted(by_bs):
            g = by_bs[bs]
            m = lambda k: 1e3 * statistics.mean(x.get(k, 0.0) for x in g)  # noqa: E731
            w(f"| {bs} | {len(g)} | {m('step'):.1f} | {m('model.forward'):.1f} | "
              f"{m('step') - m('model.forward') - m('sampler.sample'):.2f} | {m('attn.prepare_metadata'):.2f} |")
    # top aten ops
    w("\n## top operators by self CPU time (whole run)\n")
    w("| op | calls | self CPU ms | % |")
    w("|---|---:|---:|---:|")
    ka = [e for e in prof.key_averages() if not e.key.startswith(("step[", "sched.", "model.", "attn.", "sampler.", "kv."))]
    total_self = sum(e.self_cpu_time_total for e in ka)
    for e in sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]:
        w(f"| `{e.key}` | {e.count} | {e.self_cpu_time_total / 1e3:.1f} | {100 * e.self_cpu_time_total / total_self:.1f}% |")
    # attention breakdown
    w("\n## attention (`attn.forward`) share\n")
    att = next((e for e in prof.key_averages() if e.key == "attn.forward"), None)
    mf = next((e for e in prof.key_averages() if e.key == "model.forward"), None)
    if att and mf:
        w(f"attn.forward: {att.count} calls, {att.cpu_time_total / 1e3:.0f} ms total = "
          f"{100 * att.cpu_time_total / mf.cpu_time_total:.1f}% of model.forward")
    w(f"\ntrace: `{os.path.relpath(trace_path)}` ({os.path.getsize(trace_path) / 2**20:.1f} MiB)")
    text = "\n".join(lines) + "\n"
    with open(out_md, "w") as f:
        f.write(text)
    return text


def gaps_from_trace(trace_path: str, top: int = 5) -> str:
    """Find the largest gaps between consecutive top-level step ranges on the main thread."""
    with open(trace_path) as f:
        ev = json.load(f)["traceEvents"]
    steps = sorted((e for e in ev if e.get("ph") == "X" and str(e.get("name", "")).startswith("step[")),
                   key=lambda e: e["ts"])
    gaps = [(steps[i + 1]["ts"] - (steps[i]["ts"] + steps[i]["dur"]), i) for i in range(len(steps) - 1)]
    out = [f"\n## gaps between scheduler steps (from the trace)\n",
           f"{len(steps)} step ranges; total gap {sum(g for g, _ in gaps) / 1e3:.1f} ms; "
           f"largest {', '.join(f'{g / 1e3:.2f} ms' for g, _ in sorted(gaps, reverse=True)[:top])}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/models/Qwen3-0.6B")
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--shared-prefix", type=int, default=256)
    ap.add_argument("--unique-min", type=int, default=32)
    ap.add_argument("--unique-max", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--prefill-budget", type=int, default=512)
    ap.add_argument("--cache", default="radix", choices=["radix", "naive"])
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "traces"))
    ap.add_argument("--with-stack", action="store_true", help="record Python stacks (bigger trace)")
    ap.add_argument("--kv-tokens", type=int, default=4096)
    ap.add_argument("--record-shapes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    llm = LLM(args.model, load_tokenizer_=False, device="cpu", max_total_tokens=args.kv_tokens, max_seq_len=2048,
              prefill_budget=args.prefill_budget, cache_type=args.cache, max_running_req=64)
    prompts = make_workload(args.num_requests, args.shared_prefix, args.unique_min, args.unique_max, args.seed)
    sp = SamplingParams(max_tokens=args.output_len, ignore_eos=True)
    # warmup with a disjoint prompt (does not seed the prefix cache with the workload's prefix)
    llm.generate([[7] * 64], SamplingParams(max_tokens=4, ignore_eos=True))

    # reset counters so the summary only covers the profiled workload
    from minisgl.scheduler.prefill import PrefillStats
    from minisgl.scheduler.scheduler import SchedulerStats

    llm.scheduler.prefill_manager.stats = PrefillStats()
    llm.scheduler.stats = SchedulerStats()
    timer = StepTimer()
    instrument(llm, timer)
    trace_path = os.path.join(args.out_dir, "minisgl_trace.json")
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts, record_shapes=args.record_shapes, with_stack=args.with_stack) as prof:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t0
    prof.export_chrome_trace(trace_path)
    args._prefill_stats = llm.scheduler.prefill_manager.stats
    args._sched_stats = llm.scheduler.stats
    text = summarize(timer, trace_path, prof, os.path.join(args.out_dir, "profile_summary.md"), args, wall, outputs)
    text += gaps_from_trace(trace_path)
    with open(os.path.join(args.out_dir, "profile_summary.md"), "w") as f:
        f.write(text)
    print(text)
    llm.shutdown()


if __name__ == "__main__":
    main()
