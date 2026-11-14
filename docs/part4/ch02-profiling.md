# Part IV · Ch2 — Profiling and Trace Analysis

Course text: `part4/Chapter2_Profiling-and-Trace-Analysis.md` (case study: SGLang PR #13587
by BBuf, which removed a separate `sorted_ids.fill_()` kernel from `moe_align_block_size`).
Deliverables for this chapter:

- [`benchmarks/profile.py`](../../benchmarks/profile.py): torch.profiler trace of mini-sglang
  serving a workload, with scheduler phases as named ranges.
- [`profiling-walkthrough.md`](profiling-walkthrough.md): the actual trace read end to end,
  with 4 findings and 3 proposed optimizations.
- [`labs/part4/ab_compare.py`](../../labs/part4/ab_compare.py): the profiler-off before/after
  harness (fixed conditions, repeats, ranges, correctness check).

## Key ideas

- A **benchmark** says *what* is slow (TTFT, TPOT, throughput); a **profile** says *why*.
  Profile to form and test a hypothesis, then benchmark without the profiler to measure.
- In a trace, look for **long** events, **frequent** small events (a 5 µs kernel × 60 layers ×
  every step adds up), and **gaps**. Before calling a stream idle, check the other streams and
  the CPU submission track. CPU block width ≠ GPU kernel time.
- Amdahl: a local 16.7% cut of a segment that is a small slice of the step gives ~1% end to
  end (PR #13587). Always size the target first: what fraction of the step is it?
- Validate both performance (same conditions, repeats, profiler off) and correctness (that
  the required work still happens: the PR kept the padding, it just moved into the kernel).

## SGLang how-to

- Server-side capture: set `SGLANG_TORCH_PROFILER_DIR`, then `POST /start_profile` and
  `/stop_profile`, or `python -m sglang.benchmark.serving ... --profile`. Implementation:
  `srt/managers/scheduler_components/profiler_manager.py::SchedulerProfilerManager`.
  Docs: `docs/docs/developer_guide/benchmark_and_profiling.mdx`.
- Profile one rank and a short window (a few steps); full traces get huge (our 35-step CPU
  run is 67 MiB). Use `record_shapes` only when needed; in this session it OOM-killed a 15 GB
  machine on the full workload.

## Exercises — worked answers

**1. Higher TTFT: hypotheses, and the trace evidence that separates them.**
TTFT = queueing + (tokenization / IPC) + prefill time (+ KV transfer in PD).
- *Queueing* (the request waited for admission): the gap between arrival and the first
  scheduler step that includes it; waiting-queue length; admission rejected for KV or budget
  (in our code, `PrefillAdder` returning `None` because `estimated_len + reserved_size >
  available_size`). The trace shows many steps between arrival and the request's first prefill.
- *Slow prefill* (long prompts, a cache miss, a bad backend): long `step[prefill]` ranges
  dominated by GEMM/attention kernels. Check the prefix-cache hit (our summary's
  "reused" line, SGLang's `#cached-token` log).
- *Chunking*: one long prompt split across many steps. Count the request's chunked steps.
- *Interference*: prefill steps shared with or delayed by other prefills (our Finding 4).
- *Front-end*: tokenizer or HTTP time before the scheduler sees the request (the tokenizer
  process track; the server's queue-time metrics).

**2. Is a gap on one GPU stream an idle GPU? Why doesn't a big kernel speedup give a proportional e2e gain?**
No. Other streams (comm, copy, a second compute stream) may be busy during that gap.
Check them all, and the CPU track: a gap with the CPU busy in Python means launch- or
scheduler-bound, while a gap waiting on an event means a dependency (a transfer, an
all-reduce). End-to-end time is the critical path through all work, so a kernel's speedup
only helps by its share of the path. If it overlapped other work it may not help at all.
PR #13587 cut one segment 16.7% (7.85 → 6.54 µs), but that segment is a tiny slice of a
DeepSeek-R1 step dominated by attention, expert GEMMs and communication. Hence ≈ +1% throughput
at concurrency 16–32 and nothing measurable at concurrency 1. In our own trace, making all
of `sched.*` free would gain ≤ 0.8% on CPU (Finding 2). Size the target first.

**3. Design a credible before/after comparison.**
- *Fix*: hardware (same GPU model and count, same machine if possible), driver, CUDA, and
  library versions, the commit except for the change, model and checkpoint, dtype and
  quantization, all server flags, the request set (seeded dataset, input/output lengths,
  `--random-range-ratio`), concurrency and request rate, warmup count, and cache state
  (fresh server or flushed cache).
- *Profiler*: enable it only in separate diagnostic runs, to locate the bottleneck before
  the change and to confirm afterwards that the targeted event shrank or vanished. Capture
  the same rank, same phase and a short window in both. Disable it for the numbers you report.
- *Performance*: ≥3 repeats per side, report mean and range (or CI), several
  concurrency levels, and the metric that matches the claim (TTFT for prefill changes,
  TPOT/ITL for decode, throughput for batching).
- *Correctness*: identical greedy outputs where the change should be bit-neutral; otherwise
  an accuracy eval (GSM8K, or something harder and more relevant) on both sides with the
  same settings and repeats. Plus unit tests for the changed function.
- *Report*: the command lines, the environment, the tables, the trace screenshots, and what
  remains unverified. `labs/part4/ab_compare.py` implements this protocol for mini-sglang.
