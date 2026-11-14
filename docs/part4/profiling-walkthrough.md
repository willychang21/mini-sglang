# Profiling walkthrough: reading a real mini-sglang trace

This walkthrough reads the trace produced by [`benchmarks/profile.py`](../../benchmarks/profile.py)
and follows Part IV Ch2's loop: **symptom → hypothesis → trace → proposed change → how to
validate**. Per the brief, nothing here changes `python/minisgl/` (the scheduler and attention
code is owned by other sessions). The optimizations are proposals with expected impact.

## 1. Setup

```bash
python benchmarks/profile.py --model /tmp/models/Qwen3-0.6B
# -> benchmarks/traces/minisgl_trace.json      (Chrome/Perfetto; ~67 MiB)
#    benchmarks/traces/profile_summary.md      (numbers below)
# the committed copy: benchmarks/traces/minisgl_trace.json.gz  -> gunzip, open in https://ui.perfetto.dev
```

- **Machine:** CPU only, 4 threads, torch 2.x CPU build. Measured copy bandwidth is 54 GB/s
  (read+write), so about 27 GB/s read. A 4096² fp32 matmul runs at 0.59 TFLOP/s.
- **Model:** Qwen3-0.6B in fp32 (2.4 GB of linear weights incl. the tied LM head).
- **Workload:** 8 requests submitted at once. Each is a shared 256-token "system prompt"
  plus 32–256 unique tokens (prompts 332–480 tokens), 32 output tokens, greedy,
  `prefill_budget=512`, radix cache on. The run does one warmup on a disjoint prompt and
  resets the counters before profiling.
- **Instrumentation:** the script wraps scheduler phases in `record_function` ranges from the
  outside (monkey-patching), so the timeline shows `step[prefill|decode]` → `sched.schedule` →
  `sched.prepare_batch` → `attn.prepare_metadata` …, `sched.forward` → `model.forward` →
  28 × `attn.forward` → aten ops, `sampler.sample`, and `sched.process_results` → `kv.cache_req`.
  A lightweight wall-clock timer records the same phases per step (the "summary" numbers).
  The profiler inflates absolute times by ~10%, so compare *ratios* inside one run.

## 2. The timeline at a glance

```
t(s)  0         1.37       2.93       4.24  5.00                                 13.0
      |prefill 1 |prefill 2 |prefill 3 |pf 4|d|d|d|d|d|d|d| ... 31 decode steps ...|
       390+122t   358+117    147+173+   6+   each ~256 ms, batch = 8
                  +37        192        188+76
```

The prefill batch contents were confirmed by logging `Batch.reqs` (uid, cached_len, extend_len):

```
prefill 1: (1, 0, 390, Req), (2, 0, 122, ChunkedReq)
prefill 2: (2, 122, 358, Req), (3, 256, 117, Req), (4, 256, 37, ChunkedReq)
prefill 3: (4, 293, 147, Req), (5, 256, 173, Req), (6, 256, 192, ChunkedReq)
prefill 4: (6, 448, 6, Req), (7, 256, 188, Req), (8, 256, 76, Req)
```

Summary table (from `profile_summary.md`, wall-clock timer):

| | steps | batch | tokens/step | step ms | model.forward | non-model (scheduler + bookkeeping) |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 4 | 2.8 | 452 | 1242 | 99.8% | 1.75 ms (0.1%) |
| decode | 31 | 8 | 8 | 256 | 99.0% | 1.75 ms (0.7%) |

- Prefix cache: 3342 prompt tokens admitted, **1536 reused (46%)**, 1806 computed.
- Gaps between steps: **0.8 ms total over 35 steps** (largest 0.04 ms). The loop is fully
  synchronous on CPU, so there is no idle "GPU stream" to find. Section 5 explains why this
  changes on a GPU.

## 3. Where time goes: prefill vs decode

**Prefill is compute-bound.** A 452-token step does ≈ 2 × 0.44 B (non-embedding params) × 452
≈ 0.40 TFLOP of GEMM, plus attention. Inside `model.forward`, attention is 298 ms of
1198 ms, which leaves ~900 ms of GEMM, or ≈ **0.44 TFLOP/s, about 75% of the measured fp32 peak**.
Inside prefill attention, SDPA itself dominates (247 ms/step). The KV gather is small
(26 ms) relative to the O(L²) compute.

**Decode is memory-bound.** Two runs with the same model:

| decode batch | step ms | model.forward ms |
|---:|---:|---:|
| 8 | 256 | 254 |
| 32 (second run: 32 requests, 16 output tokens) | 311 | 306 |

4× the batch costs only 1.2× the time. That is the signature of a step dominated by
reading weights (2.4 GB per step regardless of batch), which is Part III Ch2's
"time per step ≳ weight bytes / bandwidth". Ignoring attention, the GEMMs plus LM head
move 2.4 GB in ~188 ms, about 12.7 GB/s, roughly half the measured read bandwidth.

**Top operators (whole run, self CPU):** `aten::mm` 61%,
`_scaled_dot_product_flash_attention_for_cpu` 17%, **`aten::index` 13%**, everything else ≤2%.

## 4. Finding 1: decode attention spends half its time copying KV

Hypothesis: `aten::index` (13% of the whole run) is the
`flat_k_cache(layer)[md.kv_indices]` gather in `TorchAttnBackend.forward`.

Trace evidence: aggregating the aten ops *inside* `attn.forward` per decode step gives:

```
step[decode]  model.forward 288 ms   attn.forward 101 ms
    attn/aten::index                         46.7 ms
    attn/aten::_scaled_dot_product_flash...  43.7 ms
    attn/aten::index_put_ (store_kv)          2.2 ms
```

The arithmetic agrees. Each decode step gathers the full history of K and V for every request
and every layer: 8 requests × ~416 tokens × 8 heads × 128 × 4 B × 2 × 28 layers ≈ **763 MB per
step**. At the ~16 GB/s these gathers achieve, that is ≈ 48 ms, the 46.7 ms in the trace.
On top of that, the padded SDPA then *reads* the copy again. So **attention is 35% of the decode
forward, and half of it is a copy that real paged kernels never make.**

## 5. Finding 2: scheduler CPU overhead is negligible here and would not be on a GPU

Per step, the non-model time is ~1.75 ms: `sched.prepare_batch` 0.74 ms
(`attn.prepare_metadata` 0.43 ms of it, which has a per-request Python loop that fills
`kv_indices`), `sampler.sample` ~1 ms (argmax over 151,936 logits + `.to("cpu")`),
`sched.process_results` 0.18 ms, `kv.allocate_paged` 0.12 ms. At batch 32 it grows to
2.35 ms. On this CPU that is 0.7% of a 256 ms decode step.

On a GPU the forward shrinks ~100× but the Python does not. Qwen3-0.6B in bf16 reads 1.2 GB
per decode step, so ≈ 0.4 ms of HBM time on an H100, plus kernel-launch costs for ~28 × 15
kernels in eager mode. The same 1.75 ms of scheduler work would then be comparable to or
*larger than* the forward. Because `normal_loop` is strictly sequential
(schedule → forward → `.to("cpu")` sync → process), the GPU would sit idle during it. On a
GPU trace that appears as a gap on the compute stream between steps, which is exactly
what Ch2 §2.2 tells you to look for. This is a *prediction* from the CPU trace, not a
measurement: rerun `benchmarks/profile.py` on CUDA (it adds `ProfilerActivity.CUDA`
automatically) to confirm.

## 6. Finding 3: requests in the same prefill batch recompute a shared prefix

Symptom: 8 requests share 256 tokens, so the best case reuses 7 × 256 = 1792 tokens; we
reused 1536. Trace and batch log: request 2 is admitted in the **same** step as request 1,
before request 1's prefix is inserted into the radix tree (`cache_req` runs in
`_process_last_data`, after the forward). So request 2 matches `cached_len = 0` and computes
the 256-token prefix again, spread over two chunks. That is 256 of 1806 computed prefill
tokens, **14% wasted prefill compute**. With a larger `prefill_budget` the waste grows: in the
A/B lab (default budget 2048, 6 requests) most requests land in the first batch, and radix
only improves throughput by 17%.

## 7. Finding 4: prefill steps stall decode

The four prefill steps (5.0 s) run back to back before the first decode step, and any
decoding request waits ~1.2–1.5 s per prefill step instead of 0.26 s, a **~5× ITL spike**.
This is Part III Ch5's interference and comes from the prefill-first policy in
mini-sglang's `_schedule_next_batch`.

## 8. Proposed optimizations (not implemented)

| # | change | where | expected impact (this workload) | how to validate |
|---|---|---|---|---|
| 1 | **Stop materializing the KV history per layer.** On GPU, use a paged kernel that reads KV in place (FlashInfer/Triton decode). On CPU, (a) build `kv_indices` once per step, as today, but gather *only the new rows* into a persistent per-request contiguous buffer, or (b) for decode, use one gather for K and V together (a single `index_select` on the `[2, …]` buffer) and SDPA without padding by grouping equal-length requests. | `attention/torch_backend.py` (attention owners) | Removes most of the 46.7 ms/step gather, about **−15% decode step time** (−47 of 288 ms in the trace). Larger at longer contexts, because the gather is O(batch × context × layers). | Same `profile.py` run: `attn/aten::index` per decode step → near 0; tests still match HF token for token; `ab_compare.py` on TPOT, 3 repeats. |
| 2 | **In-batch prefix dedup.** When admitting request *j*, if its prefix overlaps an uncached prefix of a request already admitted in this batch by ≥ T tokens (SGLang uses `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32`), defer *j* to the next step so it matches the tree after insertion. | `scheduler/prefill.py::PrefillManager.schedule_next_batch` (scheduler owners) | **−256 of 1806 prefill tokens (−14%)**, about −0.6 s of the 5.0 s prefill phase here. The gain scales with (#requests sharing a prefix that land in one batch) × prefix length. Request 2's TTFT gets slightly *worse* (it waits a step), so check mean and p99 TTFT. | `profile_summary.md` "prefix cache" line → 1792 reused; `ab_compare.py --setting cache_type` gap widens; TTFT mean and p99. |
| 3 | **Take CPU work off the critical path** (GPU deployments): vectorize `prepare_metadata` (no per-request loop), keep sampled tokens on device and defer the `.to("cpu")` until the result is needed, then overlap scheduling of step *i+1* with the forward of step *i* (SGLang's `event_loop_overlap` + `FutureMap`). | `scheduler/scheduler.py`, `attention/torch_backend.py`, `engine/engine.py` | ≤1% on this CPU (overhead is 0.7%). On a GPU, the estimated 1.75 ms of host work per decode step is comparable to the forward, so overlap could cut decode step time by roughly **20–50% at small batch**. This estimate must be confirmed with a CUDA trace. | CUDA trace: the GPU-stream gaps between steps disappear; TPOT at batch 1–8 improves; outputs identical. |

A fourth, policy-level option is decode-aware scheduling (cap prefill tokens while decodes
are running, or mixed prefill+decode batches like SGLang's `--enable-mixed-chunk`). It
trades TTFT for ITL, so it needs both metrics reported.

## 9. Caveats

- CPU timings are not GPU timings. The *structure* (what is compute-bound, what is a copy, what
  is host overhead) transfers; the absolute numbers do not.
- The profiler adds ~10% overhead, so final comparisons should use `labs/part4_ch01_ab_compare.py`
  with the profiler off (Ch2 §2.3, step 3).
- One run, one workload. Before claiming a gain, repeat runs and report the range.
