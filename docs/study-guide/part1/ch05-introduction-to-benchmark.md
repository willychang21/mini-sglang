# Part I · Chapter 5 — Introduction to Benchmark

> Tutorial: `part1/Chapter5_Introduction_to_Benchmark.md` · Lab: `labs/part1_ch05_metrics.py` ·
> Related code: `python/minisgl/llm/llm.py` (`GenerateOutput.ttft`, `.token_times`),
> `benchmarks/bench_serving.py`, `benchmarks/bench_batching.py`

## Key ideas

- **Two kinds of benchmark, never to be mixed.** A *model capability* benchmark (MMLU,
  GSM8K, HumanEval, SWE-bench) asks "does it answer well?". An *inference workload*
  benchmark asks "does the service run fast and reliably?" (TTFT, TPOT, throughput,
  P99). In this course, "benchmark" means the second kind.
- **Latency metrics for one request** (timeline: submit → first token → … → last
  token):
  - **TTFT** = time of first token − submit time. It includes **queueing + prefill**.
  - **ITL** = gap between two consecutive output tokens. There are `n_out − 1` values
    per request, and they show jitter.
  - **TPOT** = `(E2E − TTFT) / (n_out − 1)`. It is the mean ITL of one request: one
    number per request.
  - **E2E** = time of last token − submit time = TTFT + (n_out − 1) × TPOT.

  Example from the chapter: TTFT 250 ms, TPOT 25 ms, 200 output tokens give
  E2E = 250 + 199 × 25 = 5,225 ms. Prefill is under 5% of that, so long outputs are
  decode-dominated. 25 ms/token is 40 tok/s; 100 ms/token (10 tok/s) is the usual
  floor for comfortable reading.
- **Throughput** = output tokens (or requests) per second over the whole run.
  **Goodput** = throughput counting only requests that meet the SLO (for example
  TTFT ≤ 500 ms). Raising concurrency increases throughput until it saturates, while
  TTFT keeps rising. The useful operating point is just before latency runs away.
- **Distributions, not means.** Report P50/P90/P99 (SGLang's tools print mean,
  median, P90, P95, P99). Tail latency comes from queueing, interference and
  stragglers. A mean mixes the fast majority with the slow tail and describes neither
  group.
- **Designing a benchmark:** pick the target and a matching workload (sharegpt for
  chat, random for ceilings only, generated-shared-prefix for prefix caching,
  agentic-trace for agents). Fix concurrency or request rate, hardware, model,
  precision and quantization, and the output-length distribution. Warm up first, and
  fix the random seed.
- **SGLang's four tools** cover different layers: `bench_serving` (HTTP + scheduler;
  closest to production), `bench_one_batch_server` (one batch through HTTP),
  `bench_offline_throughput` (scheduler, no HTTP), `bench_one_batch` (kernels only;
  never quote it as online latency).

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| Per-token timestamps (offline) | `python/minisgl/llm/llm.py:GenerateOutput` (`ttft`, `token_times`), `LLM.generate` | seconds since the batch was submitted; one timestamp per scheduler iteration |
| TTFT/TPOT/E2E over HTTP | `benchmarks/bench_serving.py:one_request` | streaming SSE; TPOT = `(end − ttft) / (max_tokens − 1)`, the same convention as the lab |
| Concurrency sweep, P99 TTFT, throughput | `benchmarks/bench_serving.py:run_level` | fixed concurrency through an `asyncio.Semaphore` (like `--max-concurrency`) |
| Offline throughput (like `bench_offline_throughput`) | `benchmarks/bench_batching.py` | uses `LLM.generate`, no HTTP |
| Where queueing comes from | `python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one` | waits for a free row (`max_running_req`) and KV capacity; FCFS |
| Why ITL jitters | `python/minisgl/scheduler/scheduler.py:Scheduler._schedule_next_batch` | prefill-first: a new arrival's prefill step pauses every running decode |
| Step counters | `python/minisgl/scheduler/scheduler.py:SchedulerStats` | prefill/decode steps and tokens, batch sizes |
| Warm-up | `benchmarks/bench_serving.py` sweeps levels in order; the lab warms up explicitly | Ch5 pitfall 3 |

README's concurrency table (`bench_serving.py`, CPU) is the latency/throughput
trade-off of Sec. 3.2. Going from concurrency 1 to 16: mean TTFT 0.21 → 1.34 s,
TPOT 63 → 207 ms, throughput 14.7 → 65.9 tok/s.

## Common misconceptions

**Misconception:** TPOT and ITL are two names for the same number.
**Actually:** TPOT is one average per request. ITL is one value per gap. A request can
have a fine TPOT and still stall visibly. In the lab run shown below, request 12 has
TPOT 2.44 ms but a max ITL of 5.85 ms, because another request's prefill ran in
between.

**Misconception:** TTFT is the prefill time.
**Actually:** It is queueing + prefill (+ tokenization and network for HTTP). In the
lab, queued requests have TTFT 41–61 ms while prefill takes about 8 ms.

**Misconception:** Higher throughput means a better configuration.
**Actually:** Only if the latency SLO still holds. Going from scenario A to B in the
lab cuts throughput (5120 → 3954 tok/s), and goodput drops further, because 4 of 16
requests miss the TTFT SLO.

**Misconception:** The mean summarizes the latency distribution.
**Actually:** In the lab's scenario B, TTFT has P50 7.7 ms, mean 18.7 ms and P99 60.4 ms.
No request had a TTFT anywhere near 18.7 ms.

**Misconception:** All percentile implementations agree.
**Actually:** With few samples they do not. The lab interpolates linearly (like
`np.percentile`). `benchmarks/bench_serving.py` picks `xs[int(0.99·n)]`, which with 16
requests is simply the maximum. Report the sample count, and use enough requests for
P99 to mean something (at least a few hundred).

## Lab

```bash
python labs/part1_ch05_metrics.py          # ~4 s on tiny; --model real, --num-requests, --slots
```

The lab runs the same 16 requests (prompts 8–120 tokens, 16–48 output tokens,
`ignore_eos`) twice through `LLM.generate`. Scenario A has 16 running slots. Scenario B
has `max_running_req=12`, so 4 requests have to queue. Look for the following: TPOT
equals the mean of ITL for every request (asserted); TTFT is identical for all
requests in A; and in B the queued requests produce a P99 far above P50, with
2-step ITL gaps.

```text
== 2. scenario A: 16 requests, 16 running slots
scheduler steps: 1 prefill + 46 decode
|           TTFT ms | 11.0 | 11.0 | 11.0 | 11.0 | 11.0 |
|           TPOT ms |  2.1 |  2.1 |  2.3 |  2.3 |  2.3 |
wall 99 ms | output 5120 tok/s | 161.6 req/s | goodput (TTFT <= 22.0 ms = 2 x P50 of A) 161.6 req/s (16/16 requests)

== 3. scenario B: 16 requests, 12 running slots (4 must queue)
scheduler steps: 5 prefill + 62 decode
|            metric | mean |  P50 |   P90 |   P99 |   max |
|           TTFT ms | 18.7 |  7.7 |  51.7 |  60.4 |  61.2 |
|           TPOT ms |  2.2 |  2.2 |   2.4 |   2.4 |   2.4 |
| ITL ms (all gaps) |  2.1 |  1.9 |   3.3 |   5.9 |   5.9 |
wall 128 ms | output 3954 tok/s | 124.8 req/s | goodput (TTFT <= 22.0 ms = 2 x P50 of A) 93.6 req/s (12/16 requests)
| req | prompt | n_out | TTFT step | TTFT ms | TPOT ms | max ITL ms | E2E ms |
|   8 |    109 |    17 |         0 |     7.7 |    1.99 |       2.32 |   39.5 |
|  12 |     95 |    16 |        17 |    41.6 |    2.44 |       5.85 |   78.2 |
|  15 |      8 |    41 |        26 |    61.2 |    1.68 |       4.01 |  128.2 |
TTFT: P50 7.7 ms, mean 18.7 ms, P99 60.4 ms (P99/P50 = 7.8x).
```

Why B looks like this: admission is FCFS and limited to 12 rows
(`PrefillAdder._try_allocate_one`). Request 12 can start only when the shortest
admitted request (request 8, 17 tokens) finishes, so its first token arrives at
step 17. Each later admission is its own prefill step (prefill-first). Running
requests produce nothing in that step, which shows up as 2-step ITL gaps (44 of
them). Absolute milliseconds depend on the machine. The step counts and assertions
do not.

## Exercises

**1. What do TTFT, TPOT and ITL measure? Why does optimizing TPOT help long-output
workloads more than optimizing TTFT?**

TTFT is the wait until the first token (queueing + prefill), so it is the user's
"is anything happening?" latency. TPOT is the average time per output token after the
first, which is the decode rate. ITL is each individual gap and shows stalls. Since
E2E = TTFT + (n_out − 1)·TPOT, TPOT is multiplied by the output length and TTFT is
paid once. With 200 output tokens (TTFT 250 ms, TPOT 25 ms), decode is 4,975 of
5,225 ms (95%). Halving TPOT saves 2.5 s; halving TTFT saves 0.125 s. Prefill is a
single compute-bound pass, while decode is n memory-bound steps (Ch2), so long
outputs are dominated by decode.

**2. What does it mean when the mean and P99 differ by an order of magnitude? Why
watch P99 for stability?**

The distribution has a heavy tail: most requests are fast, but a few are much slower
(they queued behind a full batch, waited for KV memory, were preempted, shared a step
with a huge prefill, or hit GC or network hiccups). The mean is dragged up by the tail
but stays far below it, so it describes neither the typical request nor the bad
ones. P99 is the latency that 1 in 100 requests exceeds. At production volume that is
many users every minute. With fan-out it is worse: an agent that makes 10 sequential
calls hits at least one P99 event with probability 1 − 0.99¹⁰ ≈ 9.6%. SLOs are
written against tails because the tail is what users notice and what breaks timeouts.
The lab shows the mechanism: 4 of 16 requests waited for a slot, giving P99/P50 = 7.8×.

**3. Chat users report: "no response for a long time after sending, but fast once
output starts". Which metric is abnormal, what should be optimized?**

**TTFT** is high and TPOT/ITL are fine. Split TTFT into queueing time and prefill time
first.
- *Queueing* (requests wait for a running slot or for KV capacity, as in the lab's
  scenario B): add capacity or replicas, lower the admitted concurrency per instance,
  free KV (smaller `max_tokens` reservations, KV quantization), use a better
  scheduling policy.
- *Prefill* (long prompts: system prompt + history + RAG context): enable prefix
  caching so the shared system prompt and history are not recomputed every turn
  (RadixAttention, Part II Ch8), use chunked prefill so long prompts do not
  monopolize steps, trim or summarize context, or separate prefill from decode
  (PD disaggregation, Part III Ch5).

**4. Why is a direct token/s comparison across models unreliable?**

Tokens are not a common unit. Different tokenizers split the same text into
different numbers of tokens: the chapter's example is 1,000 Chinese characters →
1,000 tokens for model A, 1,500 for model B. B can report 1.5× the tokens/s while
delivering the same text at the same speed. A larger vocabulary also makes each token
more expensive (bigger LM head). Compare on the same text (characters/words per
second, or requests/s and latency on an identical request set), or retokenize all
outputs with one common tokenizer. Input/output length distributions also have to
match, because they change the prefill/decode mix.

**5. Which benchmark for a math-reasoning LLM and for a speech-synthesis model? Why
can't they share a dataset?**

- *Math-reasoning LLM.* Quality: a capability benchmark with checkable answers (GSM8K,
  MATH, AIME-style problems). Serving: a workload benchmark with reasoning-length
  outputs (short prompts, thousands of output tokens), so TPOT, E2E and decode
  throughput dominate.
- *Speech synthesis (TTS).* Quality: speech metrics (MOS/naturalness, intelligibility
  via ASR word error rate, speaker similarity) on a TTS test set. Serving: time to
  first audio chunk and real-time factor (seconds of audio per second of compute)
  under streaming load, which matters for real-time speech (TTFT and tail latency in
  the chapter's table).

They cannot share a dataset because the evaluation target must match the data: input
modality, output type and correctness criterion all differ. Math problems with
numeric answers cannot score audio, and text-to-speech prompts cannot test reasoning.
The request-length profile also differs, so performance numbers would not transfer
either ("measuring coding ability with conversation data produces meaningless
results").

**6. Considerations for a reproducible, comparable benchmark.**

- *Target and workload:* say whether it is capability or serving, and use a
  dataset that matches production (input/output length distributions, shared
  prefixes, multi-turn); record the dataset, seed and number of requests.
- *Load:* fixed `--max-concurrency` or `--request-rate` (and arrival process); sweep
  several levels rather than reporting one.
- *Environment:* GPU model and count, interconnect (TP/NVLink/RDMA), driver/CUDA,
  engine version or commit, model and revision, precision/quantization, and all
  server flags (memory fraction, context length, chunked prefill, cache settings).
- *Warm-up:* discard initial requests (CUDA init, graph capture, allocator growth).
  The lab's cold first request took 5.3 ms against 1.6 ms warm.
- *Output length:* fix it (`ignore_eos`) or fix its distribution; compare only runs
  with the same input/output lengths.
- *Randomness:* fix the sampling seed (top-p kernels can take different paths), or
  use greedy decoding; repeat runs and report the variance.
- *Metrics:* TTFT, TPOT, ITL and E2E as mean **and** P50/P90/P99; output and request
  throughput; goodput against a stated SLO; and how each is computed (e.g. the TPOT
  convention, the percentile method).
- *Comparison hygiene:* change one variable at a time, and publish the exact command
  so others can rerun it (Part 0.1, Part IV Ch3).
