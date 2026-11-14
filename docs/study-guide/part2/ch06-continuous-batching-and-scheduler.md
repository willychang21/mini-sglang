# Part II · Chapter 6 — Continuous Batching and the Scheduler

> Tutorial: `part2/Chapter6_Continuous-Batching-and-the-Scheduler.md`, `part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md` · Lab: `labs/part2_ch06_batching_timeline.py` · Bench: `benchmarks/bench_batching.py`

## Limitation → mechanism → validation

| Limitation (Ch5 version) | Mechanism | Metric | Validated here by |
|---|---|---|---|
| Requests run one at a time, or in fixed groups where a finished request's place stays empty until the whole group ends | Continuous batching: `normal_loop` re-chooses the batch every step; `PrefillAdder` admits by slots / token budget / KV reservation; chunked prefill; `DecodeManager` running set | throughput, TTFT | **test**: `tests/test_scheduler.py::test_continuous_batching_matches_hf` (tokens == HuggingFace for every slot/budget/KV setting, incl. `max_running_req=3, prefill_budget=16`), `test_abort_frees_resources`, `test_eos_stops_generation` · **bench**: `benchmarks/bench_batching.py` · **lab**: all sections |

README numbers (`bench_batching.py`, 16 requests, Qwen3-0.6B CPU, outputs identical in both modes):

| mode | wall s | output tok/s | mean TTFT s |
|---|---:|---:|---:|
| one request at a time (`max_running_req=1`) | 46.5 | 14.4 | 22.93 |
| continuous batching | 13.4 | 49.9 | 2.01 |

The price shows up in `bench_serving.py`: mean TPOT rises from 63 ms (concurrency 1) to 207 ms
(concurrency 16). Continuous batching trades per-token latency for throughput and TTFT.

## Key ideas

* **Batching** runs several independent requests through one forward pass: tokens are packed
  (`[N, d_model]`, no padding of prompts), each request keeps its own KV and mask. It amortizes
  weight reads, which is what makes small-batch decode cheap per token.
* **Fixed batching fails on unequal lengths.** Output lengths are unknown (`max_tokens` is only an
  upper bound), so a group runs as long as its longest member; finished members leave holes that
  new arrivals cannot fill.
* **Continuous batching = re-decide membership between forward passes.** It does not insert work
  into a running kernel and does not remove the per-request autoregressive dependency; it only
  lets the *next* batch differ from the last one.
* **Three different sets.** Waiting queue (`PrefillManager.pending_list`, `PendingReq`), decode
  running set (`DecodeManager.running_reqs`), and the current `Batch`. A request can be running
  without being in the current batch (paused by a prefill step).
* **Three different limits** checked by `PrefillAdder`: request slots (table rows), the per-step
  prefill token budget (compute), and the KV reservation (memory, in tokens). Any one can block
  admission, so a non-empty queue does not imply a prefill step.
* **KV reservation is pessimistic:** uncached prompt + *maximum* output (+ a page in this repo)
  for the new request, plus what running requests may still need. Pages are still allocated
  lazily; the reservation only guarantees that admitted requests can never run out mid-generation.
* **Prefill-first policy:** `prefill_manager.schedule_next_batch(...) or
  decode_manager.schedule_next_batch()`. Good for TTFT, bad for the inter-token latency of running
  requests; a step is either all-prefill or all-decode in this version.
* **Chunked prefill** caps one step's prefill tokens. An intermediate chunk (`ChunkedReq`) must not
  sample: its last-position logits predict the next *prompt* token. It returns to the *front* of
  the queue, and with prefill-first, running requests wait for all chunks.
* **FCFS with head-of-line blocking:** admission stops at the first request that does not fit;
  smaller requests behind it do not jump ahead.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| one scheduler iteration | `python/minisgl/scheduler/scheduler.py:Scheduler.normal_loop` | receive (blocking only when idle) → schedule → forward → process |
| offline driver | `Scheduler.run_until_idle`, `python/minisgl/llm/llm.py:LLM.generate` | `LLM.generate` calls `normal_loop` until all uids finish |
| enqueue / reject | `Scheduler._process_one_msg` (`UserMsg` branch) | clips `output_len` to `max_seq_len` and KV capacity; errors for empty/too long |
| prefill-first policy | `Scheduler._schedule_next_batch` | `or` falls back to decode |
| waiting queue | `python/minisgl/scheduler/prefill.py:PrefillManager.pending_list`, `add_one_req`, `PendingReq` | `PendingReq.chunked_req` holds chunk progress |
| form a prefill batch | `PrefillManager.schedule_next_batch` | FCFS `break`; `pending_list = chunked_list + pending_list[len(reqs):]` |
| admission checks | `PrefillAdder.try_add_one`, `_try_allocate_one` | budget > 0; free row; `estimated_len + reserved_size <= available_size`, checked again after `lock` |
| chunking + reservation growth | `PrefillAdder._add_one_req` | `chunk_size = min(token_budget, remain_len)`; `reserved_size += remain_len + output_len + page_size` |
| request slots | `python/minisgl/scheduler/table.py:TableManager.available_size`, `allocate`, `free` | one row of `page_table` + `token_pool` per request |
| KV capacity | `python/minisgl/scheduler/cache.py:CacheManager.available_size` | free pages × page_size + evictable radix tokens |
| running requests' reservation | `python/minisgl/scheduler/decode.py:DecodeManager.inflight_tokens` | Σ (`max_device_len - cached_len + page_size`) |
| running set | `DecodeManager.running_reqs`, `filter_reqs`, `remove_req`, `schedule_next_batch` | decode batch = every running request, sorted by uid |
| intermediate chunk | `python/minisgl/core.py:ChunkedReq` | `can_decode` is always `False` |
| length stop | `Req.can_decode`, `Req.remain_len` | `max_device_len = prompt + output_len` |
| batch prep | `Scheduler._prepare_batch` | `output_rows` excludes `ChunkedReq`; updates `SchedulerStats` |
| forward + running-set update | `Scheduler._forward` | writes sampled tokens to `token_pool`, then `decode_manager.filter_reqs` |
| results, EOS, completion | `Scheduler._process_last_data` | skips `ChunkedReq`; EOS from `eos_token_ids` unless `ignore_eos`; `cache_req(finished=False)` after a prefill |
| cleanup | `Scheduler._free_req_resources` | `table_manager.free` + `cache_manager.cache_req(finished=True)` |
| cancellation | `_process_one_msg` (`AbortBackendMsg`), `PrefillManager.abort_req`, `DecodeManager.abort_req` | a waiting request without resources is just removed |
| counters | `scheduler/scheduler.py:SchedulerStats` | prefill/decode steps and tokens, batch sizes |

## This repo vs. the tutorial's reference code

Checked against the reference files at commit 9a91cfa (`scheduler/{scheduler,prefill,decode,cache}.py`).

* **KV reservation adds a page.** Reference: `estimated_len = extend_len + output_len`,
  `reserved_size += remain_len + output_len`, `inflight_tokens = Σ remain_len + (page_size - 1) ·
  n_running`. Here: `+ page_size` in the first two, and
  `inflight_tokens = Σ (max_device_len - cached_len + page_size)`. Between decode steps
  `device_len = cached_len + 1`, so this repo reserves `remain_len + 1 + page_size` per running
  request, 2 tokens more than the reference. Both are upper bounds (a request writes K/V only for
  positions `< max_device_len - 1`); this one is slightly more conservative. Lab sections 6-7 show
  the effect (`17` vs `15` for request A; `1301 <= 1300` fails where the chapter's `1300 <= 1300`
  passes).
* **Message admission** also clips `max_tokens` to the KV capacity (`num_pages·page_size − 2·page_size −
  input_len`) and replies with an error (instead of only logging) for empty or too-long prompts.
  The capacity clip guarantees that a request alone always fits, so the queue cannot wedge.
* **EOS** uses the set `ModelConfig.eos_token_ids` (from `config.json` / `generation_config.json`),
  not the tokenizer's single `eos_token_id`; replies carry `finish_reason`
  (`length` / `stop` / `abort` / `error`).
* **No overlap loop**, so no `finished_reqs` duplicate-cleanup guard and no
  `engine.stream` handling. `normal_loop` is the only loop.
* `ChunkedReq` only overrides `can_decode`; the reference also makes `append_host` raise. Both
  skip chunked requests in `_process_last_data`.
* `PrefillManager.abort_req` returns `(found, req_with_resources)`, so aborting a request that is
  still waiting produces an `abort` reply too.
* The config field is `prefill_budget` (reference: `max_extend_tokens`).
* The reference calls `cache_manager.check_integrity()` when idle (`run_when_idle`); this repo
  only calls it from tests and labs.

## Common misconceptions

**Misconception:** continuous batching mixes new prefills and running decodes in every batch.
**Actually:** in this version each step is either a prefill batch or a decode batch. "Continuous"
means membership changes between steps (lab section 1: `prefill[C]` at step 2, `decode[B, C]` at step 3).

**Misconception:** if the waiting queue is non-empty, the next step is a prefill.
**Actually:** only if `PrefillAdder` admits at least one request. A full request table (lab section
1, step 1: `C: no free slot`) or the KV reservation (lab section 6: 15 decode steps with 3 free
slots) makes `schedule_next_batch` return `None`, so decode runs.

**Misconception:** a running request advances every step.
**Actually:** prefill-first pauses it during every prefill step (B at step 2 of A/B/C; B at steps
1-3 while C's three chunks prefill in lab section 5).

**Misconception:** chunked prefill keeps running requests responsive.
**Actually:** only if the policy interleaves decode between chunks. Here the unfinished chunk goes
back to the front of `pending_list` and prefill has priority, so chunks run back to back.

**Misconception:** the KV check compares actual memory in bytes.
**Actually:** it compares token counts, and it assumes every request generates its full
`max_tokens`. It is an admission bound; pages are allocated as tokens are computed.

**Misconception:** releasing a request slot frees its KV.
**Actually:** `TableManager.free` returns only the row. KV goes through `cache_req(finished=True)`:
with the radix cache the computed prefix stays in the tree (evictable), only duplicates and the
unaligned tail go back to `free_slots`.

**Misconception:** admission picks whatever fits.
**Actually:** FCFS with `break`: the first request that does not fit blocks everyone behind it.

## Lab

```bash
python labs/part2_ch06_batching_timeline.py            # tiny model, ~4-8 s
python labs/part2_ch06_batching_timeline.py --model real   # ~60 s, same timelines
```

The lab drives `Scheduler.normal_loop` by hand, injecting each request at the step where the
tutorial says it arrives ("arrives during step k" = received at the start of step k+1), and wraps
`Scheduler._schedule_next_batch` and `PrefillAdder._try_allocate_one` to log the queue, the
running set, free slots, the batch and every admission check. `*` marks a `ChunkedReq`. Every
timeline is asserted against the tutorial's table.

```text
== 1. A/B/C (concept §3.2 = code §5.2): 2 slots, outputs 2/5/2
| step | waiting | running | free slots | batch: req[cached->device) *=chunked | outputs A/B/C | events | admission checks                    |
|    0 |     A B |       - |          2 |             prefill: A[0->3) B[0->3) |         1/1/0 |        | A: 6+0=6 <= 2048; B: 9+6=15 <= 2048 |
|    1 |       C |     A B |          0 |              decode: A[3->4) B[3->4) |         2/2/0 | A done |                     C: no free slot |
|    2 |       C |       B |          1 |                     prefill: C[0->3) |         2/2/1 |        |                   C: 6+5=11 <= 2044 |
|    3 |       - |     B C |          0 |              decode: B[4->5) C[3->4) |         2/3/2 | C done |                                     |
|    4 |       - |       B |          1 |                      decode: B[5->6) |         2/4/2 |        |                                     |
|    5 |       - |       B |          1 |                      decode: B[6->7) |         2/5/2 | B done |                                     |
| step |           tutorial |        mini-sglang | match |
|    0 | prefill[A,B] 1/1/0 | prefill[A,B] 1/1/0 |   yes |
...   (all 6 rows match; section 2 matches all 10 rows of the A..E example)
```

mini-sglang's prefill-first policy reproduces both tutorial tables exactly (the tables were derived
from this policy). Other things to look for:

* Section 4: static batching of A/B/C takes 7 steps (C's first token at step 5 instead of 2);
  for 8 requests with outputs `[2, 9, 3, 8, 2, 7, 4, 6]` and 4 slots, static = 16 steps (mean
  decode batch 2.36), continuous = 14 steps (3.30), mean first-token step 4.50 → 2.25, same tokens.
* Section 5: `C[0->4)*`, `C[4->8)*` produce no token; at step 2 the queue is `C* D`.
* Section 6: `B: 17+17=34 > 28` … for 15 steps while 3 slots are free.

## Exercises

### Concept chapter

**1. A finishes early in a fixed batch while C waits. Why is that wasted even if A's own
computation is removed?** Removing A only saves A's work; the batch slot it held stays empty, so B
decodes alone (batch size 1) for the remaining steps while C sits idle. Batching's benefit (one
weight read serving many requests) is lost, and C's TTFT grows by B's whole remaining length. Lab
section 4: static A/B/C runs `decode[B]` at steps 2-4 with C waiting, and C's first token comes at
step 5 instead of step 2.

**2. Why change membership between model steps, and does it remove the autoregressive
dependency?** A forward pass is a fixed computation over fixed tensors (`input_ids`, `positions`,
`out_loc`, attention metadata are built in `_prepare_batch` before the pass), so the only safe
moment to add/remove requests is between passes. It does not remove the dependency: a request
still needs token *t* before computing *t+1*; only *which other requests* share the next pass
changes.

**3. Under §3.2's assumptions, when could C first prefill if A generated only one token? If C
needed two prefill steps?** (a) A's single token comes from its prefill, so A finishes at step 0
and C prefills at step 1 (lab section 3a: `prefill: C[0->3)` at step 1, 6 steps total). (b) With a
budget that splits C into chunks of 6 and 4, C prefills at steps 2 and 3 and B, still in the
running set, is not executed in either step (lab section 3b: "not executed at steps [2, 3]", 7
steps total); B's output is delayed by one extra step.

**4. Only C's prefill runs while B stays in the running set: contradiction?** No. The running set
records requests that still need decode; the batch is what runs this step. Prefill-first chose a
prefill batch, so B is paused, not removed (lab section 1, step 2: running `B`, batch
`prefill: C`). B's state (`cached_len`, its page-table row, its KV) is untouched.

**5. 600 uncached input, 200 max output, other reservations 500, available 1200: does it pass?**
No: 600 + 200 + 500 = 1300 > 1200. This repo evaluates `extend_len + output_len + page_size +
reserved_size <= available_size` in `PrefillAdder._try_allocate_one`; with page_size 1 that is
1301 > 1200, also rejected. Lab section 7 calls the real method:

```text
| available | page_size | chapter: extend+output+reserved | this repo (+page_size) |
|      1200 |         1 |             1300 <= 1200: False |    1301 <= 1200: False |
|      1300 |         1 |              1300 <= 1300: True |    1301 <= 1300: False |
```

**6. Why must intermediate chunks not submit tokens, and why does chunking alone not guarantee
timely decode?** The logits at the end of an intermediate chunk predict the next *prompt* token,
which is already known; sampling there would append a bogus token and also make the request look
decodable. `ChunkedReq.can_decode` is `False`, `_prepare_batch` leaves it out of `output_rows`,
`_process_last_data` skips it, and `filter_reqs` keeps it out of the running set. Chunking only
creates scheduling points; with prefill-first and the chunk re-queued at the front, the scheduler
picks the next chunk every time (lab section 5: B is paused at steps 1-3 while C's chunks run).

**7. Throughput up, but users see more pauses: what to examine?** Per-request TPOT / ITL
distribution (p50/p90/p99, not just the mean), TTFT, per-user tokens/s, together with the phase
sequence: how many prefill steps ran between two decode steps of a running request, the prefill
budget, and the arrival rate. Here: `SchedulerStats.prefill_steps/decode_steps/batch_sizes`,
`GenerateOutput.token_times` gaps, and `bench_serving.py` (TPOT 63 → 207 ms as concurrency rises).

**8. A request emits EOS early: what state changes, and why may its KV remain?**
`_process_last_data` sees the token in `eos_token_ids` (unless `ignore_eos`), sends a
`DetokenizeMsg(finished=True, finish_reason="stop")`, calls `decode_manager.remove_req` and
`_free_req_resources`: the table row returns to `TableManager`, and `cache_req(finished=True)`
inserts `[0, cached_len)` into the prefix cache. With the radix cache those pages stay in the tree
(evictable) for future prefix hits; only duplicate pages and the unaligned tail are freed. Even
freed pages keep their values until reused (`tests/test_scheduler.py::test_eos_stops_generation`).

**9. Same ~85 % GPU-Util, different throughput: why, what else to measure?** GPU-Util is the
fraction of time *any* kernel runs, not how much of the GPU it uses. A batch-1 decode kernel and a
batch-32 decode kernel can both keep the GPU "busy" while the latter does ~32x the useful work per
weight read. Measure output tokens/s, tokens per forward pass (batch size), achieved FLOP/s and
memory bandwidth, SM occupancy (Nsight), plus TTFT/ITL. Not reproducible here (CPU only); the
nearest proxy is the mean decode batch in lab section 4 (2.36 vs 3.30 at equal work).

### Code walkthrough

**1. Why might decode run with a non-empty waiting queue?** `Scheduler._schedule_next_batch` falls
back to `DecodeManager.schedule_next_batch` whenever `PrefillManager.schedule_next_batch` returns
`None`, which happens when the head of `pending_list` fails `PrefillAdder.try_add_one`: no free row
(`TableManager.available_size == 0`) or the KV inequality fails. (The budget check cannot fail for
the head because the adder starts with the full `prefill_budget`.) Lab section 1 step 1 (slot) and
section 6 (KV).

**2. Can a request enter with a free slot but insufficient KV?** No. `_try_allocate_one` checks the
slot first, then `estimated_len + reserved_size > cache_manager.available_size` (and again after
`lock(handle)`, because locking a radix match makes those tokens non-evictable). Lab section 6: B
is rejected for 15 steps with 3 free rows, and is admitted at step 16 once A's pages return.

**3. Why must intermediate prefill chunks not commit generated tokens?** See concept exercise 6.
Code path: `PrefillAdder._add_one_req` builds a `ChunkedReq` when `chunk_size < remain_len`;
`Scheduler._prepare_batch` excludes it from `output_rows` (nothing written to `token_pool`);
`_process_last_data` `continue`s on it; `DecodeManager.filter_reqs` drops it (`can_decode` is
`False`); `PrefillManager.schedule_next_batch` puts its `PendingReq` at the front of `pending_list`.

**4. With §5.2's assumptions, when can C first enter if A generates only one token?** Step 1: A
finishes at the end of step 0's prefill (after `complete_one`, `device_len == max_device_len`, so
`Req.can_decode` is false and `_process_last_data` marks it finished), `_free_req_resources`
returns its row, and C is received and admitted at the start of
step 1. Lab section 3a.

**5. Where do completion and cancellation enter cleanup, and why does releasing a slot not delete
all KV?** Completion: `Scheduler._process_last_data` → `decode_manager.remove_req` +
`_free_req_resources`. Cancellation: `_process_one_msg` (`AbortBackendMsg`) →
`PrefillManager.abort_req` (a waiting request; resources only if it is mid-chunk) or
`DecodeManager.abort_req` → `_free_req_resources`. That function frees the table row
(`TableManager.free`) and hands the KV to `CacheManager.cache_req(req, finished=True)`, which
inserts the computed prefix into the prefix cache and frees only what the cache does not keep
(already-cached duplicates and the tail). The slot (row) and the KV pages have separate owners.
