# Scheduler, advanced: four gaps of Chapters 5-6

The tutorial's scheduler (Chapter 6) and serving layer (Chapter 5) are correct
but deliberately simple. This note covers four things real SGLang does that the
teaching version leaves out. For each one: the problem, the design used here,
the invariants that make it correct, and where the same idea lives in SGLang.

| # | Topic | Code | Tests | Benchmark |
|---|---|---|---|---|
| 1 | Multi-tokenizer abort race | `server/api_server.py` (`FrontendManager.route`), `server/args.py`, `server/launch.py` | `tests/test_tokenizer_routing.py` | – |
| 2 | Optimistic admission + retraction | `scheduler/retract.py`, `scheduler/scheduler.py` (`_retract_decode_if_needed`), `scheduler/cache.py` (`release_for_retract`), `scheduler/prefill.py` | `tests/test_retract.py` | `benchmarks/bench_batching.py --sections admission` |
| 3 | Backpressure | `server/api_server.py` (`dispatch`, `_cut_off`, `reap_stalled`) | `tests/test_backpressure.py` | – |
| 4 | Overlap scheduling | `scheduler/scheduler.py` (`overlap_loop`, `_cpu_step`, `_mark_launched`, `_run_forward`, `_release_row`) | `tests/test_overlap.py`, plus `test_scheduler.py` and `test_retract.py` run under both loops | `benchmarks/bench_overlap.py` |

SGLang references are to `python/sglang/srt/` in
[sgl-project/sglang](https://github.com/sgl-project/sglang) (main, late 2026).
Names move around, so search for the function name if a path is stale.

---

## 1. The multi-tokenizer abort race

### Problem

With `--num-tokenizers N > 1`, the API server had **one** ZMQ `PUSH` socket that
all tokenizers `connect`ed to. ZMQ load-balances a `PUSH` socket round-robin
*per message*, so a request's `TokenizeMsg` and its later `AbortMsg` (sent when
the client disconnects) usually go to **different** tokenizers:

```
frontend --TokenizeMsg(uid=7, long prompt)--> tokenizer A  (busy tokenizing...)
frontend --AbortMsg(uid=7)-----------------> tokenizer B  -> AbortBackendMsg(7) -> scheduler
                                                                   "uid 7? never heard of it" (ignored)
tokenizer A ------------------------------------------> UserMsg(7) -> scheduler: runs to max_tokens
```

The abort overtakes the tokenization of its own request. The scheduler drops an
abort for an unknown uid (it has nothing to free), and then runs the request to
completion for a client that is gone. It is a race, so it depends on timing:
long prompts and idle tokenizers make it likely.

### Design

Make the route a function of the uid. The frontend binds **one `PUSH` socket per
tokenizer** (`tokenizer-0 ... tokenizer-{N-1}` IPC endpoints) and sends every
message of `uid` to tokenizer `uid % N` (`FrontendManager.route`).

### Invariants

* **Per-uid FIFO end to end.** Frontend → tokenizer `uid % N` is one ZMQ
  socket pair (FIFO). The tokenizer processes its inbox in order and forwards
  through one `PUSH` socket to the scheduler (FIFO per sender). So for a given
  uid, `UserMsg` always reaches the scheduler before `AbortBackendMsg`.
* Load balancing becomes static (`uid % N`) rather than round-robin. Uids are
  sequential, so this is still balanced.

### Tests

`tests/test_tokenizer_routing.py` runs four real `tokenize_worker` processes
between a real `FrontendManager` and a fake scheduler socket. It submits 32
long prompts, aborts each one immediately, and checks the arrival order:

* `test_shared_socket_reproduces_abort_race`: the old shared-socket topology.
  At least one abort overtakes its request.
* `test_uid_routing_preserves_per_uid_order`: with uid routing, every uid
  arrives as `[user, abort]`.

### In SGLang

* `managers/multi_tokenizer_mixin.py`. With `--tokenizer-worker-num > 1`, each
  HTTP worker process owns its own `TokenizerManager`, and a request's
  `GenerateReqInput` and its `AbortReq` are both sent by the worker that
  accepted the HTTP request. Per-rid order holds because the route is the
  connection, not a load balancer.
* `MultiDetokenizerRouter._pick` in the same file is the same idea for the
  other direction. Each `rid` is pinned to one detokenizer by
  `zlib.crc32(rid) % num_workers` so `detokenize_state_table` stays consistent.
* `Scheduler.abort_request` (`managers/scheduler.py`). For a running request,
  the scheduler marks it with `FINISH_ABORT` and lets the normal finish path
  free it.

---

## 2. Optimistic admission and retraction (preemption)

### Problem

`PrefillAdder` (Chapter 6 §4.2) admits a request only if
`extend_len + output_len + reserved_size <= available_size`, where
`reserved_size` holds the full remaining `output_len` of every running request.
That reservation can never be over-committed, so decode never runs out of KV.
But `max_tokens` is an upper bound: most chat requests stop at EOS long before
it, so most of the reservation is never used. Concurrency, and with it
throughput, stays far below what the pool could hold.

### Design (`--admission optimistic`, `LLM(admission="optimistic")`)

1. **Reserve a fraction.** Admission reserves `ceil(output_len * new_token_ratio)`
   instead of `output_len`. Running requests are counted as
   `max(0, ceil(output_len * ratio) - generated)`
   (`retract.reserved_output_tokens`, `DecodeManager.inflight_tokens(ratio)`).
   The prompt (`extend_len`) is always reserved in full, so a prefill
   allocation can never fail. Only decode can.
2. **Adapt the ratio.** `NewTokenRatioTracker` (`scheduler/retract.py`, same
   defaults as SGLang):
   - start at `init_new_token_ratio = 0.7`;
   - on every decode step that fits, decay linearly toward
     `min = init * 0.14` over `new_token_ratio_decay_steps = 600` steps
     (become more optimistic);
   - after a retraction, jump to the estimate
     `(decoded + retract_decode_steps * n) / (sum(output_len) + 1)` over the
     survivors, clipped to `[min, 1]` (become more careful).
3. **Check every decode step.** Before building a decode batch,
   `Scheduler._retract_decode_if_needed` compares the pages that step needs
   (`CacheManager.pages_needed`, the same arithmetic as `allocate_paged`)
   with `CacheManager.available_pages` (free pages + evictable radix pages).
4. **Retract.** If the step does not fit, sort the running requests
   (`retract.retraction_order`): keep the ones that generated the most, and
   among equals keep shorter prompts, with `uid` as the tiebreak so every TP
   rank picks the same order. Then pop from the end: retract at least one, and
   keep going until the rest fits. Each victim (`Scheduler._retract_one`):
   - returns its table row;
   - releases its KV with `CacheManager.release_for_retract`. The tail pages
     needed as headroom for the survivors' next `retract_decode_steps`
     (default 20) steps go **straight to the free list**. The rest of
     `[0, cached_len)` goes to the **radix cache** via `cache_req`
     (unlocked, therefore evictable);
   - goes back into the waiting queue as a `PendingReq` whose `input_ids` are
     prompt + every generated token, and whose `output_len` keeps the same
     `max_device_len` (`retracted=True`). It goes ahead of new arrivals but
     **behind unfinished chunked prefills**, because those hold pages and must
     be allowed to finish.
5. **Resume.** Re-admission is ordinary prefill. `match_req` finds whatever
   prefix survived in the radix cache, so only the rest is recomputed. Greedy
   output is unchanged, because a prefill over `prompt + generated` produces
   the same next token as the decode would have.

Why release the tail directly instead of inserting everything into the cache?
Radix eviction removes whole LRU **leaves**. The retracted request's freshly
inserted leaf is usually the only evictable one, so the allocation that
triggered the retraction would evict all of it just to get a few pages.
Freeing exactly the headroom from the tail first keeps the prefix (typically
the whole prompt) cached. SGLang's `release_req` makes the same headroom
calculation; it evicts `remaining * SGLANG_RETRACT_DECODE_STEPS` tokens from the
tree right away.

### Invariants

* Conservative mode is unchanged: the ratio is pinned to 1.0, never decays, and
  never retracts (`test_conservative_never_retracts_and_optimistic_runs_wider`).
* A retracted request's resumed `max_device_len` equals its original one, so
  `max_tokens` is honored across any number of retractions. Usage reports the
  **original** prompt length (`Scheduler.prompt_lens`), not prompt + generated.
* No page or table row leaks (`check_integrity` after every test). Retracting
  never double-frees: `release_for_retract` only frees pages the request owns,
  that is from `cache_handle.cached_len` on. Pages before that belong to the
  cache.
* Progress. If even one running request does not fit, all of them are
  retracted; unlike SGLang, mini-sglang has no abort-on-OOM path. The only
  remaining memory holders are chunked prefills, which were admitted with
  room for themselves. Retracted requests resume through prefill, which
  itself samples a token, so ping-pong still makes progress.
* An abort finds a retracted request in the waiting queue like any other
  pending request (`test_abort_while_retracted`).

### Tests (`tests/test_retract.py`, both loops)

* Outputs equal HuggingFace greedy under a 160-token KV pool. This holds for
  `page_size` 1 and 4, for `radix` and `naive` caches, and with chunked prefill
  (`prefill_budget=8`). Retractions are asserted to happen.
* `test_resume_reuses_radix_cache`: every re-admitted request starts with
  `cached_len > 0`.
* EOS mixes, abort-while-retracted, tracker arithmetic.

### Benchmark (Qwen3-0.6B, fp32, 4-core CPU, `bench_batching.py --sections admission`)

| workload | admission | KV tokens | wall s | output tok/s | peak batch | mean decode batch | retractions | identical |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exact lengths (ignore_eos) | conservative | 1024 | 32.8 | 37.6 | 12 | 9.2 | 0 | True |
| exact lengths (ignore_eos) | optimistic | 1024 | 30.9 | 39.9 | 13 | 10.0 | 0 | True |
| long exact outputs (ignore_eos) | conservative | 1024 | 118.5 | 35.1 | 7 | 5.3 | 0 | True |
| long exact outputs (ignore_eos) | optimistic | 1024 | 115.5 | 36.0 | 9 | 6.6 | 3 | True |
| chat, stops at EOS, max_tokens=256 | conservative | 2048 | 8.2 | 32.0 | 7 | 6.4 | 0 | True |
| chat, stops at EOS, max_tokens=256 | optimistic | 2048 | 7.0 | 37.4 | 10 | 8.2 | 0 | True |

How to read it:
* **Concurrency** (peak and mean decode batch) rises in every workload. That
  is the quantity admission controls.
* **Chat** is the case optimism exists for: `max_tokens=256`, but answers stop
  at EOS after a few dozen tokens. Peak batch goes 7 → 10, and throughput
  +17%. No retraction is needed, because the reserved-but-unused budget is
  never touched.
* **Long exact outputs** (`ignore_eos`, 96-160 tokens) are the worst case:
  every request really uses its whole budget, so the over-commit comes due and
  3 retractions happen. Outputs stay identical. (That resumed requests reuse
  their cached prefix is checked in `test_resume_reuses_radix_cache`, not
  measured here.)
* Throughput gains are smaller than the concurrency gains because a CPU decode
  step is compute-bound: its cost grows almost linearly with batch size. On a
  GPU, decode is memory-bound and a bigger batch is nearly free, so the same
  concurrency gain translates far more directly into throughput.

### In SGLang

| here | SGLang |
|---|---|
| `NewTokenRatioTracker` (`scheduler/retract.py`) | `managers/scheduler_components/new_token_ratio_tracker.py: NewTokenRatioTracker` (`decay_step`, `reset`, `estimate_new_token_ratio_after_retract`); env `SGLANG_INIT_NEW_TOKEN_RATIO`, `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR`, `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS`, `SGLANG_RETRACT_DECODE_STEPS` (`environ.py`) |
| `PrefillAdder(new_token_ratio=...)`, `DecodeManager.inflight_tokens(ratio)` | `managers/schedule_policy.py: PrefillAdder.__init__(new_token_ratio)`, `_get_running_request_total_token_offset`, `add_one_req` |
| `Scheduler._retract_decode_if_needed` | `managers/scheduler.py: Scheduler.update_running_batch` (`check_decode_mem`, then `retract_decode`, otherwise `decay_step`) |
| `retraction_order` | `managers/schedule_batch.py: ScheduleBatch._get_decode_retraction_order` (sort by `(len(output_ids), -len(origin_input_ids))`, pop from the end) |
| `_retract_one` + `release_for_retract` | `ScheduleBatch.retract_decode` → `release_req` (evicts `remaining * SGLANG_RETRACT_DECODE_STEPS` tokens) → `Req.reset_for_retract`; requeued by `Scheduler._add_request_to_queue(req, is_retracted=True)` |

Differences: SGLang applies ratio 1.0 to `ignore_eos` requests, since their
length is known. Here the ratio applies to every request, so the exact-length
benchmark and the tests exercise retraction. SGLang aborts the last request if
even it alone does not fit; here it is retracted instead.

---

## 3. Backpressure for slow or stuck clients

### Problem (Chapter 5 §5.4)

The frontend's shared `listen` coroutine appends every reply to
`ack_map[uid]` as fast as the backend produces it. The client's coroutine
drains that list only when it can hand data to the HTTP connection. If a
streaming client stops reading (its TCP window fills, `send` blocks), the list
grows for as long as generation runs, one object per token. Many such clients
can grow memory without bound, and a client that never returns keeps its entry
forever. The scheduler cannot slow down for one client: the request shares
every decode step with the rest of the batch.

### Design: bound, then cancel (`FrontendManager`)

1. **Coalesce** (`dispatch` + `_merge`). A reply arriving while one is still
   undelivered is merged into it: incremental text is concatenated, and the
   latest status and counters win. `ack_map[uid]` therefore holds **at most one
   object**, and its cost is the undelivered text, not one Python object per
   token. A slow streaming client receives bigger chunks; nothing is lost.
2. **Size cap** (`--max-buffered-bytes`, default 1 MiB). If undelivered text
   exceeds the cap, the request is cut off.
3. **Stall timeout** (`--client-stall-timeout`, default 60 s). A reaper task
   (`reap_stalled`) cuts off any uid whose replies have waited longer than the
   timeout since the consumer last took them.
4. **Cut off** (`_cut_off`):
   - send `AbortMsg` upstream, so the scheduler frees the request's KV and slot;
   - drop all later replies for the uid;
   - replace the buffer with one terminal error reply
     (`finish_reason="abort"`, `error="client too slow ..."`). A non-streaming
     client gets HTTP 400; a stream gets an error event and `[DONE]`.

   If the consumer never comes back, the reaper drops even that entry after
   another timeout. If it does come back after that, `wait_for_ack` yields a
   synthetic error instead of hanging.

Either limit can be disabled with `0`.

### Invariants

* `len(ack_map[uid]) <= 1` at all times, and undelivered text per uid stays
  `<= max_buffered_bytes` (plus one reply).
* A request whose **backend** is slow (queued, long prefill) is never cut off.
  The stall clock only matters while replies are waiting
  (`test_idle_request_waiting_for_tokens_is_not_stalled`).
* A fast consumer is never cut off, even with a tiny cap
  (`test_fast_consumer_is_never_cut_off`).
* All per-uid state is gone after the request ends: normally, cut off, or
  reaped (`/debug/state` returns to zeros).

### Tests

`tests/test_backpressure.py` drives `FrontendManager.dispatch` directly, as
`listen` would for every detokenizer message, and records upstream messages.
It covers coalescing, the size cap, the stall timeout plus reaping, and the
two cases above that must not trip. `test_server.py::test_backpressure_flags_reach_frontend`
checks the CLI plumbing end to end.

### In SGLang

* `managers/tokenizer_manager.py: ReqState.out_list` is the per-request buffer
  (our `ack_map[uid]`). `TokenizerManager._handle_batch_output` appends;
  `_wait_one_response` drains the whole list atomically.
* `TokenizerManager._coalesce_streaming_chunks`: with incremental streaming,
  queued chunks are coalesced into one, concatenating `text` / `output_ids`
  with the last chunk's meta info. It logs "Streaming backlog" once 20 or
  more chunks queue up. Our `_merge` does the same, eagerly.
* Cancel on disconnect: `_wait_one_response` checks
  `request.is_disconnected()` and calls `abort_request`. Request timeouts are
  aborts that carry an `abort_message`, answered with HTTP 503
  (`Scheduler.abort_request`).

---

## 4. Overlap scheduling

### Problem

`normal_loop` is strictly sequential: `schedule N → forward N → process N →
schedule N+1 ...`. The device idles while the CPU schedules and processes. On a
GPU the CPU part (Python scheduling, metadata, detokenization hand-off) can be
a large fraction of a decode step for small models. SGLang's
`event_loop_overlap` hides it: the CPU prepares batch N+1 and processes N while
the GPU runs.

### Design (`--enable-overlap`, `LLM(disable_overlap=False)`)

```
forward thread (main):  [forward N  ][forward N+1        ][forward N+2 ]
CPU thread:             [proc N-1, sched N+1][proc N, sched N+2][...]
```

`Scheduler.overlap_loop` (one iteration):
1. Submit `_cpu_step(done)` to the CPU thread. It processes the finished batch
   N, receives messages, and schedules batch N+2 (`_mark_launched`).
2. Meanwhile, run batch N+1's forward on this thread (`_run_forward`).
3. Join; batch N+1 becomes `done` and N+2 becomes `ready`.

With nothing to overlap (start-up, idle), the CPU step runs inline and can
block on the socket, exactly like `normal_loop`.

**Why the forward stays on the main thread.** On CUDA the scheduler thread
would just enqueue kernels on a stream and move on. A CPU has no asynchronous
device, so two threads are needed, and which one runs the forward matters.
PyTorch's CPU kernels use OpenMP, and every OS thread that opens a parallel
region gets its **own** OpenMP team. Once a second team exists (4+4 threads on
4 cores), libgomp's spin-waiting stops paying off, and *every* parallel region
in the process becomes slower; SDPA went from 26 µs to 80+ µs in an isolated
test. The main thread already owns the team that loaded the model, so it keeps
the forward. The CPU thread calls libgomp's per-thread `omp_set_num_threads(1)`
(`_limit_openmp_to_one_thread`). `torch.set_num_threads` would not work here,
because it is process-wide. The initializer also warms ATen with an op
**below** the parallel grain size; a bigger warm-up op would itself create the
second team.

### The hard parts and their invariants

1. **Lengths advance at scheduling time, not after the forward.**
   `Engine.forward_batch` calls `req.complete_one()` after the model runs. In
   overlap mode the next schedule happens while the forward is still running,
   so `_mark_launched` does `complete_one()` + `DecodeManager.filter_reqs()` on
   the CPU thread right away. The forward receives a shallow copy of the batch
   with `reqs = []`; the model reads only prepared tensors. This needs no
   change to `engine/`.
2. **token_pool write ordering.** A decode request's input token is the token
   sampled by the *previous* batch. That token is only in `token_pool` on the
   device side, not yet on the host. `_run_forward` does both the gather of
   batch N+1's input ids and the scatter of its sampled tokens. Forwards run
   one at a time on one thread, so N+1's gather always follows N's scatter.
   That is the guarantee stream order gives on GPU; SGLang adds a `FutureMap`
   to make it explicit.
3. **Host tokens lag by one batch.** When batch N is processed, its requests
   already carry N+1's lengths. So:
   - finish-by-length is judged by host tokens,
     `len(input_ids) >= max_device_len`, not by `can_decode`. With
     `can_decode`, the request would stop one token early; that was the first
     bug the tests caught;
   - while N+1 is in flight, `len(input_ids) == cached_len` holds for each of
     its requests after processing N. `cache_req` inserts exactly the tokens
     on the host, and the KV of the last one is being written by N+1. That is
     safe because anything that reuses those pages runs in a later batch.
4. **`finished_reqs` guard.** A request that hits EOS in batch N was already
   scheduled into N+1, because EOS was not known yet. The same goes for a
   request aborted or retracted while its batch was in flight. Its N+1 token
   is not real: `_process_results` skips it and counts it in
   `stats.discarded_tokens`. It is never freed twice, never replied to twice,
   and never put back in the running set. Requests that finish by **length**
   are never over-scheduled, because `filter_reqs` drops them at launch.
5. **KV pages may be freed immediately; table rows may not.** A KV page freed
   while a batch that reads or writes it is in flight can only be handed to a
   batch launched *later*, and forwards run in launch order. A **table row**
   is different: the next admission writes its prompt into `token_pool[row]`
   on the CPU thread *right away*, while the in-flight forward is still about
   to scatter the dead request's token into the same row. `_release_row`
   therefore parks the row in `_graveyard` tagged with the newest launched
   batch, and `_bury_until(seq)` returns it after that batch is processed.
   `test_fast_row_reuse_keeps_token_pool_consistent` reproduces the clobber
   deterministically: it uses 1 row, short EOS prompts alternating with long
   prompts, and a slowed forward. The test fails if the deferral is removed.
6. **Aborts mid-flight.** These are just case 4 + case 5: remove the request
   from the queue or running set, free its pages, and park its row. The
   abort reply goes out immediately; the in-flight token is discarded
   (`test_abort_mid_flight`).
7. **Retraction mid-flight.** The victim's host tokens lack the in-flight
   token, so its `PendingReq.output_len` is computed as
   `max_device_len - len(input_ids)`, which is valid in both loops. The
   in-flight token is discarded and re-sampled on resume. For greedy decoding
   that gives the same token.
8. **Ending.** `drain()` runs and processes whatever is still scheduled, so
   `LLM.generate` returns with every row and page back
   (`check_integrity`), and `run_forever` exits cleanly.
9. **Not with TP > 1.** The request broadcast and the TP all-reduces share one
   Gloo group on CPU. Issuing them from two threads could interleave them
   differently on each rank, so overlap falls back to `normal_loop` with a
   warning.

### Default

`normal_loop` stays the default and remains the teaching path. `overlap_loop`
passes the full scheduler and retraction suites under both loops, and the
served pipeline in overlap + optimistic mode. But it buys nothing measurable on
this hardware (next section), so it is opt-in.

### Benchmark (`benchmarks/bench_overlap.py`, 4-core CPU, fp32)

| model | requests | loop | wall s | output tok/s | mean TPOT ms | forward ms/step | un-hidden CPU ms/step | speedup | identical |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| tiny Qwen3 (CPU-overhead bound) | 16 | normal | 0.14 | 7530.3 | 2.1 | 1.50 | 0.63 | 1.00x | True |
| tiny Qwen3 (CPU-overhead bound) | 16 | overlap | 0.15 | 6722.2 | 2.2 | 1.60 | 0.78 | 0.89x | True |
| tiny Qwen3 (CPU-overhead bound) | 64 | normal | 0.30 | 13841.3 | 4.9 | 2.84 | 1.72 | 1.00x | True |
| tiny Qwen3 (CPU-overhead bound) | 64 | overlap | 0.33 | 12415.3 | 4.4 | 3.18 | 1.90 | 0.90x | True |
| Qwen3-0.6B (compute bound) | 4 | normal | 4.58 | 27.9 | 131.0 | 142.50 | 0.74 | 1.00x | True |
| Qwen3-0.6B (compute bound) | 4 | overlap | 4.69 | 27.3 | 129.8 | 145.63 | 0.79 | 0.98x | True |
| Qwen3-0.6B (compute bound) | 16 | normal | 9.47 | 54.1 | 256.7 | 294.74 | 1.26 | 1.00x | True |
| Qwen3-0.6B (compute bound) | 16 | overlap | 9.00 | 56.9 | 237.7 | 280.16 | 1.04 | 1.05x | True |

How to read it:
* **Qwen3-0.6B.** The CPU work per step (~1 ms) is under 1% of a 140-300 ms
  forward, so even perfect overlap could not move the needle; ±5% is noise.
* **Tiny model.** CPU work is a third of the step, which is exactly where
  overlap should help on a GPU. On a CPU it doesn't: the forward already uses
  every core and needs the GIL between ops, so scheduling running beside it
  slows the forward down by about as much as it hides.
* The first version ran the forward on a worker thread, with the scheduler on
  the main thread. It measured 0.4-0.85x on the same table, which is why the
  forward now lives on the main thread.

Expect a real gain on a GPU, where `_run_forward` only enqueues kernels.
Wiring that up means moving the `token_pool` gather/scatter onto the compute
stream and `next_tokens.to("cpu", non_blocking=True)` plus an event, which is
`engine/` work.

### In SGLang

| here | SGLang |
|---|---|
| `Scheduler.overlap_loop` / `normal_loop` | `managers/scheduler.py: Scheduler.event_loop_overlap` / `event_loop_normal`; `--disable-overlap-schedule` |
| `_cpu_step` processing the previous batch after launching the next | `event_loop_overlap`: `result_queue`, `pop_and_process()` after `run_batch(batch)`; `should_process_last_batch_early` (process first when a prefill result is needed for TTFT, or for grammar) |
| `_run_forward` gather/scatter of `token_pool` | `managers/overlap_utils.py: FutureMap` (placeholder token ids written by forward N, resolved by forward N+1 on the same stream); `managers/tp_worker.py` |
| `finished_reqs` skip in `_process_results` | `managers/scheduler_components/batch_result_processor.py: process_batch_result_decode` (`still_in_batch_rows`: "Left the batch after launch ... continue") |
| `filter_reqs` at launch (length-finished requests not rescheduled) | `Scheduler._reqs_finishing_inflight` + `enable_skip_finishing_decode` in `update_running_batch` |
| `_release_row` / `_graveyard` | deferred freeing of request slots and KV in `release_kv_cache` / `free_group_begin`/`free_group_end`; stream ordering via `_apply_war_barrier` in `event_loop_overlap` |
