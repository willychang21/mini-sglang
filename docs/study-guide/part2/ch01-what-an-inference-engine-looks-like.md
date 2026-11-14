# Part II · Chapter 1 — What an Inference Engine Looks Like

> Tutorial: `part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md` (concept only, no `_code` file) · Lab: none of its own; use `labs/part2_ch02_request_path.py` (request path, states) and `labs/part2_ch04_kv_cache.py` (Req length fields) · Bench: —

## Limitation → mechanism → validation

Chapter 1 is the map, not a mechanism. Its §4.1 table defines the
*limitation → new capability → primary validation* row for every later chapter
(reproduced, with this repo's validation, in [`../README.md`](../README.md)).
Chapter 1's own "validation" is that you can point at the code for every box:

| Box in the chapter | Validated here by |
|---|---|
| `SamplingParams`, `Req`, `Batch`, `Context` | read `python/minisgl/core.py` (170 lines, all four in one file) |
| request states `WAITING → PREFILL → DECODE → FINISHED / ABORTED` | `labs/part2_ch02_request_path.py` sections 2, 3, 5 print each transition |
| process structure | `python/minisgl/server/launch.py:start_backend`, `tests/test_server.py` (starts the real multi-process server) |

## Key ideas

* **Model vs. engine.** The model maps token ids to logits. The engine decides
  *which* requests run, *when*, *which tokens* each one feeds this step, and
  *where* their K/V lives. A loop around `model(input_ids)` lacks all four
  (queueing, batching, stop handling, cache allocation and freeing, text output).
* **Four responsibilities.** I/O (tokenize / detokenize / stream), request
  orchestration (lifecycle, cleanup), computation scheduling (which batch, prefill
  or decode), state and memory management (KV allocation, reuse, release).
* **Two views of the same system.** *Logical layers* (interface → orchestration →
  execution → state/memory) say what each part is responsible for. *Processes*
  (API server, tokenizer, detokenizer, one scheduler per TP rank) say who runs it.
  The two do not map one-to-one: scheduler and engine are separate layers but
  share one process.
* **Control path vs. data path.** The control path carries request *state*
  (waiting, prefilling, decoding, finished): scheduler, managers, messages. The
  data path carries *tensors*: `input_ids`, `positions`, `out_loc`, logits, K/V.
  `Batch` is where the two meet: the scheduler's decision, turned into tensors.
* **The four shared objects.** `SamplingParams` (how to pick tokens) → `Req`
  (one request's tokens, lengths and cache handle) → `Batch` (the requests of one
  forward pass + their prepared tensors) → `Context` (per-process runtime: KV pool,
  page table, attention backend, the batch being executed).
* **`cached_len` vs. sequence length.** A request's tokens split into
  `[0, cached_len)` (K/V already in the pool) and `[cached_len, device_len)` (to
  compute this step). `extend_len = device_len - cached_len` is the number of
  tokens the step processes: the whole prompt in prefill, 1 in decode, less than
  the prompt after a prefix-cache hit.
* **Incremental roadmap.** Each chapter removes one bottleneck of the previous
  version and proves it with one metric; correctness is established before speed.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| sampling configuration | `python/minisgl/core.py:SamplingParams` | `temperature=0.0` means greedy; `is_greedy` also true for `top_k == 1` |
| one request | `python/minisgl/core.py:Req` | `input_ids`, `table_idx`, `cached_len`, `output_len`, `uid`, `sampling_params`, `cache_handle`, `output_ids`; `device_len` / `max_device_len` set in `__post_init__` |
| lengths | `Req.extend_len`, `Req.remain_len`, `Req.can_decode`, `Req.complete_one` | `can_decode` is the length stop condition |
| partially prefilled request | `python/minisgl/core.py:ChunkedReq` | `can_decode` is always `False` (Ch6) |
| one forward pass | `python/minisgl/core.py:Batch` | `reqs`, `phase` (`"prefill"`/`"decode"`), `return_all_logits`, then `input_ids`, `positions`, `out_loc`, `attn_metadata` filled later |
| shared runtime | `python/minisgl/core.py:Context`, `set_global_ctx` / `get_global_ctx` | `Context.forward_batch(batch)` sets the batch layers read through `get_global_ctx().batch` |
| WAITING | `python/minisgl/scheduler/prefill.py:PrefillManager.pending_list` (`PendingReq`) | not yet a `Req`: no table row or KV slots |
| PREFILL | `scheduler/prefill.py:PrefillManager.schedule_next_batch` → `Batch(phase="prefill")` | |
| DECODE | `python/minisgl/scheduler/decode.py:DecodeManager.running_reqs` | |
| FINISHED / ABORTED | `python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data` (`"length"`, `"stop"`), `_process_one_msg` (`"abort"`, `"error"`) | the reason travels in `DetokenizeMsg.finish_reason` |
| process structure | `python/minisgl/server/launch.py:start_backend`, `launch_server` | process names `minisgl-TP{i}-scheduler`, `minisgl-tokenizer-{i}`, `minisgl-detokenizer-0` |
| the naive loop of §2.1 | `python/minisgl/llm/naive.py:generate_without_kv_cache` | kept as the Chapter 3/4 baseline |

## This repo vs. the tutorial's reference code

* Same package layout (`python/minisgl/`), but **no `kernel/`, `moe/` or
  `benchmark/` subpackages**: everything is pure PyTorch, only dense Qwen3 is
  supported, and benchmarks live in the top-level `benchmarks/`. There is an
  extra `speculative/` package (Chapter 10) and a `config.py`.
* `Batch` has **no padding fields**: there are no CUDA graphs, so no batch is
  padded to a captured size. It has a `return_all_logits` flag (used by
  speculative verification and by the Ch3 lab).
* `Req` stores `output_len` and derives `max_device_len = len(input_ids) + output_len`;
  it also keeps `output_ids` and a `table_idx` (its row in `page_table` / `token_pool`).
* `Context` holds `page_table`, `kv_cache`, `attn_backend` and the current batch;
  the backend is always `TorchAttnBackend` (no FlashAttention/FlashInfer).
* The README "Differences" list adds: no overlap loop, no CUDA graph, Gloo
  `broadcast_object_list` for TP request broadcast.

## Common misconceptions

**Misconception:** An inference engine is an HTTP wrapper around `model.generate`.
**Actually:** HTTP is one thin layer (`server/api_server.py`, which never touches
the model). The bulk of the engine is scheduling and memory management:
`scheduler/` and `kvcache/` together are larger than `models/` + `layers/`.

**Misconception:** The states `WAITING/PREFILL/...` are an enum in the code.
**Actually:** The chapter says they are an abstraction. Here a state is *which
container holds the request*: `pending_list` (waiting), a prefill `Batch`,
`running_reqs` (decode), or gone (finished/aborted, reason in `finish_reason`).

**Misconception:** One logical layer = one process.
**Actually:** Scheduler and engine are different layers in the same process
(`_run_scheduler` builds both); the API server's interface layer is split
between the main process and tokenizer/detokenizer workers.

**Misconception:** `cached_len` is the number of generated tokens.
**Actually:** It counts tokens whose K/V is already stored, prompt included (and
prefix-cache hits). Generated-token count is `len(req.output_ids)`.

**Misconception:** `Context` belongs to a request.
**Actually:** There is one `Context` per engine process (`Engine.__init__` calls
`set_global_ctx`); it points at whichever `Batch` is running.

## Lab

There is no Chapter 1 lab; the map becomes concrete in Chapter 2's:

```bash
python labs/part2_ch02_request_path.py
```

Look for one `uid` passing `TokenizeMsg → UserMsg → (prefill, decode, …) →
DetokenizeMsg → UserReply`, the `finish_reason` column (`length`, `stop`,
`abort`), and the resource counters returning to their starting values:

```
| step |   phase | uid | next_token | finished | finish_reason | ...
|    0 | prefill |  42 |      53603 |    False |          None | ...
|    4 |  decode |  42 |      69489 |     True |        length | ...
aborted_reqs=1; free table rows 64/64; free pages 512/512
```

For `cached_len / device_len / extend_len` per step, see the trace table in
`python labs/part2_ch04_kv_cache.py` (section 3).

## Exercises

### Concept chapter

**1. Why is a program that can run `model(input_ids)` not yet an inference engine?**
It only computes logits for one sequence. It has no queue for concurrent
requests, no policy for which requests share a forward pass, no per-request
stop handling, no KV storage (so every step recomputes the whole prefix, see
`llm/naive.py`), no allocation/freeing of that storage, and no path back to the
client as incremental text. Those are exactly the four responsibilities the
engine adds.

**2. Responsibilities of the model vs. the engine (≥3 each).**
Model: embedding lookup; attention/MLP/norm computation (`models/qwen3.py`);
producing logits through the LM head; storing its weights. Engine: tokenization
and detokenization (`tokenizer/`); admission, batching and prefill/decode
choice (`scheduler/`); KV allocation, reuse and freeing (`scheduler/cache.py`,
`kvcache/`); sampling and stop conditions (`engine/sample.py`,
`Scheduler._process_last_data`); serving, streaming, cancellation (`server/`).
(Sampling is arguably neither: here it lives in the engine, not the model.)

**3. What do `Req`, `Batch`, `Context` represent and how are they related?**
`Req` = one request's state over its whole life (tokens, `cached_len`,
`device_len`, cache handle, sampling params). `Batch` = the subset of requests
run in one forward pass, plus the tensors prepared for it (`input_ids`,
`positions`, `out_loc`, `attn_metadata`) and its `phase`. `Context` = per-process
runtime shared by all passes (KV pool, page table, attention backend) with a
pointer to the current `Batch`. So `Context ⊃ current Batch ⊃ Reqs ⊃
SamplingParams + cache handle` (`core.py`).

**4. States a request passes through; when FINISHED / ABORTED?**
WAITING (in `pending_list`) → PREFILL (a prefill batch; with chunked prefill,
several prefill steps as a `ChunkedReq`) → DECODE (in `running_reqs`, one token
per step) → FINISHED. FINISHED when the sampled token is in `eos_token_ids` and
`ignore_eos` is false (`finish_reason="stop"`), or when `remain_len` hits 0, i.e.
`max_tokens` (clamped by `max_seq_len` / KV capacity) is reached
(`"length"`). ABORTED when an `AbortBackendMsg` arrives (client disconnect;
`"abort"`) or the request is rejected at admission: empty or too-long prompt,
tokenization failure (`"error"`, `Scheduler._reply_error`). A request can be
aborted from WAITING as well as from DECODE (`PrefillManager.abort_req`,
`DecodeManager.abort_req`).

**5. Why separate the control path from the data path? Example of each.**
They change for different reasons and fail in different ways: a scheduling bug
(wrong request admitted, slot not freed) is a control-path bug; a wrong logit
(bad `positions`, wrong `out_loc`) is a data-path bug. Keeping them apart lets
you test each alone (this repo checks the data path token-for-token against
HuggingFace in `tests/test_model.py`, and the control path with
`CacheManager.check_integrity` and the scheduler tests). Control: `UserMsg`,
`DetokenizeMsg(finished, finish_reason)`, `pending_list` / `running_reqs`
membership. Data: `batch.input_ids`, `batch.positions`, `batch.out_loc`, the KV
pool, logits.

**6. Why single-request generation before KV cache, HTTP, continuous batching?**
Each later mechanism must reproduce the output of the simpler version, so the
simple version is the test oracle. If you start with a cached, batched, served
engine and get a wrong token, the bug could be in RoPE positions, slot
allocation, the mask offset by `cached_len`, batch packing, or message routing:
too many suspects. Here `llm/naive.py` (no cache) is compared against
HuggingFace, then the cached scheduler is compared against both
(`labs/part2_ch04_kv_cache.py` section 5: "identical output: True").
Performance claims also need a baseline to measure against.

**7. Responsibilities of official `engine`, `scheduler`, `kvcache`, `server`, `tokenizer`.**

| Official directory (reference commit 9a91cfa) | Responsibility | This repo |
|---|---|---|
| `engine/` | everything on one TP worker: model, KV pool, page table, attention backend, sampler, forward + sample (plus CUDA graphs) | `engine/engine.py:Engine.forward_batch`, `engine/sample.py:Sampler` (no CUDA graphs) |
| `scheduler/` | the loop (receive → schedule → forward → process), waiting queue, running set, table/cache managers, I/O to tokenizer/detokenizer | `scheduler/scheduler.py:Scheduler.normal_loop`, `prefill.py`, `decode.py`, `cache.py`, `table.py`, `io.py` (no `overlap_loop`) |
| `kvcache/` | physical KV pool and prefix caches (naive / radix) with their handles | `kvcache/mha_pool.py:MHAKVCache`, `naive_cache.py`, `radix_cache.py:RadixPrefixCache` |
| `server/` | CLI args, process launch, FastAPI frontend | `server/args.py`, `server/launch.py`, `server/api_server.py:FrontendManager` |
| `tokenizer/` | text ↔ ids: tokenization (with chat template), incremental detokenization, the worker loop | `tokenizer/tokenize.py:TokenizeManager`, `detokenize.py:DetokenizeManager`, `server.py:tokenize_worker` / `detokenize_worker` (the reference runs both roles through `tokenize_worker`) |
