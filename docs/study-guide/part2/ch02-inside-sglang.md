# Part II · Chapter 2 — Inside SGLang: The Path of a Request

> Tutorial: `part2/Chapter2_Inside-SGLang.md`, `part2/Chapter2_Inside-SGLang_code.md` · Lab: `labs/part2_ch02_request_path.py` · Bench: —

## Limitation → mechanism → validation

| Limitation | Mechanism | Primary validation | Where validated here |
|---|---|---|---|
| The architecture is clear, but not how one request flows through the actual source | Trace the path API server → tokenizer → scheduler + engine → detokenizer → API server, following the `uid` | Draw the request path and locate the key files | `labs/part2_ch02_request_path.py` (one uid, every message printed), `tests/test_detokenizer.py` (incremental text == full decode), `tests/test_server.py` (the same path across real processes) |

## Key ideas

* **Three data transformations, one identifier.** Text → token ids (tokenizer),
  context → new token ids (scheduler + engine, repeated), token ids → text
  (detokenizer). The `uid` assigned by the API server rides along unchanged on
  every message, which is how replies find their HTTP connection.
* **Five message types** (`message/__init__.py`):
  `TokenizeMsg` (API → tokenizer), `UserMsg` (tokenizer → scheduler),
  `DetokenizeMsg` (scheduler → detokenizer, one per request per step),
  `UserReply` (detokenizer → API), and for cancellation `AbortMsg` →
  `AbortBackendMsg`. `BatchMsg` wraps several for one socket send.
* **Scheduler decides, engine executes.** `_schedule_next_batch` picks requests
  and builds the batch; `_forward` hands it to `Engine.forward_batch` with a plain
  function call. Module boundary ≠ process boundary.
* **One step of the loop** (`Scheduler.normal_loop`): receive messages (block only
  when idle) → schedule (prefill first, else decode) → forward + sample →
  `_process_last_data` (append token, check length/EOS, free finished requests,
  emit `DetokenizeMsg`s).
* **The detokenizer is stateful.** BPE tokens can carry a fraction of a UTF-8
  character, so per-token decoding produces `�`. Per uid it keeps the token list
  and two offsets and only emits text once it no longer ends in `�`.
* **Processes buy overlap and isolation, and cost messages.** Tokenizing B's long
  prompt does not pause A's decode; TP ranks get one scheduler process each; the
  number of tokenizers can grow where the queue is. The price: pickling + ZMQ hops,
  start-up/shutdown coordination, and state that dies with a process.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| uid creation (server) | `python/minisgl/server/api_server.py:FrontendManager.new_user` | also creates `ack_map[uid]`, `event_map[uid]` |
| uid creation (offline) | `python/minisgl/llm/llm.py:LLM._new_uid` | |
| send `TokenizeMsg` | `server/api_server.py:_submit` → `FrontendManager.send_one` | |
| text → ids | `python/minisgl/tokenizer/tokenize.py:TokenizeManager.tokenize_one` | chat messages go through `apply_chat_template(add_generation_prompt=True)` |
| tokenizer worker loop | `python/minisgl/tokenizer/server.py:tokenize_worker` | `TokenizeMsg` → `UserMsg`, `AbortMsg` → `AbortBackendMsg`; tokenization errors → `UserMsg(error=...)` |
| scheduler inbox | `python/minisgl/scheduler/io.py:ZmqSchedulerIO.receive_msg` / `OfflineIO` | offline: an in-process deque |
| admission checks | `python/minisgl/scheduler/scheduler.py:Scheduler._process_one_msg` | rejects empty / too-long prompts with an error reply; clamps `max_tokens` |
| waiting queue | `python/minisgl/scheduler/prefill.py:PrefillManager.pending_list` | |
| the loop | `Scheduler.normal_loop` / `run_forever` | no `overlap_loop` in this repo |
| who runs next | `Scheduler._schedule_next_batch`, `_prepare_batch` | prefill batch or decode batch, never mixed |
| call the engine | `Scheduler._forward` → `python/minisgl/engine/engine.py:Engine.forward_batch` | |
| sampling | `python/minisgl/engine/sample.py:Sampler.sample`, `sample_impl` | |
| length / EOS check | `Scheduler._process_last_data` | `finished, reason = not req.can_decode, "length"`; EOS → `"stop"` unless `ignore_eos` |
| free resources | `Scheduler._free_req_resources` | table row + `cache_manager.cache_req(finished=True)` |
| ids → text | `python/minisgl/tokenizer/detokenize.py:DetokenizeManager.detokenize`, `_step`, `DecodeStatus` | |
| detokenizer worker | `tokenizer/server.py:detokenize_worker` | |
| route reply by uid | `server/api_server.py:FrontendManager.listen`, `wait_for_ack` | |
| `data: [DONE]` | `server/api_server.py:_stream` | |
| process start | `python/minisgl/server/launch.py:start_backend`, `_run_scheduler`, `launch_server` | |

SGLang counterparts (from the tutorial's table, under `python/sglang/srt/`):
`entrypoints/http_server.py` + `managers/tokenizer_manager.py` (frontend and
tokenizer), `managers/scheduler.py` (`event_loop_normal` / `event_loop_overlap`),
`managers/tp_worker.py` + `model_executor/model_runner.py` (engine),
`managers/detokenizer_manager.py`.

## This repo vs. the tutorial's reference code

* **Separate detokenizer entry point.** The reference launches tokenizer *and*
  detokenizer with `tokenize_worker` and branches on message type; here
  `launch.py` uses `tokenize_worker` for tokenizers and `detokenize_worker` for
  the detokenizer.
* **No `overlap_loop`**, only `normal_loop` (README "Deliberate simplifications").
* **Rejected requests get an answer.** The reference logs and drops a too-long
  prompt (the client hangs). Here `_process_one_msg` sends
  `DetokenizeMsg(finished=True, finish_reason="error")` for empty, too-long,
  untokenizable prompts and non-positive `max_tokens`.
* **`finish_reason` is real**: `"length"`, `"stop"`, `"abort"`, `"error"` travel
  in `DetokenizeMsg` and `UserReply`; the reference's finish reason does not
  distinguish a length limit from EOS (its stream end chunk always says `"stop"`).
* **Abort reaches the detokenizer.** After an abort the scheduler sends a final
  `DetokenizeMsg(next_token=-1, finished=True, finish_reason="abort")`, so
  `DetokenizeManager` deletes the uid's `DecodeStatus` (reference: the state stays).
* **Detokenizer holds back instead of `find_printable_text`.** When the new text
  ends in `�`, `_step` returns `""` and waits (unless the request just finished,
  which flushes); the reference emits a printable prefix of it.
* `Scheduler._forward` reads inputs from `table_manager.token_pool` (the
  reference's `self.token_pool`), and `Engine.forward_batch` copies sampled
  tokens to CPU synchronously (`next_tokens.to("cpu")`, no `non_blocking`).
* `--num-tokenizers` (alias `--num-tokenizer`); the default device is CUDA if
  available, otherwise CPU (`--device cpu` here).

## Common misconceptions

**Misconception:** The five stages are five processes.
**Actually:** The API server does both the first and the last stage, and the
scheduler and engine share a process. With TP=1 and one tokenizer there are
four processes: main (API server), `minisgl-TP0-scheduler`,
`minisgl-tokenizer-0`, `minisgl-detokenizer-0`.

**Misconception:** A request in the waiting queue will run on the next step.
**Actually:** `PrefillManager.schedule_next_batch` admits requests FCFS only
while the token budget, free table rows and KV reservation allow; the first one
that does not fit stops admission for this step.

**Misconception:** The scheduler checks EOS before it emits the token.
**Actually:** The token is appended first and always sent; EOS is detected on
it and the EOS token itself is part of `token_ids` (lab section 3:
`finish_reason=stop` with `token_ids` ending in the EOS id). Its text is empty
because the detokenizer uses `skip_special_tokens=True`.

**Misconception:** Detokenizing token by token and concatenating gives the same
text. **Actually:** Not when a character spans tokens (lab section 4: the 🚀 in
"KV cache 🚀 ok" is three tokens, each of which decodes to `�` alone).

**Misconception:** Multiple processes are always faster.
**Actually:** They add pickling and ZMQ hops on every message and only help
when the stages would otherwise wait on each other; more tokenizers do nothing
if the GPU (here: the CPU forward pass) is the bottleneck.

## Lab

```bash
python labs/part2_ch02_request_path.py          # ~6 s; tiny random weights + real Qwen3 tokenizer
```

What to look for:

1. The same uid (42) on `TokenizeMsg`, `UserMsg`, every `DetokenizeMsg` and every `UserReply`.
2. Step 0 is a prefill that already emits a token; steps 1..4 are decodes; the
   last row has `finish_reason=length` because `remain_len` reached 0.
3. Pretending the 3rd token is EOS: `stop` after 3 tokens; with `ignore_eos=True`, `length` after 5.
4. The detokenizer offsets:

```
| token | BPE piece | decode([t]) alone | (surr,read) before |     after |  emitted |
| 82707 |      'KV' |              'KV' |             (0, 0) |    (0, 1) |     'KV' |
|  6500 |  'Ġcache' |          ' cache' |             (0, 1) |    (1, 2) | ' cache' |
| 11162 |     'ĠðŁ' |              ' �' |             (1, 2) |    (1, 2) |       '' |
|   248 |       'ļ' |               '�' |             (1, 2) |    (1, 2) |       '' |
|   222 |       'Ģ' |               '�' |             (1, 2) |    (2, 5) |     ' 🚀' |
|  5394 |     'Ġok' |             ' ok' |             (2, 5) | (dropped) |    ' ok' |
per-token decode : 'KV cache ��� ok'
incremental      : 'KV cache 🚀 ok'
```

   `read_offset` marks what has been emitted; `surr_offset` keeps one emitted
   token as left context so the diff `decode(ids[surr:]) - decode(ids[surr:read])`
   is computed with the same neighbours both times. While the diff ends in `�`
   the offsets do not move; the third emoji byte releases ` 🚀` at once. On
   `finished` the uid's state is dropped.
5. Abort after 2 tokens: `finish_reason=abort`, the other request still
   finishes with `length`, and table rows / free pages return to 64/64 and 512/512.

## Exercises

### Concept chapter

**1. Draw the full request path; label inputs/outputs and the three transformations.**

```
client ──HTTP JSON──▶ API server (FrontendManager.new_user → uid)
   TokenizeMsg(uid, text|messages, SamplingParams) ──ZMQ──▶ tokenizer
      [T1: text → (chat template) → input_ids]
   UserMsg(uid, input_ids, SamplingParams) ──ZMQ──▶ scheduler rank 0
      pending_list → prefill Batch → Engine.forward_batch → logits → sample
      [T2: context → next token; repeats as decode steps]
   DetokenizeMsg(uid, next_token, finished, finish_reason) ──ZMQ──▶ detokenizer
      [T3: token ids → incremental text, per-uid DecodeStatus]
   UserReply(uid, incremental_output, finished) ──ZMQ──▶ API server.listen
      ack_map[uid] / event_map[uid] → wait_for_ack → SSE chunk or JSON ──▶ client
```

**2. Mark the generation loop; what do scheduler and engine do; separate processes?**
The loop is `Scheduler.run_forever` → `normal_loop`, one iteration per token
step (prefill once, then decode). The scheduler chooses requests, allocates KV
pages, builds positions / `out_loc` / metadata, applies stop conditions and
frees resources. The engine runs the model forward and samples. They need not
be separate processes and are not: `_forward` calls `self.engine.forward_batch`
directly; the per-step decisions depend on the engine's result, so a process
boundary there would only add latency.

**3. Why does the detokenizer need per-request state? What breaks with independent decoding?**
Because a character can be split across tokens and because some tokenizers
decode a token differently depending on its left neighbour. Decoding each token
alone emits `�` for each byte fragment (lab: `'KV cache ��� ok'`), and the
client cannot repair it since the fragments are already sent. The state (ids +
`surr_offset`/`read_offset`) must be per uid because replies of many requests
interleave in one detokenizer.

**4. If the user disconnects midway, what must the frontend and the scheduler handle?**
Frontend: notice the disconnect (`request.is_disconnected()` or task
cancellation), stop writing, delete `ack_map[uid]` / `event_map[uid]`, send
`AbortMsg(uid)`, and ignore late replies for that uid (`listen` checks
`reply.uid not in self.ack_map`). Scheduler: find the request whether it is
still waiting (`PrefillManager.abort_req`) or running
(`DecodeManager.abort_req`), free its table row and KV pages
(`_free_req_resources`), and (here) tell the detokenizer so its state is
dropped. See `labs/part2_ch05_frontend.py` section 4 and lab section 5 above.

**5. What is gained by separating tokenizer and scheduler processes? Why count communication costs?**
Gain: CPU-bound tokenization of a long new prompt overlaps the running decode
steps instead of stalling them; tokenizers can be scaled (`--num-tokenizers`)
independently; crashes and logs are localised per process. Costs: every
request and every token is pickled and copied through ZMQ (`BaseMsg.encode`),
start-up waits for all workers' ready acks, shutdown must be coordinated
(`exit_when_parent_dies`), and a dead process loses its state while others wait.
If the queue is in the forward pass, more processes do not help.

### Code walkthrough

**1. Where is the uid created, `TokenizeMsg` sent, `UserReply` received? How are they linked?**
Created in `server/api_server.py:FrontendManager.new_user` (monotonic counter,
also allocates `ack_map[uid]` and `event_map[uid]`); `TokenizeMsg(uid=uid, ...)`
built and sent in `_submit` via `send_one`; `UserReply` received in
`FrontendManager.listen` and consumed in `wait_for_ack(uid)`. They are linked
only by the integer uid copied onto each message: `tokenize_worker` copies
`msg.uid` into `UserMsg`, the scheduler stores it in `PendingReq.uid` → `Req.uid`
and puts it on every `DetokenizeMsg`, the detokenizer keys `status` by it and
copies it into `UserReply`. Lab section 1–2 asserts the uid is equal at every hop.

**2. `_schedule_next_batch` vs. `_forward`: which decides, which calls the engine?**
`Scheduler._schedule_next_batch` decides:
`prefill_manager.schedule_next_batch(self.prefill_budget) or
decode_manager.schedule_next_batch()`, then `_prepare_batch` allocates pages
(`cache_manager.allocate_paged`), computes `positions`, `out_loc`, attention
metadata and sampling args. `Scheduler._forward` executes: gathers
`batch.input_ids` from `table_manager.token_pool`, calls
`self.engine.forward_batch(batch, sample_args)`, writes the sampled tokens back
into `token_pool`, and updates the running set (`decode_manager.filter_reqs`).

**3. Interrupt a streaming client; follow `AbortMsg` to cancellation and cleanup.**
`_stream_with_cancellation` sees `request.is_disconnected()` (or is cancelled)
→ raises `CancelledError` → `aclosing` closes `_stream` and `wait_for_ack`
(their `finally` pops the uid) → task `FrontendManager.abort_user(uid)` pops
again and `send_one(AbortMsg(uid))` → `tokenize_worker` turns it into
`AbortBackendMsg` → `Scheduler._process_one_msg`: `prefill_manager.abort_req`
then `decode_manager.abort_req` → `_free_req_resources` (table row back,
`cache_req(finished=True)` frees or caches the KV) → final
`DetokenizeMsg(finish_reason="abort")` → `DetokenizeManager.detokenize` deletes
the uid → the `UserReply` is dropped by `listen` (uid already gone). Checked by
`tests/test_server.py::test_client_disconnect_aborts_and_cleans_up` (server
`/debug/state` returns to `{"ack_map": 0, "event_map": 0}`), this lab's
section 5, and `labs/part2_ch05_frontend.py` section 4. In the reference,
`abort_user` first sleeps 0.1 s and the detokenizer is never told.

**4. Find the length and EOS checks, and where completion becomes `data: [DONE]`.**
`Scheduler._process_last_data`:
`finished, reason = not req.can_decode, "length"` (length; `can_decode` is
`remain_len > 0` with `remain_len = max_device_len - device_len`), and
`if not req.sampling_params.ignore_eos and token in self.eos_token_ids:
finished, reason = True, "stop"`. The flag travels in `DetokenizeMsg.finished`
→ `UserReply.finished`; `wait_for_ack` returns after yielding a finished reply,
the `async for` in `server/api_server.py:_stream` ends, and `_stream` yields the
final chunk with `finish_reason` and `usage`, then `b"data: [DONE]\n\n"` (it also
sends `[DONE]` right after an error chunk). Lab sections 2–3 show both reasons.
