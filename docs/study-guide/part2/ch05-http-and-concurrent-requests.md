# Part II · Chapter 5 — HTTP and Concurrent Requests

> Tutorial: `part2/Chapter5_HTTP-and-Concurrent-Requests.md`, `part2/Chapter5_HTTP-and-Concurrent-Requests_code.md` · Lab: `labs/part2_ch05_frontend.py` · Bench: `benchmarks/bench_serving.py`

## Limitation → mechanism → validation

| Limitation | Mechanism | Primary validation | Where validated here |
|---|---|---|---|
| The engine is reachable only through Python function calls | FastAPI frontend, uid-keyed `ack_map` / `event_map`, one shared `listen` coroutine, SSE streaming, cooperative cancellation | End-to-end latency | `benchmarks/bench_serving.py` (TTFT / TPOT / throughput vs concurrency, numbers in README), `tests/test_server.py` (real multi-process server: offline == HTTP output, streaming == non-streaming, 8 concurrent requests isolated, disconnect cleans `/debug/state`, errors returned), lab `part2_ch05_frontend.py` (routing and cleanup without GPU, ZMQ or HTTP) |

## Key ideas

* **Client / server / HTTP.** A client sends `POST /v1/chat/completions` with a
  JSON body; the server keeps the model loaded and answers many clients. HTTP
  defines the message format, not how the model runs.
* **The HTTP coroutine never runs the model.** It turns the request into a
  `TokenizeMsg` with a fresh uid, sends it over ZMQ, and *awaits* replies.
  Tokenizing, scheduling and computing happen in other processes, so the event
  loop is free to accept new connections.
* **`async def` is not enough.** A coroutine that calls something blocking
  (no `await`) holds the event loop's only thread; every other connection waits.
* **Data vs notification.** `ack_map[uid]` is a list that stores replies;
  `event_map[uid]` is an `asyncio.Event` that wakes the waiter. An Event is a flag,
  not a counter: several `set()` calls can share one wake-up, and the list keeps
  all the data.
* **One producer, many consumers.** `listen` is the only reader of the
  detokenizer socket and dispatches by uid; each request's `wait_for_ack(uid)`
  consumes only its own list. Interleaved arrivals (`A1 B1 B2 A2`) stay ordered
  per request.
* **wait → clear → swap list → yield.** No `await` between `clear()` and the swap,
  so `listen` cannot interleave there; replies arriving while the consumer is
  yielding go to the fresh list and re-set the Event.
* **Concurrency ≠ parallelism ≠ batching.** Accepting B while A waits
  (concurrency) is the frontend's job; running work at the same instant
  (parallelism) is about processes / cores; sharing one forward pass (batching)
  is the scheduler's job (Ch6).
* **Streaming changes delivery, not computation.** SSE (`data: {...}\n\n`, then
  `data: [DONE]`) shows text sooner; the number of forward passes is the same.
  SSE events and tokens are not 1:1 (a held-back partial character produces no
  event).
* **Cooperative cancellation.** A disconnect becomes `AbortMsg` → `AbortBackendMsg`
  → scheduler removes the request and frees its KV. It is a message, so it takes
  time; tokens already in flight arrive late and must be dropped.
* **Lifecycle cleanup on every exit path** (finish, early break, cancel, error),
  and **backpressure**: nothing slows the scheduler when a client reads slowly.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| request model | `python/minisgl/server/api_server.py:OpenAICompletionRequest`, `Message` | `prompt` may be a string or token ids; unknown fields (`n`, `stop`) are ignored by Pydantic |
| sampling params | `OpenAICompletionRequest.sampling_params` | `temperature` defaults to 1.0 |
| routes | `v1_chat_completions`, `v1_completions` → `_handle` | also `/v1/models`, `/health`, `/debug/state` |
| HTTP → internal message | `api_server.py:_submit` | chat route uses `messages` (as dicts) if given, else `prompt`; neither → 400 |
| per-uid state | `FrontendManager.new_user`, `ack_map`, `event_map` | |
| shared listener | `FrontendManager.listen`, `_create_listener_once`, `send_one` | drops replies whose uid is gone |
| consumer | `FrontendManager.wait_for_ack` | cleanup in `finally` |
| non-streaming | `FrontendManager.collect` (with `aclosing`), `_collect_or_abort` | polls `is_disconnected()` every 0.5 s |
| streaming | `_stream` (SSE encoder), `_stream_with_cancellation`, `StreamingResponse(media_type="text/event-stream")` | |
| abort | `FrontendManager.abort_user` → `AbortMsg` | |
| abort conversion | `python/minisgl/tokenizer/server.py:tokenize_worker` | `AbortMsg` → `AbortBackendMsg` |
| abort handling | `python/minisgl/scheduler/scheduler.py:Scheduler._process_one_msg`, `_free_req_resources`; `scheduler/prefill.py:PrefillManager.abort_req`; `scheduler/decode.py:DecodeManager.abort_req` | then `DetokenizeMsg(finish_reason="abort")` |
| detokenizer cleanup | `python/minisgl/tokenizer/detokenize.py:DetokenizeManager.detokenize` | `del self.status[uid]` on any finished msg |
| sockets | `python/minisgl/utils/zmq_utils.py:ZmqAsyncPushQueue`, `ZmqAsyncPullQueue`; `FrontendManager.bind` | |
| startup | `python/minisgl/server/launch.py:launch_server`, `start_backend` | waits for ready acks before `uvicorn.run` |

## This repo vs. the tutorial's reference code

All verified against the excerpts in the `_code` chapter (README "Differences" list):

* **`wait_for_ack` cleans up in `finally`** and returns right after yielding a
  finished reply. The reference deletes the maps *after* its loop, which never
  runs when the caller breaks on `finished` (lab section 3 reproduces the leak
  with a copy of the reference function).
* **Callers use `contextlib.aclosing`** (`collect`, `_stream`,
  `_stream_with_cancellation`), so the generator is closed (and its `finally`
  runs) immediately on break or exception, not whenever it is garbage-collected.
* **Non-streaming disconnects are detected**: `_collect_or_abort` runs
  `collect` as a task and polls `request.is_disconnected()`; the reference
  non-streaming path never checks.
* **`abort_user` has no `sleep(0.1)`**; it pops the maps and sends `AbortMsg`.
* **The detokenizer is told about aborts** (final `DetokenizeMsg` with
  `finish_reason="abort"`), so `DetokenizeManager.status` does not leak.
* **Errors are replies**: too long / empty prompt, tokenization failure,
  `max_tokens <= 0` → `finish_reason="error"` → HTTP 400 (non-streaming) or an
  SSE `error` event + `[DONE]`; a request with neither `messages` nor `prompt`
  gets 400 before any uid is created. The reference drops a too-long prompt and
  the client waits forever.
* **Protocol details**: `finish_reason` is `length`/`stop`/`abort` (reference:
  always `stop`), `usage` reports real token counts (reference: placeholders),
  chat chunks use `object="chat.completion.chunk"` (reference:
  `text_completion.chunk`), `/v1/completions` streams `choices[0].text`.
  Still not implemented: `n`, `stop`, model selection by `model`.
* **Not changed**: no backpressure (a slow client still lets `ack_map[uid]` grow),
  and a request whose `AbortMsg` overtakes its `TokenizeMsg` is not cancelled
  (see the last misconception).

## Common misconceptions

**Misconception:** Declaring the route `async def` makes it non-blocking.
**Actually:** Only `await` points yield. Lab section 5: `time.sleep(0.3)` inside a
coroutine opens a 311 ms gap in a 50 ms ticker; `await asyncio.sleep(0.3)` and
`await asyncio.to_thread(time.sleep, 0.3)` keep it at 50 ms.

**Misconception:** One `event.set()` = one reply.
**Actually:** Lab section 1: B received 3 replies from 3 `set()` calls but woke
up only twice (B1 and B2 arrived in one socket message).

**Misconception:** Non-streaming means the server blocks until the answer is done.
**Actually:** The coroutine awaits; other requests progress. Only the client
receives everything at once.

**Misconception:** Accepting ten HTTP requests means the GPU runs them together.
**Actually:** That depends on the scheduler (Ch6). The frontend only queues them.

**Misconception:** Seeing `finished=True` proves all state is reclaimed.
**Actually:** In the reference, the frontend maps leak after a normal finish
(break before the trailing `del`), and the detokenizer keeps aborted uids.

**Misconception:** An abort always stops the request.
**Actually:** It is a message that must find the request. If the scheduler
receives `AbortBackendMsg` before the request's `UserMsg` (possible with
`--num-tokenizers > 1`, since the frontend's PUSH socket round-robins messages
across tokenizer workers), `_process_one_msg` finds nothing and forgets the
abort; the request then runs to `max_tokens` with nobody listening (its replies
are dropped by `listen`). Scheduler-level reproduction (run from `labs/`):

```python
from _common import make_llm, make_tiny_model
from minisgl import SamplingParams
from minisgl.message import AbortBackendMsg, UserMsg
llm = make_llm(make_tiny_model(), load_tokenizer_=False)
llm.io.put(AbortBackendMsg(uid=99))   # abort overtakes the request
llm.io.put(UserMsg(uid=99, input_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=6, ignore_eos=True)))
llm.scheduler.run_until_idle()
print([m.finish_reason for m in llm.io.outbox])   # [None, None, None, None, None, 'length']
```

## Lab

```bash
python labs/part2_ch05_frontend.py      # ~4 s, no model, no sockets
```

It drives the real `FrontendManager` (and the real `listen`, `_stream`,
`_collect_or_abort`, `_stream_with_cancellation`) with in-memory stand-ins for
the two ZMQ queues and for starlette's `Request`:

```
arrival order (one row per socket message): A1 B1 B2 | A2 | B3 A3
| consumer | received (in order) | event.set() calls | wake-ups |
|        A |  ['A1', 'A2', 'A3'] |                 3 |        3 |
|        B |  ['B1', 'B2', 'B3'] |                 3 |        2 |

|            wait_for_ack | caller uses aclosing | state left after break | state left after aclose() |
|  this repo: try/finally |                  yes |                  False |                     False |
|  this repo: try/finally |                   no |                   True |                     False |
| reference: trailing del |                  yes |                   True |                      True |

|                 scenario | state left in ack_map | AbortMsg sent |                              note |
| collect() task cancelled |                 False |             - |           finally in wait_for_ack |
| non-streaming disconnect |                 False |          True |                      returns None |
|     streaming disconnect |                 False |          True | 1 chunk sent, CancelledError=True |
```

For the full path with real processes run `python -m pytest -q tests/test_server.py`
(about a minute; starts the server twice), or start the server and use `curl -N`
as in the README.

## Exercises

### Concept chapter

**1. Why is local generation not enough for other applications? Roles of client, server, HTTP.**
A Python call needs the caller in the same process with the model loaded; a web
page, phone app or another machine cannot do that, and each would otherwise load
its own copy of the weights. The server owns the loaded model and serves many
callers; the client sends a request and renders the answer; HTTP (plus the
OpenAI-style API on top) is the agreed message format between them.

**2. What do the example's method, path, headers, body contain? What do `messages` and `stream` express?**
Method `POST` (submit a task), path `/v1/chat/completions` (the endpoint),
headers `Host: localhost:1919` (destination) and `Content-Type: application/json`
(body format), body = JSON with `model`, `messages`, `max_tokens`, `stream`.
`messages` is the conversation so far as `{role, content}` items (the server keeps
no history; the client resends it). `stream` changes only how results return:
one JSON at the end (`false`) or SSE chunks as they appear (`true`).

**3. Stages from the client through the model and back; why separate network handling from computation?**
Client → API server (parse, validate, uid, submit) → tokenizer (chat template,
ids) → scheduler (queue, batch) → engine (forward, sample) → detokenizer
(incremental text) → API server (route by uid, SSE / JSON) → client.
Generation takes seconds; parsing takes microseconds. If the server computed in
the request coroutine, one long generation would block every other connection;
separating them also lets the scheduler batch requests that arrived on different
connections.

**4. Why concurrency? How do interleaved replies avoid the wrong client?**
Users arrive while others are still generating; serving them strictly in order
makes everyone wait for the longest request ahead of them. Each request gets a
uid that travels with every internal message; replies return on one shared
channel and are appended to `ack_map[uid]` and signalled through
`event_map[uid]`, and each connection reads only its own uid's list. Verified in
lab section 1 and `tests/test_server.py::test_concurrent_requests_are_isolated`
(8 concurrent requests, each equal to its offline output).

**5. HTTP concurrency vs computation parallelism vs backend batching.**
Concurrency: many unfinished requests managed at once (the frontend accepts B
while A waits). Parallelism: work executing at the same instant (tokenizer,
scheduler and detokenizer processes on different cores; TP ranks). Batching:
several requests share one forward pass (the scheduler's `Batch`). Any one can
exist without the others; e.g. the `LLM` offline API batches with no HTTP at all.

**6. What does streaming improve, and why not model computation?**
It reduces the time until the user sees the first text (perceived latency /
TTFT as experienced) and lets them read while the rest generates. The model still
runs the same prefill and the same number of decode steps; streaming only changes
when finished text is sent. Total completion time is unchanged (slightly worse,
if anything, from more writes).

**7. Which components must a cancellation traverse; why not immediate; what needs cleanup?**
API server (detect disconnect, drop per-uid state, send `AbortMsg`) → tokenizer
(convert to `AbortBackendMsg`) → scheduler (find the request in the waiting queue
or running set, remove it, free table row + KV pages; here also notify the
detokenizer) → detokenizer (drop decode state). It is not immediate because it is
a queued message: the scheduler handles it at the start of its next loop
iteration, a forward pass already running completes, and its tokens still flow
back (the frontend discards them). Cleanup: frontend `ack_map`/`event_map`
entries, the request's queue/running-set membership, its `token_pool` /
`page_table` row, its KV pages (or radix-tree locks), and `DetokenizeManager.status[uid]`.

### Code walkthrough

**1. Why can a synchronous generation function in an `async def` route block other connections? Where must it yield?**
The event loop runs one coroutine at a time on one thread and switches only at
`await`. `blocking_generate(...)` contains none, so the loop is stuck until it
returns. It must yield wherever it waits: on the backend (`await
state.send_one(...)`, `await event.wait()` inside `wait_for_ack`), or push the
blocking work off the loop (`await asyncio.to_thread(...)`, or, as here, into
other processes). Lab section 5:

```
|                             handler body | handler ms | max gap between 50 ms ticks ... (ms) |
|             time.sleep(0.3) in async def |        300 |                                  311 |
|                 await asyncio.sleep(0.3) |        302 |                                   50 |
| await asyncio.to_thread(time.sleep, 0.3) |        301 |                                   51 |
```

**2. Trace `messages`, `max_tokens`, `stream` through the route.**
Both routes call `_handle(req, request, chat)`. `messages`: in `_submit`, if
`chat and req.messages` → `[m.model_dump() for m in req.messages]` (list of
dicts) → `TokenizeMsg.text` → `TokenizeManager.tokenize_one` applies the chat
template (with `chat_template_kwargs`). `/v1/completions` uses only `prompt`.
`max_tokens`: `req.sampling_params()` → `SamplingParams.max_tokens` →
`TokenizeMsg` → `UserMsg` → `Scheduler._process_one_msg` clamps it to
`min(max_tokens, max_seq_len - input_len, capacity - input_len)` →
`Req.output_len` → `max_device_len`, checked by `can_decode`. `stream`: in
`_handle`, `True` → `StreamingResponse(_stream_with_cancellation(_stream(uid,
chat), request, uid), media_type="text/event-stream")`; `False` →
`_collect_or_abort` → one JSON body (or 400 / 499).

**3. Roles of `ack_map` and `event_map`; why is neither enough alone?**
`ack_map[uid]` holds replies not yet consumed; `event_map[uid]` wakes the
consumer. Event only: the consumer knows something arrived but not what (an
Event carries no data, and several sets collapse into one). List only: the
consumer would have to poll it in a loop (busy-wait or sleep-and-check), wasting
CPU or adding latency.

**4. Walk through wait → clear → take-and-replace → yield; why are several replies kept when several `set()`s share one wake-up?**
`await event.wait()` sleeps until the flag is set. `event.clear()` resets the
flag *before* reading, so any reply appended later sets it again and is not
missed. `pending = ack_map[uid]; ack_map[uid] = []` takes ownership of
everything that arrived so far and gives `listen` a fresh list; there is no
`await` between clear and swap, so `listen` cannot run in between. Then each
reply in `pending` is yielded. Because data lives in the list and the Event only
says "look", N `set()`s before a wake-up leave N items in the list and all N are
yielded in one round (lab section 1: B woke twice for three replies).

**5. Inject interleaved replies for A and B; verify isolation and cleanup.**
That is lab sections 1, 3 and 4: construct `FrontendManager` without `bind()`,
replace `_recv` / `_send` with in-memory queues, start the real `listen()` as a
task, create A and B with `new_user()`, run two consumers, push
`BatchMsg([A1, B1, B2])`, `A2`, `BatchMsg([B3(fin), A3(fin)])`. Asserts: A got
`[A1, A2, A3]`, B got `[B1, B2, B3]`; `ack_map == event_map == {}` afterwards; a
late reply for A is ignored; early break with `aclosing`, task cancellation,
non-streaming and streaming disconnects all leave no state and (for disconnects)
send `AbortMsg`.

**6. Locate role, incremental text, finish reason and `[DONE]` in the stream generator.**
Here: `server/api_server.py:_stream`. Role: `if first: delta["role"] =
"assistant"` (chat only, in the same chunk as the first text). Incremental text:
`delta["content"] = reply.incremental_output` (chat) or `choice["text"]`
(completions); empty increments are not sent except the role chunk. Finish
reason: after the loop, `end = {"index": 0, "finish_reason": finish_reason or
"stop"}` with the last reply's real reason, plus `usage`. `[DONE]`: `yield
b"data: [DONE]\n\n"` (also after an error chunk). Reference counterpart:
`stream_chat_completions`. Lab section 2 prints the chunks.

**7. If the caller breaks on `finished`, does `wait_for_ack` run its trailing deletions? A GPU-free check, and the guarantees still needed.**
Reference: no. Breaking leaves the async generator suspended at `yield`; when it
is closed, `GeneratorExit` is raised at that `yield` and statements after the
loop never run, so `ack_map[uid]` / `event_map[uid]` leak. This repo puts the
deletions in `finally` (runs on return, close and cancellation) and closes the
generator deterministically with `aclosing`. The GPU-free check is lab section 3,
which runs a copy of the reference function next to the real one:

```
| reference: trailing del | yes | True | True |    ← leaks even after aclose()
|  this repo: try/finally |  no | True | False |   ← held only until the generator is closed
```

Still needed for full lifecycle guarantees: cancellation must reach every stage
(here the detokenizer is told via the final `"abort"` message), every rejection
must produce a terminal reply (here `finish_reason="error"`), an abort must not be
lost if it overtakes its request (not handled; see misconceptions), and per-uid
buffers need a bound or timeout for slow clients (backpressure; not handled).
