# Issues found while writing the study guide

Bugs and tutorial/repo mismatches found while answering the exercises. The study
guide does **not** change engine code (`python/minisgl/`); each entry has a
reproduction so the owner of that code can decide. Reproductions run from the
repository root unless stated otherwise and use the tiny lab model.

Status legend: **bug** (wrong or fragile behaviour), **gap** (known limitation,
also present in the reference), **tutorial** (problem in the tutorial text).

## Engine

### 1. An abort that overtakes its request is lost (bug; reference has it too)

`Scheduler._process_one_msg` (`scheduler/scheduler.py`) silently drops an
`AbortBackendMsg` whose uid is not pending or running yet. With
`--num-tokenizers > 1`, the API server's PUSH socket round-robins messages over
the tokenizer workers, so a client that disconnects right after sending can have
its `AbortMsg` delivered by a different tokenizer *before* the `TokenizeMsg`. The
request then runs to `max_tokens` with nobody listening (the frontend already
dropped the uid; `FrontendManager.listen` ignores the replies).

```bash
cd labs && python -c "from _common import make_llm, make_tiny_model; from minisgl import SamplingParams; \
from minisgl.message import AbortBackendMsg, UserMsg; llm=make_llm(make_tiny_model(), load_tokenizer_=False); \
llm.io.put(AbortBackendMsg(uid=99)); \
llm.io.put(UserMsg(uid=99,input_ids=[1,2,3],sampling_params=SamplingParams(max_tokens=6,ignore_eos=True))); \
llm.scheduler.run_until_idle(); print([m.finish_reason for m in llm.io.outbox])"
# -> [None, None, None, None, None, 'length']   (expected: a single 'abort', no compute)
```

Possible fix: remember recently aborted unknown uids (bounded set) and reject a
later `UserMsg` with that uid. Chapter note: Part II Ch5 (misconceptions).

### 2. Out-of-vocabulary token ids crash the scheduler (bug)

`_process_one_msg` validates length but not token-id range. A prompt id
`>= vocab_size` raises `IndexError` in `VocabParallelEmbedding.forward` inside
the scheduler loop. Over HTTP, `/v1/completions` accepts `prompt: List[int]`
(`server/api_server.py:OpenAICompletionRequest`), so one malformed client request
kills the scheduler process and with it the server for everyone.

```bash
cd labs && python -c "from _common import *; quiet_logs(); from minisgl import SamplingParams; \
make_llm(resolve_model('tiny'), load_tokenizer_=False).generate([[600]], SamplingParams(max_tokens=2))"
# -> IndexError: index out of range in self   (tiny vocab is 512)
```

Possible fix: reject with `_reply_error` when `max(input_ids) >= vocab_size` or
`min(input_ids) < 0`.

### 3. No backpressure for slow streaming clients (gap; reference has it too)

`FrontendManager.ack_map[uid]` grows without bound if a client reads slower than
the engine produces. The tutorial mentions this; this repo does not address it.

### 4. Matching mutates the tree even when admission fails (gap; same as reference)

`RadixPrefixCache._tree_walk` splits nodes and refreshes LRU timestamps during
`match_prefix`, even when `PrefillAdder._try_allocate_one` then rejects the
request for lack of KV. Harmless for correctness (a split preserves contents),
but a request that is retried every step keeps its prefix "recently used". Lab:
`python labs/part2_ch08_radix_tree.py` section 7 shows a re-match splitting a
node.

### 5. KV reservation is 2 tokens per request more conservative than the reference (mismatch, not a bug)

This repo reserves `max_device_len - cached_len + page_size` per running request
(`scheduler/decode.py:DecodeManager.inflight_tokens`) and adds `+ page_size` to
both `estimated_len` and the `reserved_size` increment
(`scheduler/prefill.py:PrefillAdder._try_allocate_one` / `_add_one_req`). The
reference at 9a91cfa reserves `remain_len + (page_size - 1)` per running request.
Between decode steps `device_len = cached_len + 1`, so this repo reserves
`remain_len + 1 + page_size`, i.e. two tokens more per request. It is safe (it
over-reserves), but numbers in Ch6 exercise 5 differ by the extra page:

```bash
python labs/part2_ch06_batching_timeline.py   # section 7: 1301 <= 1300 rejected
                                              # where the chapter's 600+200+500 = 1300 <= 1300 admits
```

### 6. `CacheManager.check_integrity` is only meaningful when idle (gap)

It counts `free pages + pages in the prefix cache == num_pages`; pages held by
running requests (not yet inserted) are not counted, so it raises mid-run. The
scheduler never calls it (tests call it after `generate`). The reference calls it
in its idle loop; a debug flag to do the same here would catch leaks earlier.

## Tutorial text

### 7. Part II Ch9 code chapter: GPU count vs `--tp-size` (tutorial)

The text speaks of serving Qwen3-32B on three RTX 5090 GPUs, while the command
shown uses `--tp-size 2`. (TP size must also divide the head counts, so 3 would
not work for most models.)

### 8. Part II Ch9 concept chapter: row/column naming (tutorial)

It says weights are split "along the output (row) dimension". In the
`nn.Linear` weight `[out, in]`, splitting the *output* dimension is
**column parallel** (`layers/linear.py:ColumnParallelLinear`); the naming refers
to the math `Y = XA` with `A: [in, out]`. Mixing the two conventions is a common
source of confusion; see the Ch9 note.

### 9. Part II Ch7: block table granularity (tutorial vs. mini-sglang)

Ch7 describes a block table with one entry per *logical block* (plus a CPU copy)
and blockwise online-softmax attention. mini-sglang (reference and this repo)
stores one *token slot* per position in a device-only `page_table`; this repo's
attention gathers K/V through it and runs one padded SDPA call. Equivalent
results, different data structure; see the Ch7 note.
