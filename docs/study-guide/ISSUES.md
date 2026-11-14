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

## Tutorial text

### 5. Part II Ch9 code chapter: GPU count vs `--tp-size` (tutorial)

The text speaks of serving Qwen3-32B on three RTX 5090 GPUs, while the command
shown uses `--tp-size 2`. (TP size must also divide the head counts, so 3 would
not work for most models.)

### 6. Part II Ch9 concept chapter: row/column naming (tutorial)

It says weights are split "along the output (row) dimension". In the
`nn.Linear` weight `[out, in]`, splitting the *output* dimension is
**column parallel** (`layers/linear.py:ColumnParallelLinear`); the naming refers
to the math `Y = XA` with `A: [in, out]`. Mixing the two conventions is a common
source of confusion; see the Ch9 note.
