# Speculative decoding in the scheduler

Chapter 10 builds speculative decoding for **one request, offline**:
`SpeculativeDecoder` drives an `Engine` through `SequenceRunner`, which owns
one page-table row with a private, contiguous slot range, so "rolling back"
is just moving an integer. This note covers the next step, which is what
SGLang actually ships: speculation for **every running request at once**,
inside the continuous-batching scheduler, on the shared paged KV cache, with
the radix cache, retraction, aborts and the HTTP server all still working.

```
python -m minisgl --model Qwen/Qwen3-0.6B --speculative-algorithm ngram --speculative-num-draft-tokens 4
python -m minisgl --model Qwen/Qwen3-0.6B --speculative-algorithm draft_model \
    --speculative-draft-num-layers 8            # or --speculative-draft-model /path/to/small-model
LLM(path, speculative_algorithm="ngram", speculative_num_draft_tokens=4)
```

| Piece | Code |
|---|---|
| one round for a batch: plan → allocate → draft → verify → commit | `python/minisgl/speculative/worker.py:SpecWorker.prepare`, `commit` |
| what a verify batch carries (SGLang `spec_info`) | `speculative/worker.py:SpecVerifyInfo` |
| n-gram (prompt lookup) drafter | `speculative/worker.py:NgramBatchDrafter` (wraps Chapter 10's `NgramDrafter`) |
| draft-model drafter with its own KV pool | `speculative/worker.py:DraftModelDrafter`, `_create_draft_engine`; `engine/engine.py:Engine(page_table=...)` |
| verify batch (SGLang `ForwardMode.TARGET_VERIFY`) | `core.py:Batch.phase == "verify"`, `scheduler/scheduler.py:Scheduler._prepare_verify` |
| commit + replies + finishing | `scheduler/scheduler.py:Scheduler._process_verify` |
| freeing rejected proposals' pages | `scheduler/cache.py:CacheManager.free_spec_tail` |
| admission reserves k extra tokens | `scheduler/config.py:SchedulerConfig.spec_reserve_tokens`, `scheduler/decode.py:DecodeManager.estimated_inflight_tokens`, `scheduler/prefill.py:PrefillAdder._reserve` |
| flags | `scheduler/config.py:SchedulerConfig.speculative_*`, `server/args.py` |
| tests | `tests/test_spec_scheduler.py`, `tests/test_server.py::test_speculative_server_matches` |
| benchmark | `benchmarks/bench_spec_serving.py` |

## 1. One round

Prefill is untouched: a request's first token always comes from an ordinary
prefill batch (chunked or not). Speculation replaces only the **decode** step.
When the scheduler would run a decode batch, `scheduler.py:Scheduler._schedule_next_batch`
hands it to `_prepare_verify` instead:

```
for every running request (L = confirmed tokens = len(input_ids)):
    k_i   = min(k, remaining - 1)                          # plan
    ngram : proposals = lookup(input_ids)[:k_i]; k_i = len(proposals)
    k_i   = shrink until the round's pages fit             # _fit_to_pages
if every k_i == 0: run a plain decode batch instead        # nothing to verify
allocate pages for [cached_len, L + k_i)                   # one allocate_paged call
draft : proposals = draft model, k_i autoregressive steps  # writes draft K/V
token_pool[row, L : L+k_i] = proposals
verify: ONE target forward, extend [L-1, L+k_i) of every request, all logits
commit: greedy_commit / rejection_commit per request, roll back, free pages
```

The verify batch is just an extend batch: each request contributes
`[last_token, d1..dk_i]` at positions `L-1 .. L+k_i-1` on top of its cached
prefix, and `return_all_logits=True` makes the LM head return one row per
position. Requests with `k_i = 0` contribute a single row: for them this round
*is* a decode step (and their token is the "bonus" of an empty proposal). The
logits are packed request by request; `SpecVerifyInfo.num_rows` tells
`commit` where each request's `k_i + 1` rows start.

Row `j` of request `i` predicts position `L + j`, so rows `0..k_i-1` verify
`d1..dk_i` and row `k_i` gives the bonus token after a full accept — the same
alignment as Chapter 10 (`speculative/verify.py` docstring); the batched
greedy path takes one `argmax` over the whole verify batch and uses
`greedy_commit_ids`.

## 2. Invariants: which positions hold valid KV

`L` = confirmed tokens of a request. These hold at every point where anything
else in the scheduler can look at a request (admission, retraction, aborts,
`cache_req`, the next round):

| | between rounds (= after plain decode) |
|---|---|
| host tokens | `len(input_ids) == L` |
| `device_len` | `L` |
| `cached_len` | `L - 1`: K/V valid for `[0, L-1)`; the newest token has none yet |
| pages | cover `[0, ceil(cached_len / P) · P)` |

Inside one round:

| phase | target K/V valid on the confirmed path | written but not confirmed | pages held |
|---|---|---|---|
| after `prepare` | `[0, L-1)` | — | `[0, ceil((L+k_i)/P)·P)` |
| after draft (draft model) | `[0, L-1)` (target) | draft pool: `[draft_len, L+k_i-1)` | same |
| after verify forward | `[0, L+k_i)` written: `L-1` and `d1..dk_i` | `d1..dk_i` until judged | same |
| after `commit` (`n` accepted, `m` committed) | `[0, L+n)` = `[0, cached_len)` | stale K/V of rejected `d_{n+1}..` sits in slots past `cached_len` | `[0, ceil(cached_len/P)·P)` |

`commit` sets `cached_len = L + m - 1`, `device_len = L + m`. Two cases:

* a proposal was rejected or all were accepted: `m = n + 1` (accepted prefix +
  correction / bonus). The correction's K/V is **not** valid: position `L+n`
  holds the rejected `d_{n+1}`'s K/V (or nothing, for the bonus). It is
  recomputed as the first input of the next round, exactly like the newest
  token in plain decode (Chapter 10, §5.2).
* the round stopped *on* an accepted proposal (EOS or the length limit): `m = n`
  and `cached_len = L + n - 1` — position `L+n-1` has valid K/V but the
  invariant does not need it; the request finishes now anyway.

Pages past `ceil(cached_len / P)` only ever held rejected proposals (or nothing:
a draft that stopped at EOS proposes fewer than were allocated for), so
`CacheManager.free_spec_tail` returns them straight to the free list; they were
never inserted into the radix cache. The page that holds position
`cached_len` stays with the request when partially filled (page size > 1):
its stale slots are overwritten by the next round's verify, which always
starts at `cached_len`.

Because the state between rounds is identical to plain decode, nothing outside
the round needed to change:

* **radix cache**: `cache_req(finished=True)` inserts `input_ids[:cached_len]`
  with `page_table[:cached_len]` — only confirmed K/V.
* **retraction** (`_retract_one`, `release_for_retract`) and **aborts** run
  between rounds and see `cached_len = L - 1`, pages to `ceil(cached_len)`.
* **chunked prefill**: a `ChunkedReq` is never in the decode set, so it never
  speculates; its final chunk produces the first token as before.
* **EOS / max_tokens mid-round**: `greedy_commit` / `rejection_commit` stop at
  the first EOS and at `remaining`; `k_i <= remaining - 1` also guarantees the
  verify never writes K/V at or past `max_device_len` (≤ `max_seq_len`, the
  page-table width). Only the last committed token can finish a request, and
  `_process_verify` sends one `DetokenizeMsg` per committed token (the
  detokenizer and the HTTP stream already handle several tokens per step).

### KV reservation

Admission (Chapter 6) promises every running request room for the rest of its
output. A round briefly holds up to `k` extra slots per request (the rejected
proposals) before `commit` frees them, so `SchedulerConfig.spec_reserve_tokens
= k` is added per running request in `DecodeManager.estimated_inflight_tokens`
and per admitted request in `prefill.py:PrefillAdder._reserve`. With conservative
admission this makes the round always fit
(`test_conservative_tight_kv_never_retracts` asserts no retraction and no
trimming on a tiny pool). With optimistic admission the scheduler first
retracts until a *plain* decode step fits (`_retract_decode_if_needed`,
unchanged), then `SpecWorker._fit_to_pages` shrinks proposals to whatever pages
remain: losing a few proposals is far cheaper than retracting a request.
`test_tight_kv_retracts_and_matches_hf` exercises both.

SGLang does the same accounting differently: it reserves
`get_alloc_reserve_per_decode()` (2 × the per-round draft footprint, the 2×
covering overlap mode's one-round lag) in `new_tokens_required_next_decode`
and **keeps** the over-allocated slots across rounds (`kv_allocated_len >
kv_committed_len`) instead of freeing them each round
(`srt/speculative/eagle_utils.py::eagle_prepare_for_decode`). Reusing them saves allocator traffic
on GPU; freeing them keeps this implementation's between-round state
identical to plain decode, which is what lets the radix cache / retraction
code stay unchanged.

## 3. The draft model and its KV

`DraftModelDrafter` runs a second `Engine` (the draft) that has its own KV pool
but **shares the target's page table**: `Engine(cfg, page_table=target.page_table)`.
A token's draft K/V therefore lives at the same slot number as its target K/V,
and the target's allocator, radix cache, retraction and abort logic manage both
pools without knowing the draft exists. This is SGLang's design too: the
standalone / EAGLE draft worker is created with the target's
`req_to_token_pool` and `token_to_kv_pool_allocator`
(`srt/speculative/standalone_worker_v2.py::StandaloneWorkerV2.alloc_memory_pool`).

Per request the drafter tracks `draft_len`: `[0, draft_len)` holds valid draft
K/V. Each round:

1. **draft extend** (SGLang `DRAFT_EXTEND`): one batched draft forward over
   `[draft_len, L)` of every speculating request; its last row predicts `d1`.
2. **draft decode** × `k_i - 1`: feed `d_j`, predict `d_{j+1}` (a request drops
   out at EOS or at its `k_i`). These write draft K/V for `[draft_len, L+k_i-1)`
   into slots the round has already allocated — which is why the draft model
   needs pages *before* proposing (`BatchDrafter.needs_kv`), while the n-gram
   drafter proposes first and allocates only for what it found.
3. after commit, `draft_len = min(L + k_i - 1, L + n, cached_len)`: the draft's
   K/V for accepted proposals is valid; its last proposal was never fed back.

`draft_len` starts at 0 for every admitted request (including resumed
retractions and requests with a radix hit), so the first round re-runs the
draft over the whole prompt. That is a deliberate simplification: SGLang runs
the draft over the prompt in the same step as the target prefill
(`srt/speculative/eagle_worker_v2.py::EagleDraftWorker._draft_extend_for_prefill`)
and after every verify (`EagleDraftWorker._draft_extend_for_decode`), so every slot in the shared pool always holds
both models' K/V and prefix hits are reused by the draft too. Here a radix hit
only saves target work. Writing draft K/V into slots shared through the radix
cache is harmless: another request reading them computed the same token at
the same position over the same prefix, and in any case draft K/V can only
change *which* tokens are proposed — verification makes the output identical
regardless (`test_unrelated_draft_model_is_lossless` uses a random
checkpoint as the draft).

`draft_model` requires `tp_size == 1` (the draft engine would need its own TP
group). The n-gram drafter is deterministic, verification runs on
all-gathered logits, and the rejection sampler's CPU generator is seeded the
same on every rank, so `ngram` needs nothing extra for TP.

## 4. Sampling

Greedy requests commit with `greedy_commit_ids` (lossless: identical to
target-only greedy, which every test checks against HuggingFace). Sampling
requests use Chapter 10's `rejection_commit` with `p = logits_to_probs(verify
rows)` and `q` = the draft model's processed distribution, or one-hot for
n-gram proposals. Draft sampling and rejection sampling share one CPU
`torch.Generator` seeded with `EngineConfig.seed` (`SpecWorker.generator`), so
a run is reproducible (`test_sampling_is_seeded`). A batch may mix greedy and
sampling requests; each is committed by its own rule.

## 5. The overlap loop

Speculation **requires the normal loop**: the overlap loop schedules batch N+1
before batch N's tokens are known, but a round's input lengths depend on how
many proposals the previous round accepted. `Scheduler.__init__` logs a warning
and turns overlap off when speculation is on; `python -m minisgl` rejects
`--enable-overlap` together with `--speculative-algorithm`. SGLang's v2 spec
workers do support overlap, by allocating a fixed worst-case footprint per
request (the 2× reserve above) and passing accept lengths through `FutureMap`
so the next batch can be launched before the CPU knows them.

## 6. Mapping to SGLang

Checked against SGLang `main` at `bd47834` (newer than the pin used by the
rest of `docs/`; the v2 spec workers move quickly).

| mini-sglang | SGLang |
|---|---|
| `SchedulerConfig.speculative_algorithm` | `srt/speculative/spec_info.py::SpeculativeAlgorithm` (`NGRAM`, `STANDALONE`, `EAGLE`, `EAGLE3`, ...) |
| `SpecWorker` + `NgramBatchDrafter` | `srt/speculative/ngram_worker.py::NGRAMWorker.forward_batch_generation` |
| `SpecWorker` + `DraftModelDrafter` | `srt/speculative/standalone_worker_v2.py::StandaloneWorkerV2` (an `EAGLEWorkerV2` whose draft is a separate small model) |
| `SpecVerifyInfo` | `batch.spec_info`: `srt/speculative/ngram_info.py::NgramVerifyInput`, `srt/speculative/eagle_info.py::EagleVerifyInput` |
| `Batch(phase="verify", return_all_logits=True)` | `srt/model_executor/forward_mode.py::ForwardMode` `TARGET_VERIFY` (an extend mode) |
| draft extend over `[draft_len, L)` | `ForwardMode.DRAFT_EXTEND_V2`, `srt/speculative/eagle_worker_v2.py::EagleDraftWorker.draft_extend` |
| `SpecWorker.commit` | `srt/speculative/eagle_worker_v2.py::EAGLEWorkerV2.verify` → `srt/speculative/eagle_utils.py::eagle_sample` (greedy: `kernels/ops/speculative/tree.py::verify_tree_greedy`; sampling: `kernels/ops/speculative/sampling.py::tree_speculative_sampling_target_only`) |
| `CacheManager.free_spec_tail` | over-allocated slots kept across rounds (`kv_allocated_len`) |
| `spec_reserve_tokens` | `srt/mem_cache/allocation_sizing.py::get_alloc_reserve_per_decode`, `ScheduleBatch.new_tokens_required_next_decode` |
| `SpecRoundStats.accept_length` | `sglang:spec_accept_length` (`srt/observability/metrics_collector.py`) |

## 7. What SGLang does beyond this

* **Tree drafts.** EAGLE drafts `topk` branches per step and verifies a token
  *tree* (`speculative_num_draft_tokens` nodes) in one forward. Each node may
  only attend to its ancestors, so verification needs a **custom attention
  mask** (`EagleVerifyInput.custom_mask`, `srt/layers/attention/verify_mask.py`)
  and `retrieve_index / retrieve_next_token / retrieve_next_sibling` to walk
  the accepted path. Even the n-gram worker builds a tree (several matches from
  a trie over many requests). Here drafts are a single chain and the ordinary
  causal mask is the right one.
* **Draft model fed with hidden states.** EAGLE's draft head consumes the
  target's last hidden states, so the target must return them from verify and
  the draft extend runs every round; a standalone draft model (this repo's
  `draft_model`) needs only tokens.
* **CUDA graphs for verify and draft.** Verify batches have a fixed shape
  (`bs × num_draft_tokens`), so SGLang captures them
  (`srt/model_executor/runner/decode_cuda_graph_runner.py` handles
  `TARGET_VERIFY`) plus the draft decode / draft extend loops
  (`srt/speculative/eagle_draft_cuda_graph_runner.py`,
  `srt/speculative/eagle_draft_extend_cuda_graph_runner.py`). Here `GraphRunner.can_use`
  refuses verify batches; the draft engine's decode steps could use its own
  graph runner on CUDA, but that path is untested.
* **Overlap scheduling** (§5), **grammar-constrained** verification, logprobs,
  penalties ("relaxed" for speculation), and **adaptive** `k`
  (`srt/speculative/adaptive_spec_params.py`).

## 8. Results (CPU, Qwen3-0.6B)

`benchmarks/bench_spec_serving.py` (float32, 4 CPU threads, 64 output tokens,
`ignore_eos`, k = 4; "accept len" = tokens committed per request per verify
step; every speculative row produced exactly the target-only tokens):

RESULTS_TABLE

How to read it:

* **CPU decode is compute-bound, GPU decode is memory-bound.** On a GPU,
  verifying 5 positions costs about as much as decoding 1 (the weights are read
  once either way), so every accepted token is nearly free. On this CPU a
  5-token verify costs noticeably more than a 1-token decode, so speculation
  only wins when acceptance is high — the copy-heavy workload with n-gram
  lookup or a draft that agrees with the target.
* The truncated target (first 8 of 28 layers) is **not a trained draft
  model**; its proposals are rarely accepted and its forwards are pure
  overhead. A real draft (an EAGLE head, or a small model from the same family
  trained on the same data) is what makes `draft_model` pay off. The self-draft
  row is the acceptance upper bound and shows the mechanics (every proposal
  accepted) at twice the compute.
* Larger batches dilute the gain: the verify batch grows `k + 1`× in tokens
  while plain decode was already amortizing weight reads across the batch.
