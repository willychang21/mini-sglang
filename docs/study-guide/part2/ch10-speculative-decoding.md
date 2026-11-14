# Part II · Chapter 10 — Speculative Decoding

> Tutorial: `part2/Chapter10_Speculative-Decoding.md`, `part2/Chapter10_Speculative-Decoding_code.md` · Lab: `labs/part2_ch10_speculative.py` · Bench: `benchmarks/bench_speculative.py`

## Limitation → mechanism → validation

| Limitation | Mechanism | Validation here |
|---|---|---|
| Every output token needs one full sequential target forward; at low concurrency each step underuses the hardware | A cheap drafter proposes *k* tokens; the target scores all of them in **one** forward (k+1 rows); the accepted prefix plus one correction/bonus token is committed; both caches rewind to the confirmed path | **Lossless:** `tests/test_speculative.py` (greedy speculation == target greedy == HF for several *k*; self-draft accepts everything; n-gram drafter lossless; `rejection_commit` reproduces the target distribution). **Speed:** `benchmarks/bench_speculative.py`; README (Qwen3-0.6B, k=4, 64 tokens, CPU): n-gram on a copy-heavy prompt 61% acceptance, 2.42 tokens/target forward, **1.60x**; 14-layer truncated draft 3–5% acceptance, 0.29–0.30x; target-as-draft 100% acceptance yet 0.80–0.87x. **Mechanics:** the lab prints every round. |

## Key ideas

* **The dependency, not the cost, is the problem.** Token *t+1* is conditioned
  on token *t*. KV caching makes each step cheaper, batching runs many requests'
  steps together, TP splits one step across devices; none lets one request take
  two steps at once.
* **Guess, then check in parallel.** Once the draft tokens exist, the target can
  process them like a prompt chunk: one causal forward gives a distribution at
  each position. Row *j* is conditioned on the confirmed prefix plus the first *j*
  proposals.
* **Alignment.** With confirmed length `L` and `k` proposals, full-sequence rows
  `L-1 … L+k-2` verify the proposals and row `L+k-1` is the bonus row. In this
  repo the target cache holds `seq[:-1]`, so the step input is
  `[seq[-1]] + proposals` and its `k+1` output rows are exactly those rows.
* **Stop at the first rejection.** Later rows were conditioned on a token that
  is now false; their predictions are discarded even if they "match".
* **A round commits `a + 1` tokens**: `a` accepted proposals plus a correction
  (on rejection) or a bonus (on full acceptance), never more than the remaining
  budget, and nothing after a committed EOS.
* **Greedy verification** compares the proposal with the target's argmax. The
  output equals target-only greedy decoding (same numerics, same tie-breaking).
* **Sampling verification** accepts `y ~ q` with probability `min(1, p(y)/q(y))`
  and on rejection samples from `normalize(max(p − q, 0))`. Acceptance supplies
  `min(p, q)` and the residual fills `p − min(p, q)`, so each committed token is
  distributed exactly as `p`. It preserves the distribution, not the sample path.
* **Cache rewind.** Both target and draft caches may contain K/V of rejected
  tokens; they are truncated to the confirmed path. The correction/bonus token has
  never been run through the target and is fed as the first input of the next
  verification.
* **Fewer target forwards is not a speedup by itself.** Time per token =
  `T_round / E[committed per round]`; drafting cost, verification of `k+1`
  positions and bookkeeping must all be paid back.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| round loop (draft → verify → commit → rewind) | `python/minisgl/speculative/decoder.py:SpeculativeDecoder.generate` | prefill produces the first token from the target alone |
| target-only baseline | `speculative/decoder.py:autoregressive_generate` | same `SequenceRunner`, used for the lossless checks |
| statistics | `speculative/decoder.py:SpecStats` | `acceptance_rate`, `tokens_per_target_forward`, `accepted_per_round` |
| rewindable single-sequence KV | `python/minisgl/speculative/runner.py:SequenceRunner.extend`, `truncate`, `reset` | one page-table row with contiguous slots; `extend(..., all_logits=True)` returns a row per input token |
| all-position logits | `python/minisgl/core.py:Batch.return_all_logits`, `layers/embedding.py:ParallelLMHead.forward` | without it a prefill-phase batch keeps only the last position |
| model drafter | `python/minisgl/speculative/drafter.py:ModelDrafter.propose`, `rollback` | catches up on tokens confirmed last round (`seq[runner.length:]`), stops drafting at EOS |
| truncated draft model | `python/minisgl/config.py:EngineConfig.num_layers`, `models/weight.py:load_weight` | loads only the first N layers of the same checkpoint |
| n-gram (prompt lookup) drafter | `speculative/drafter.py:NgramDrafter.propose` | most recent earlier occurrence of the longest suffix n-gram (≤ 4); one-hot `q` for sampling |
| greedy acceptance | `python/minisgl/speculative/verify.py:greedy_commit` | |
| sampling acceptance | `speculative/verify.py:rejection_commit` | residual with a numerical fallback to `p` if its mass is ≤ 1e-12 |
| processed distributions | `speculative/verify.py:logits_to_probs` | temperature, top-k, top-p applied before verification |
| vocabulary check | `SpeculativeDecoder.__init__` | raises if target and draft vocab sizes differ |
| **batched, in the scheduler** | `speculative/worker.py:SpecWorker`, `scheduler/scheduler.py:Scheduler._prepare_verify` / `_process_verify` | verify batch = `Batch(phase="verify")`; see [speculative-in-scheduler.md](../../speculative-in-scheduler.md) |

## This repo vs. the tutorial's reference code

* The tutorial's reference mini-sglang (commit 9a91cfa) has **no speculative
  module**; its code chapter reads SGLang (`standalone_worker_v2.py`,
  `eagle_worker_v2.py`, `eagle_info.py`) and teaching fragments. This repo's
  `speculative/` package is a from-scratch implementation of those fragments:
  `greedy_commit` has the same structure as the tutorial's teaching function
  (EOS taken from a set instead of one id), `rejection_commit` implements its §5.1
  fragment for a whole round.
* **Offline, one sequence, outside the scheduler** (this chapter's code).
  `SpeculativeDecoder` drives an `Engine` directly through `SequenceRunner`.
  **Now integrated:** the same round also runs for a whole batch inside the
  continuous-batching scheduler and the server
  (`--speculative-algorithm ngram|draft_model`, `speculative/worker.py`), on
  the paged KV cache with the radix cache, retraction and aborts — see
  [speculative-in-scheduler.md](../../speculative-in-scheduler.md). SGLang verifies batches of
  requests with tree-shaped drafts, custom masks and `retrieve_*` indices
  (`EagleVerifyInput`); here the draft is linear and the mask is the ordinary
  causal one.
* Mapping: `ModelDrafter` ≈ SGLang's standalone draft worker;
  `SpeculativeDecoder.generate` ≈ `EAGLEWorkerV2.forward_batch_generation`
  (draft → verify → update draft state); the verification input is
  `[seq[-1]] + proposals` at positions `[L-1, L+k-1]`.
* Compatibility is checked only by vocabulary *size* (see code exercise 5).

## Common misconceptions

* **Misconception:** speculative decoding reduces total computation.
  **Actually:** it adds work (draft forwards, verifying rejected positions). It
  only trades that for fewer *sequential* target steps (README: target-as-draft
  has 100% acceptance and is still 0.80x).
* **Misconception:** after a rejection, later matching proposals still count.
  **Actually:** only the run of accepted proposals from the start counts; the
  round ends at the first rejection.
* **Misconception:** a proposed EOS ends the request.
  **Actually:** only a committed EOS does (lab section 4: proposals `[5, 9, 7]`,
  rows `[5, 3, …]` commit `[5, 3]` and generation continues).
* **Misconception:** on rejection, resample from `p`.
  **Actually:** that double-counts mass already given by acceptance; it must be
  the residual `max(p − q, 0)` (lab section 3: `[0.39, 0.61]` instead of
  `[0.3, 0.7]`).
* **Misconception:** sampling-mode speculation reproduces the same tokens as
  ordinary sampling with the same seed.
  **Actually:** it reproduces the *distribution*; the random numbers are consumed
  differently (draft samples, acceptance uniforms, residual samples).
* **Misconception:** the K/V computed for a rejected proposal can be reused for
  the correction token.
  **Actually:** K/V depends on the token itself; the slot is rewound and the
  correction is computed in the next verification forward.

## Lab

```bash
python labs/part2_ch10_speculative.py             # ~5 s (tiny model)
python labs/part2_ch10_speculative.py --model real --samples 2000   # ~15 s, Qwen3-0.6B
```

Look for: per round `L`, the target cache length (`L-1`), proposals vs the
target's argmax per verification row, accepted count, committed tokens, the
truncation length, and `output == target-only greedy: True` for every drafter.

```
-- draft = model: first 2 of 3 layers
| round |  L | tgt cache |       proposals |   target argmax rows | accepted |       committed | new L | tgt cache after |
|     4 | 12 |        11 | [310, 219, 189] | [310, 145, 433, 276] |        1 |      [310, 145] |    14 |              13 |
|     5 | 14 |        13 | [270, 301, 405] | [270, 301, 505, 349] |        2 | [270, 301, 505] |    17 |              16 |
...
|                      draft | rounds | proposed | accepted | acceptance | tokens/target fwd | target fwds | draft fwds |
| model: first 2 of 3 layers |     10 |       27 |        6 |        22% |              1.60 |          11 |         27 |
|       n-gram prompt lookup |     15 |        5 |        0 |         0% |              1.00 |          16 |          0 |
|       model: target itself |      4 |       12 |       12 |       100% |              4.00 |           5 |         12 |
```

In 16 tokens the tiny random model's output barely repeats its context, so
the n-gram drafter rarely proposes anything there (over 120 tokens it falls into
cycles and lookups start to hit, see `tests/test_speculative.py`). With `--model real` (a "repeat this sentence" prompt)
it is accepted 12/12, like the target itself, at zero draft cost.

## Exercises

### Concept chapter

**1. Why don't KV caching, batching and TP remove the adjacent-token dependency? How does speculation feed parallel verification?**
KV caching removes recomputation of the past, batching shares a step across
requests, TP splits a step across devices; each still needs token *t* before it
can compute the distribution of *t+1* for the same request. Speculation supplies
*guessed* tokens from a cheap drafter, so the target has concrete inputs for
positions *t+1 … t+k* and can run them in one causal forward, like a prompt
chunk. Code: `SpeculativeDecoder.generate` calls
`self.target.extend([seq[-1]] + proposals, all_logits=True)`.

**2. Draft `like green tea`; target accepts `like`, corrects `green` → `black`. Why not check `tea`?**
The row that would verify `tea` was computed for the prefix `I like green`, which
is now false. Its prediction (and the bonus row) answer a question about a
different context, so they are discarded; the round commits `like black`. The
next round drafts from `I like black`. In code, `greedy_commit` returns at the
first mismatch; the decoder truncates the target cache to `L + n_acc`. Lab
section 2 shows the same with indices: d1, d2 accepted, d3 rejected, row 7
(bonus) discarded.

**3. `p = [0.3, 0.7]`, `q = [0.6, 0.4]`.**
Acceptance: A `min(1, 0.3/0.6) = 0.5`, B `min(1, 0.7/0.4) = 1`. Rejection happens
only for A, with probability `0.6 × 0.5 = 0.3`. Residual `max(p − q, 0) = [0,
0.3]`, normalized `[0, 1]`: a rejection always becomes B. Final: A `0.6 × 0.5 =
0.3`, B `0.4 + 0.3 = 0.7` = `p`. Overall acceptance `Σ min(p, q) = 0.7`. Lab
section 3 (Monte Carlo through `rejection_commit`):

```
| token |   p |   q | accept prob | q*accept | residual | final | resample-from-p (wrong) |
|     A | 0.3 | 0.6 |         0.5 |      0.3 |        0 |   0.3 |                    0.39 |
|     B | 0.7 | 0.4 |           1 |      0.4 |        1 |   0.7 |                    0.61 |
Monte Carlo through rejection_commit (20000 draws): final = [0.299, 0.701], acceptance = 0.701
```

**4. Temperature / top-k / top-p vs the two methods; why no same-seed identity?**
They define the sampling distribution. Greedy verification applies when the
target decodes greedily (`temperature 0` or `top_k 1`, `SamplingParams.is_greedy`).
Sampling verification must use the *processed* `p` and `q`: both go through
`logits_to_probs(…, temperature, top_k, top_p)` before `rejection_commit`. The
top-k "k" is unrelated to the draft length. Same seed ≠ same tokens: the
speculative path consumes random numbers for draft sampling, acceptance tests and
residual draws, in a different order than plain sampling
(`autoregressive_generate` uses its own generator), so only the distribution of
each committed token is guaranteed equal.

**5. Where does the bonus come from; can it exceed the limit; does a rejected EOS stop?**
From the last verification row (`L+k-1`), conditioned on all *k* accepted
proposals: argmax (greedy) or a sample from `p` (sampling). It is appended only
if budget remains: `greedy_commit`/`rejection_commit` return as soon as
`len(output) == remaining`, and the drafter is asked for at most
`min(k, remaining)` proposals. A rejected EOS does not stop: the correction is
committed and generation continues; only a committed EOS stops. Lab section 4:

```
|           rejected EOS | [5, 9, 7] | [5, 3, 1, 2] |        10 |       [5, 3] | False |
|           accepted EOS | [5, 9, 7] | [5, 9, 7, 2] |        10 |       [5, 9] |  True |
| all accepted, budget 3 | [5, 6, 7] | [5, 6, 7, 8] |         3 |    [5, 6, 7] | False |
```

**6. 8 ms/step plain; 20 ms/round committing 2 vs 4 tokens.**
2 tokens: 10 ms/token, 0.8x (slower). 4 tokens: 5 ms/token, 1.6x. Break-even is
2.5 tokens per round. This is a decode-only number: TTFT/prefill, queueing,
memory taken by the draft model (fewer concurrent requests, smaller KV budget)
and the loss of benefit when high concurrency already keeps the GPU busy are not
in it, so end-to-end latency and throughput gains are smaller (lab section 5).

### Code walkthrough

**1. Prefix length 5, draft length 3: which rows verify, which gives the bonus?**
Zero-based full-sequence rows 4, 5, 6 verify d1, d2, d3; row 7 gives the bonus.
In this repo the target cache holds `seq[:4]`, the step input is
`[seq[4], d1, d2, d3]`, and its rows 0–3 are full rows 4–7 (lab section 2 checks
`allclose(full[4:], ver)`). See the module docstring of `speculative/verify.py`.

**2. Why discard later rows after the first rejection?**
Row *j* attends to the proposals before it. After rejecting d_i, every row
`> i` saw a false token in its context (causal mask hides only *later* tokens,
not earlier wrong ones), so its argmax/probabilities are for a sequence that
will never exist. Committing them would break the greedy-equality and the
distribution guarantees. Code: `greedy_commit` / `rejection_commit` return on
`token != proposed`.

**3. Section 5.1 example (`p = [0.4, 0.6]`, `q = [0.8, 0.2]`) with resampling from `p` after rejection?**
Acceptance gives `[0.8 × 0.5, 0.2] = [0.4, 0.2]`; rejection mass `0.4` resampled
from `p` adds `[0.16, 0.24]`: final `[0.56, 0.44]` instead of `[0.4, 0.6]`.
Checked in lab section 3 (`resampling from p gives [0.56, 0.44]`).

**4. A proposed EOS is rejected: what is committed, does generation stop?**
The accepted proposals before it plus the target's correction for that
position; generation continues unless the correction itself is EOS or the budget
is used. In this repo the draft also stops proposing after an EOS
(`ModelDrafter.propose`, `NgramDrafter.propose` truncate at EOS), and
`SpeculativeDecoder.generate` loops while `output[-1] not in eos`. Lab
section 4, "rejected EOS": `[5, 3]`, not finished.

**5. Why does equal vocabulary size not establish compatibility?**
Verification compares token **ids**; they must mean the same string in both
models (same tokenizer, same special tokens, same id order). Two tokenizers can
have the same size with different id → text maps, and a model can pad its
embedding beyond the tokenizer. This repo's check is only
`target.model_config.vocab_size == drafter.vocab_size` (`SpeculativeDecoder.__init__`);
it is safe in practice because its drafts are the same checkpoint truncated
(`EngineConfig(num_layers=...)`) or the n-gram drafter, which proposes the
target's own tokens. Greedy speculation would remain *correct* with an
incompatible draft (the target decides every token), just with ~0% acceptance:
`tests/test_speculative.py::test_greedy_speculation_equals_target_greedy` uses an
unrelated random 1-layer checkpoint as one of its drafts and still matches HF;
sampling would remain correct as long as `q` is a distribution over the same ids.

**6. Why can a correction token not reuse the rejected candidate's KV?**
K/V at a position is a projection of that position's own hidden state, which
starts from its own token embedding; the rejected token's K/V at that position is
for a different token (and every later position also saw the wrong token).
Here the target cache invariant is "holds `seq[:-1]`": after commit, the decoder
calls `self.target.truncate(min(L + n_acc, len(seq) - 1))`, which excludes the
correction/bonus, and the next round's `extend([seq[-1]] + proposals)` computes
its K/V into the same slot, overwriting the stale one (`SequenceRunner`:
attention reads only `[0, length)`). The draft cache is rolled back the same way
(`ModelDrafter.rollback`). The lab's "tgt cache after" column is always
`new L − 1`.
