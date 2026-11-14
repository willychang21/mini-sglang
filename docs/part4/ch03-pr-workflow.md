# Part IV · Ch3 — SGLang PR Workflow

Course text: `part4/Chapter3_SGLang-PR-Workflow.md` and Part 0 "Coding Ethics and Open-Source
Spirit". The full, SGLang-specific playbook (dev setup, tests, lint, CI commands, merge rules,
and five researched open issues) is in [`../contributing-to-sglang.md`](../contributing-to-sglang.md).
This page covers the chapter's key ideas and exercises.

## Key ideas

- **Problem before code.** A bug report needs a minimal repro, logs, the model, the GPU, the
  commit and the launch command, plus expected vs actual behaviour. A performance claim needs
  before/after numbers on identical conditions and a profile showing *why* it got faster.
- **Discuss substantial changes first** (feature, interface, new dependency or flag, changed
  default, refactor of a core module). Typos and clearly scoped fixes can go straight to a PR.
- **One problem per PR**, small enough to review. State what was not tested.
- Lifecycle: fork → branch from up-to-date upstream `main` → commit and open PR (title
  `[Type] scope: title`, template sections filled) → CI (GPU CI needs the `run-ci` label from a
  trusted user) → review (respond with evidence, push to the same branch, re-validate) →
  merge (CODEOWNER approval per file plus a Merge Oncall).
- Green CI ≠ mergeable. Humans still judge design and maintenance cost.
- Don't flood maintainers with unread, AI-generated PRs or comments. Part 0 calls this a "DDoS"
  on review time.

## How mini-sglang prepared you

Building mini-sglang is the "understand first" step of Part 0's *understand → ask →
contribute*. Concretely, when reading an SGLang diff you can now map it onto a component
you have written yourself (see the [source map](../sglang-source-map.md)). Each of the five
issues in the playbook is tied to a mini-sglang file, for example the logits `all_gather` in
our `ParallelLMHead` for #3365, and our `NgramDrafter` and `greedy_commit` for #50395.

## Exercises — worked answers

### 1. A real small issue → a local PR draft (not submitted)

I chose [#50395](https://github.com/sgl-project/sglang/issues/50395) (NGRAM speculative decoding
is not deterministic under `--enable-deterministic-inference`). Search done: the issue has
0 comments and no linked PR. Related checks already exist in
`python/sglang/srt/arg_groups/overrides.py` (`_deterministic_attention_backend`,
`_deterministic_allreduce_fusion_disable`). The reporter offered to write either fix, so the
right first action is to wait for the maintainer decision and not take the reporter's PR.
The draft below is for *self-review only* and assumes maintainers pick the "reject" option.

```markdown
Title: [Fix] spec: Reject NGRAM speculative decoding with --enable-deterministic-inference

## Motivation
Fixes #50395. With `--enable-deterministic-inference --speculative-algorithm NGRAM`, a greedy
request's output depends on earlier traffic: the NGRAM corpus (`cpp_ngram/ngram_corpus.py`) is
shared across requests, so the drafts change, and the deterministic extend kernel is not
invariant to the draft-tree shape. The reporter measured logprob changes in 8/8 probes and
token changes in 3/8. Deterministic mode promises batch- and history-invariant outputs, so
silently accepting this combination is a correctness bug.

## Modifications
- `arg_groups/overrides.py`: new post-process rule next to the other deterministic checks,
  raising `ValueError` when `enable_deterministic_inference` and
  `speculative_algorithm == "NGRAM"`, with a message that suggests EAGLE3 or disabling speculation.
- Unit test (CPU) for the rule: NGRAM + deterministic raises; EAGLE + deterministic and NGRAM
  alone do not.
- Docs: one sentence on the deterministic-inference page.

## Accuracy Tests
N/A: the change only rejects a configuration at startup; no forward code changes.

## Speed Tests and Profiling
N/A.

## Validation
- `python3 test/registered/unit/<path>/test_<rule>.py` → passes; fails on main for the NGRAM case.
- `pre-commit run --all-files` clean.
- Not tested: GPU end-to-end (only the startup rule changed); AMD/NPU paths.
```

Self-review against Part 0's four questions: *problem* (non-determinism, measured by the
reporter), *reproduce* (the reporter's script and command), *touched* (one arg-rule function,
one test, one doc line; not invasive), *delete test* (deleting the rule silently re-allows
the broken combination, which the unit test catches). It is small and complete, but it should only be
opened after a maintainer chooses "reject" over "document" and the reporter declines to do it.

### 2. "2× faster" after also changing the GPU and shortening the input. Does this prove the code helped? What if CI hasn't started?

No. Three variables changed at once (code, hardware, workload), so the speedup cannot be
attributed to the code. A newer GPU or a shorter input alone could explain all of it, and
shorter inputs even change which phase dominates (less prefill, so a different bottleneck).
Redo it: same GPU, same input/output lengths, concurrency, seed and warmup; only the commit
differs; ≥3 repeats with mean and range; a profile showing the targeted cost went down; and an
accuracy check if outputs can change.

If CI hasn't started, that is expected for new contributors: GPU CI requires the `run-ci` label,
which only users in `.github/CI_PERMISSIONS.json` can add. Make sure pre-commit and the relevant
unit tests pass locally, fill in the template (especially the Speed and Accuracy sections), then
politely ask the assigned Merge Oncall or a module reviewer to add the label (or use the Slack
`#pull-request` channel). Don't push empty commits or spam rerun commands. Once CI runs, read the
failure logs and tell code failures apart from infrastructure flakes before asking for a rerun
(as PR author you may use `/rerun-failed-ci` on your own PR).
