# Part 0 · Lesson 0.1 — Coding Ethics and Open-Source Spirit

> Tutorial: `part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md` · Lab: none · Related:
> Part IV Ch2 (Profiling and Trace Analysis), Part IV Ch3 (SGLang PR Workflow); in this repo,
> `benchmarks/` and `tests/` as examples of the "evidence" the lesson asks for

## Key ideas

The lesson has one idea: **review is the scarce resource, so the author must make it
cheap.** AI tools made writing code nearly free. Reading and judging code still costs
a reviewer the same time it always did. Every rule below follows from that.

1. **Own every line you submit.** Where the code came from (by hand, Stack Overflow,
   an AI agent) does not matter. What matters is that you can explain it. Four
   questions to answer before opening a PR:
   - *Motivation:* what problem does it solve, and why now?
   - *Deliverable:* what changes, and how can someone reproduce the result?
   - *Footprint:* which modules and call paths does it touch, and how invasive is it?
   - *Deletion test:* pick any function in the diff. What breaks if it is removed?

   If you cannot answer one of them, the PR is not ready. Make it smaller.
2. **Write like a person.** Issues, PR descriptions and review replies should carry
   information, not filler. A good reply has a number, a condition and a pointer to
   the evidence ("decode step at batch=n: xx ms → yy ms, profiles attached, graph
   capture costs zz MB"). AI-styled padding costs more than it wastes. It makes
   reviewers suspect the whole PR was never read.
3. **Profile first.** Do the steps in this order: measure, find the bottleneck, change
   it, measure again, compare. Amdahl's law is the reason. Removing a cost that is 3%
   of end-to-end time gives at most 3%. A performance PR needs (a) before/after
   numbers, (b) a reproducible command, and (c) profiling evidence that explains *why*
   it is faster.
4. **What a bad first PR looks like:** a huge unread AI diff; a perf claim with no
   data; an unplanned refactor of a core module; a design decision nobody discussed
   (new dependency, new abstraction, changed default); many low-effort PRs at once.
5. **What a good first PR looks like:** small, complete, fixes a real existing issue,
   comes with verification. The order is *understand → ask → contribute*. Filing a
   precise issue is already a contribution.
6. **Open-source spirit.** You build on free work (PyTorch, FlashAttention, SGLang,
   open weights), so give back. Documentation, translations, reproduced benchmarks,
   and narrowing down a bug count as much as kernels do. Résumé motives are normal.
   The problem starts when the goal becomes PR count rather than solved problems.

How this connects to the rest of the course:

- "Before/after numbers + reproducible command" is the **benchmark methodology** of
  Part I Ch5. Fix the workload, warm up, and report percentiles.
- "Profiling evidence" is **Part IV Ch2**: go from symptom to hypothesis to
  verification with a trace.
- The PR mechanics are **Part IV Ch3**: fork → branch → PR → CI → review → merge. It
  adds three habits: define a clear problem with evidence, discuss substantial changes
  first (RFC/issue), keep the scope to one problem per PR.

## Where in mini-sglang

This lesson has no code. The repo still shows the habits it asks for, and you can
copy them in your own PRs:

| Habit | Example in this repo | Note |
|---|---|---|
| Reproducible performance claim | `benchmarks/bench_kv_cache.py`, `bench_batching.py`, `bench_serving.py` + README tables | each table names the model, dtype, hardware and workload |
| Correctness evidence | `tests/test_model.py`, `tests/test_real_model.py` | output compared token by token against HuggingFace |
| Claims that check themselves | `labs/*.py` (`assert` on every printed claim), `tests/test_labs.py` | a lab that runs to the end has verified what it prints |
| Documented deviations | `README.md` "Differences from the reference implementation" | says what was changed and why, instead of hiding it |

## Common misconceptions

**Misconception:** Using AI to write a PR is against open-source etiquette.
**Actually:** The lesson says plainly that it is fine. What is not fine is
submitting code you have not read and cannot defend.

**Misconception:** "12% faster" is a complete performance PR.
**Actually:** Without the command, the workload, the hardware and the profile, the
number cannot be reproduced. It also does not show that the change is the cause
(see Part IV Ch3 exercise 2: a speedup measured after also changing the GPU and the
input length proves nothing).

**Misconception:** A big PR shows more effort and is more welcome.
**Actually:** Review cost grows faster than diff size. One focused change with a
regression test is the preferred first contribution.

**Misconception:** Only code counts as contribution.
**Actually:** Write-ups of pitfalls, docs, translations, reproduced benchmarks, and
minimal reproductions of issues are explicitly valued.

## Lab

None. The checklist below is the practical part.

### Checklist for a first SGLang PR

Before writing code

- [ ] I found (or opened) an issue that describes a real problem. I searched existing
      issues and PRs for duplicates.
- [ ] For anything beyond a typo or clearly small fix (new feature, interface change,
      new dependency, changed default, larger optimization), I proposed the approach in
      an issue/RFC and got a maintainer's direction first.
- [ ] I can reproduce the problem. I have recorded the model, GPU, SGLang version or
      commit, launch command, and expected vs. actual behaviour.
- [ ] For a performance change, I profiled first and know what fraction of end-to-end
      time the target code takes under a real workload.

While writing

- [ ] One problem per PR. No drive-by reformatting or unrelated refactors.
- [ ] I have read every line of the diff, including AI-generated parts, and can pass
      the deletion test for each function.
- [ ] For a bug fix, I added a regression test that fails before and passes after.
- [ ] I ran the relevant tests and the required formatting and lint checks (SGLang
      contribution guide).

The PR description

- [ ] It answers three questions: what was wrong, what changes, and what evidence
      supports it.
- [ ] Performance: before/after numbers with identical hardware, model, input/output
      lengths and concurrency; the exact benchmark command; a profile or trace
      explaining *why* it is faster; an accuracy check if outputs can change.
- [ ] It states what was **not** tested (e.g. "no GPU available; docs build only").
- [ ] It is written in my own words and kept short.

During review

- [ ] Replies contain data or code, not filler. Disagreements are argued with
      measurements.
- [ ] Revisions are pushed to the same branch, and validation is re-run after new
      commits.
- [ ] I read CI failures and tell code failures from environment problems before
      retrying. I ask a maintainer when a GPU CI job needs the `run-ci` label.

Good first contributions (Part IV Ch3): fix a doc, typo or broken link; fix an
outdated Cookbook command after actually running it; reproduce and narrow down an
issue without fixing it; make a small bug fix with a regression test.

## Exercises

The lesson has no exercises. Two self-checks that use this repo:

1. *Pick any function in `python/minisgl/scheduler/prefill.py` and do the deletion
   test.* Example: delete the second capacity check in
   `PrefillAdder._try_allocate_one` (the one after `cache_manager.lock(handle)`). Locking a matched
   prefix moves it from evictable to protected and shrinks `available_size`.
   Without the second check, a request could be admitted on capacity that disappears
   once its own prefix is locked, and a later `allocate_paged` would fail with
   "out of KV cache". If you can produce answers like this for every function, you
   understand the change.
2. *Rewrite a one-paragraph claim as a reviewable statement.* "Continuous batching
   makes it much faster" becomes: "`benchmarks/bench_batching.py`, Qwen3-0.6B fp32,
   4-core CPU, 16 requests: wall 46.5 s → 13.4 s, mean TTFT 22.93 s → 2.01 s, outputs
   identical" (numbers from `README.md`).
