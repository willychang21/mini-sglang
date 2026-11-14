# Contributing to SGLang: a playbook

Sources, read in this session: SGLang `c5d7c59` (`docs/docs/developer_guide/contribution_guide.mdx`,
`.github/pull_request_template.md`, `.github/MAINTAINER.md`, `.pre-commit-config.yaml`,
`test/registered/unit/README.md`), zero-to-sglang Part 0 "Coding Ethics and Open-Source
Spirit" and Part IV Ch3. Issue status is **as read on 2026-11-14** and will drift. Re-read
each issue before starting.

> Ground rule (Part 0): you must be able to answer, for every line you submit, *what problem it
> solves, how to reproduce the result, what it touches, and what happens if it is deleted.*
> AI-assisted code is fine; unreviewed AI code and AI-written review replies are not.

## 1. Dev environment

```bash
# 1. fork on GitHub, then
git clone https://github.com/<you>/sglang.git && cd sglang
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream && git checkout -b fix/<short-topic> upstream/main   # never work on main

# 2. Python >= 3.11 venv
uv venv -p 3.11 .venv && source .venv/bin/activate
# GPU box: install from source as in docs/docs/get-started/install.mdx ("From source" tab)
#   (editable install of python/ with the right CUDA extras; check that page for the exact command at your commit)
# CPU-only box (enough for unit tests / docs / Python-only fixes): editable install of python/,
#   then run the unit tests below; GPU-only kernels and e2e tests won't run.

# 3. hooks
pip install pre-commit && pre-commit install
pre-commit run --all-files        # run twice if the first run auto-fixes files
```

A Docker-based setup is in `docs/docs/developer_guide/development_guide_using_docker.mdx`.

## 2. Repository layout (what you will touch)

| path | contents |
|---|---|
| `python/sglang/srt/` | the runtime. See [`sglang-source-map.md`](sglang-source-map.md) for module ↔ mini-sglang |
| `python/sglang/srt/managers/` | `scheduler.py`, `scheduler_components/`, `schedule_policy.py`, `schedule_batch.py`, `tokenizer_manager.py`, `detokenizer_manager.py` |
| `python/sglang/srt/mem_cache/` | allocators, KV pools, radix caches, HiCache (start with its `README.md`) |
| `python/sglang/srt/layers/` | attention backends, linear/TP, MoE, quantization, sampler, logits |
| `python/sglang/srt/models/` | one file per architecture |
| `python/sglang/srt/arg_groups/`, `server_args.py` | CLI flags and post-processing rules |
| `python/sglang/kernels/` | JIT and AOT (`sglang-kernel`) kernels |
| `python/sglang/benchmark/` | `serving.py`, `one_batch.py`, `offline_throughput.py` |
| `test/registered/` | CI-registered tests; `unit/` mirrors `srt/` and needs no GPU or server |
| `test/manual/` | tests not run in CI |
| `docs/` | Mintlify docs (docs/docs/*.mdx) and the Cookbook (`docs/cookbook/`) |
| `.github/` | PR template, CODEOWNERS, MAINTAINER.md (merge oncalls), workflows |

## 3. Tests for one component

Unit tests live in `test/registered/unit/<mirror of srt path>/`. Each file registers itself
with CI at the top, uses `CustomTestCase`, and has a `__main__` entry:

```python
from sglang.test.ci.ci_register import register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
import unittest
from sglang.test.test_utils import CustomTestCase
class TestX(CustomTestCase): ...
if __name__ == "__main__":
    unittest.main()
```

```bash
python3 test/registered/unit/mem_cache/test_radix_cache_unit.py     # how CI runs one file
pytest test/registered/unit/mem_cache/ -v                           # one module
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu           # one CI suite
pytest test/registered/unit/ --cov --cov-config=.coveragerc --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main --fail-under=60   # >=60% on changed lines
```

E2E tests (launch a server) go under the matching `test/registered/<area>/`. Keep each file
under 500 s, and reuse a server across test methods. test/srt/ is retired; a pre-commit hook
rejects it.

## 4. Lint and format

From `.pre-commit-config.yaml`: isort, ruff (`F401, F821, F823, UP037` with `--fix`),
ruff-format, codespell, clang-format (C++/CUDA), nbstripout, file hygiene hooks (trailing
whitespace, EOF, YAML/TOML/AST checks, files ≤1.5 MB, private-key detection,
`no-commit-to-branch`), and local hooks forbidding test/srt/ and `docs_new/`. Docs links are
checked by Mintlify in CI (`cd docs && mint broken-links --check-anchors --check-redirects`).

Code style rules worth memorising (contribution guide):
- Minimise device syncs (`.item()`, `.cpu()`, `.tolist()`) on the critical path. Cache per-layer
  constant checks as booleans in `__init__`.
- Functions under ~100 lines, files under 2,000 lines. No new mixins. Use `msgspec.Struct` for new
  data containers. Prefer keyword arguments.
- Never `pickle.loads` network data.
- New hardware or features go in new files. The common (NVIDIA/existing) path is the first branch.

## 5. PR template, title, CI, merge

**Title:** `[Type] scope: title` or `[Type] [Hardware] scope: title`. Type is one of `Feat, Fix,
Refactor, Perf, Doc, Test, CI, Chore`. Example: `[Fix] loader: Strip target-only extra config for draft loader`.

**Body** (the `.github/pull_request_template.md` headings): *Motivation*, *Modifications*, *Accuracy
Tests* (if outputs can change), *Speed Tests and Profiling* (if speed changes: before/after,
command, trace), *Checklist* (pre-commit, unit tests, docs, benchmarks, style).

**CI:** GPU CI runs only with the `run-ci` label, which only trusted users (listed in
`.github/CI_PERMISSIONS.json`) can add. As the author you can comment `/rerun-failed-ci` on your
own PR; `/run-cpu-ci`, `/rerun-test` and `/rerun-group` need permissions. Don't spam reruns.
Read the logs and separate code failures from flaky infrastructure first.

**Merge** (`.github/MAINTAINER.md`): a bot assigns a Merge Oncall and GitHub requests
CODEOWNERS reviews. You need one CODEOWNER approval per modified file plus green required CI;
then someone with write permission merges. Ping the oncall listed for your module if the PR stalls.

## 6. Review etiquette (Part 0 + Part IV Ch3)

1. **Problem first.** Bugs: minimal repro, logs, model, GPU, commit, launch command, expected vs
   actual. Performance: profile before and after on the same hardware, model and workload.
2. **Discuss before building** anything that adds a flag, a dependency, an abstraction, changes a
   default, or refactors a core module (open or comment on an issue/RFC). Typos and clear small fixes can go straight to a PR.
3. **One problem per PR.** No drive-by reformatting. Small enough that a reviewer can hold it in their head.
4. **Claim before you start, and check for existing claims.** Most `good first issue` items in
   SGLang have several claimants and open PRs (see §7). Reviewing or testing an existing PR is
   a contribution too, and a duplicate PR wastes reviewer time.
5. **Write replies yourself, briefly, with evidence** ("tried X; decode at bs=8 went 9.1→8.7 ms;
   traces attached"), not AI boilerplate.
6. **New commits invalidate old green CI.** Re-validate after each push. Passing CI is not approval.
7. **Say what you did not test** (e.g., "validated on 1×4090 only; not tested with DP attention").

## 7. Five real open issues, with plans

How these were chosen: the open `good first issue` list had 46 items on 2026-11-14 and none
created since August 2026. I read the full threads of 12 of them (the ones closest to this
course's topics) plus 9 recent bug reports, 21 issues in total. **None of the 12 good-first-issue
threads is cleanly unclaimed**: each has claimants or open PRs, and every recent bug I read
already came with the reporter's own fix PR. The five below are the
ones where a newcomer still has a concrete, non-duplicative next step, and each connects to
something already built in mini-sglang. File paths were verified at `c5d7c59`.

### Issue 1 — #10492 Reducing the time to import sglang
- **Link:** https://github.com/sgl-project/sglang/issues/10492 (labels: good first issue, high priority)
- **Status:** open, assigned to its original volunteer. One slice is in an open PR (#49914:
  `arg_groups/model_overrides/ernie55.py` pulling `transformers` into `server_args`, 3.7 s → 1.8 s), and other
  slices appear only as fork commits. A July 2026 comment proposes small incremental PRs, one module at a time,
  using the existing `LazyImport`.
- **What it needs:** find modules that are imported eagerly on the `Scheduler` import path but only used
  on some code paths, and defer them. Each PR should be backed by `-X importtime` numbers.
- **Files:** `python/sglang/_lazy_import.py` (`LazyImport`), `python/sglang/srt/managers/scheduler.py`
  (import block), `python/sglang/srt/server_args.py`, `python/sglang/srt/arg_groups/`.
- **mini-sglang link:** `server/launch.py` spawns schedulers with `mp.get_context("spawn")`, so every
  child process re-imports the world. Import time is paid once per TP rank, per tokenizer and per detokenizer.
- **Plan:**
  1. On a CPU box, measure: `python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler" 2> imp.log`,
     then sort by cumulative time (e.g. with `tuna imp.log` or a 10-line script).
  2. Pick **one** heavy module that no one has claimed: check #49914, the thread, and open PRs that mention #10492.
  3. Comment on the issue with the measurement and the module you will take, and wait briefly for objections.
  4. Make it lazy with `LazyImport` or a function-local import. Make sure no import-time side effects
     (registrations) are lost: grep for `register_` in that module.
  5. Re-measure (median of 5 runs), and run the unit tests of the touched package.
  6. PR `[Perf] startup: Defer <module> import`, with a before/after table and the command.

### Issue 2 — #50206 Target GDS option leaks into a different draft loader
- **Link:** https://github.com/sgl-project/sglang/issues/50206
- **Status:** open, 0 comments. The reporter linked fix PR #50207 (open).
- **What it needs:** when the target uses `--load-format fastsafetensors` with
  `--model-loader-extra-config '{"enable_gds": ...}'` and the draft uses
  `--speculative-draft-load-format safetensors`, validation fails with
  `ValueError: Unexpected extra config keys for load format LoadFormat.SAFETENSORS: {'enable_gds'}`.
  The draft's load config must drop target-only keys.
- **Files (verified):** `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py::build_load_config`
  forwards `get_model().model_loader_extra_config` regardless of the `load_format` argument.
  `python/sglang/srt/model_loader/loader.py::_validate_default_loader_extra_config` rejects it.
  The draft format is read in `ModelRunner` (`model_runner.py`, `get_spec().speculative_draft_load_format`).
- **mini-sglang link:** our draft model is built from the same `EngineConfig` with `num_layers=N`
  (`config.py`, `models/weight.py` skips layers ≥ N). This is the same "derive a draft config from
  the target config" pattern, and the same class of bug.
- **Plan (non-duplicative):**
  1. Reproduce on CPU with the reporter's script (`CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python reproduce.py`,
     calling `build_load_config` + `_validate_default_loader_extra_config`).
  2. Check out PR #50207 and run the repro plus `test/registered/unit/` tests for `model_loader/`.
  3. Review it: are shared keys kept? Do unknown keys still fail? Is there a unit test that fails before the fix?
  4. If a test is missing, offer one in a review comment (or a follow-up PR only if the author agrees).
  5. Only if #50207 stalls for weeks: ask in the issue before opening anything yourself.

### Issue 3 — #50395 NGRAM speculative decoding is not deterministic with `--enable-deterministic-inference`
- **Link:** https://github.com/sgl-project/sglang/issues/50395
- **Status:** open, 0 comments, no PR. The reporter asks maintainers to choose: *reject* NGRAM in
  deterministic mode, or *document* it, and offers a PR for either.
- **What it needs:** first a maintainer decision. Then either a startup check that raises a
  clear error or warns, or a docs note. Root-causing the kernel non-determinism is a larger,
  separate task.
- **Files (verified):** `python/sglang/srt/speculative/ngram_worker.py::NGRAMWorker`,
  `python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py::NgramCorpus` (the corpus is shared
  across requests, so a request's drafts depend on earlier traffic), and
  `python/sglang/srt/arg_groups/overrides.py`, where the other `enable_deterministic_inference`
  compatibility checks live (`_deterministic_attention_backend`, `_deterministic_allreduce_fusion_disable`).
- **mini-sglang link:** our `speculative/drafter.py::NgramDrafter` builds its table from the
  *current* sequence only. With greedy verify (`verify.py::greedy_commit`), output is identical to
  plain decoding no matter what is drafted, *if* the verify forward is numerically identical. That
  is exactly what breaks here: SGLang's batched tree-verify kernel changes reductions depending on
  how many draft tokens sit between ancestors, so a different draft gives a slightly different
  logit and, eventually, a different token.
- **Plan:**
  1. Reproduce on 1 GPU with the reporter's command (`--enable-deterministic-inference
     --attention-backend triton --speculative-algorithm NGRAM`, Qwen3-0.6B) and `repro_ngram_det.py`.
  2. Confirm the control: no speculation and EAGLE3 are bit-identical.
  3. Wait for or politely ask for the maintainer decision (the reporter has dibs on the PR). If the
     reporter does not follow up, offer the smaller option.
  4. For "reject": add a post-process rule next to the existing deterministic checks in
     `arg_groups/overrides.py` raising `ValueError` for NGRAM + deterministic, plus a CPU unit test of the
     rule. For "document": a note in the deterministic-inference and speculative-decoding docs pages.

### Issue 4 — #3365 Use `gather` instead of `all_gather` for logits
- **Link:** https://github.com/sgl-project/sglang/issues/3365 (labels: good first issue, help wanted)
- **Status:** open, assigned to a maintainer. A Feb 2025 volunteer concluded SGLang needs a
  token *broadcast* step first, with no PR since. Two recent offers (Jul and Oct 2026) are unanswered,
  and one proposes an opt-in first step with parity tests.
- **What it needs:** gather logits only to the TP entry rank, sample there, then broadcast
  the sampled token ids (B ints instead of B × V floats). This touches sampling semantics in
  every TP rank, so it is design work: ask before coding.
- **Files (verified):** `python/sglang/srt/layers/logits_processor.py::LogitsProcessor`
  (`do_tensor_parallel_all_gather`, `_get_logits`), `python/sglang/srt/layers/sampler.py::Sampler`,
  `python/sglang/srt/distributed/communication_op.py` (`tensor_model_parallel_all_gather`,
  `tensor_model_parallel_gather`, `broadcast_tensor_dict`).
- **mini-sglang link:** exactly our `layers/embedding.py::ParallelLMHead`, which calls
  `all_gather_last_dim`, and our `engine/sample.py` comment: *"Every TP rank uses the same seed
  and sees identical (all-gathered) logits, so they all sample the same token without extra
  communication."* All-gather buys "no broadcast" plus deterministic agreement. Gather-plus-broadcast
  saves bandwidth (V=151,936 floats per token → 1 int) but adds a latency step and a rank-0
  dependency for grammar, logprobs and penalties.
- **Plan:**
  1. Quantify first. With `labs/part3_ch04_parallel_comm.py`-style math, all-gather of B × V fp32 logits for
     B=64 is ~39 MB gathered per step. Measure the logits all-gather time in a TP 8 trace on real
     hardware. If it is under ~1% of the step, write that in the issue and stop.
  2. If it is worth it: comment with the numbers and a design (opt-in flag, rank-0 sampling, a token
     broadcast reusing the existing broadcast utilities, and what happens to logprobs, grammar and
     penalties), and ask the assignee to confirm the scope.
  3. Implement behind a flag with parity tests: identical greedy outputs and identical seeded sampling
     vs all-gather on TP 2/4.
  4. PR `[Perf] logits: Gather logits to entry rank and broadcast tokens`, with traces and throughput at several batch sizes.

### Issue 5 — #9889 Remove `torch.nonzero()` where appropriate
- **Link:** https://github.com/sgl-project/sglang/issues/9889 (labels: good first issue, performance)
- **Status:** open. Linked open PRs #46866 and #46333. The thread disputes the premise: the named
  call sites operate on CPU tensors (no device sync), and one function was thought unused but is
  now called again. A June 2026 comment says remaining sites should only change with profiling evidence.
- **What it needs:** *profiling-backed* evidence that a specific `nonzero()` causes a host-device
  sync on a hot GPU path, before changing it. Most of the value now is an audit, not an edit.
- **Files (verified):** `python/sglang/srt/configs/janus_pro.py` (`image_token_mask.nonzero()`, line 479),
  `python/sglang/srt/models/phi4mm_utils.py` (line 75), `python/sglang/srt/models/gemma3_mm.py`
  (`prepare_attn_masks`, line 210).
- **mini-sglang link:** our `kvcache/radix_cache.py::_fast_compare_key` uses `torch.nonzero(...)` plus
  `.item()`. That is harmless because radix keys are CPU tensors, the same "is it actually a device
  sync?" question as this issue. Part 0 §3: *profile first; a 3% hotspot gives at most 3%*.
- **Plan:**
  1. Read both open PRs and their reviews first, and don't duplicate them.
  2. `grep -rn "\.nonzero(" python/sglang/srt` and, for each hit, decide from the code whether the
     tensor is on the GPU on the serving path.
  3. For GPU-side hits on a hot path, capture a trace (Ch2) showing a `cudaStreamSynchronize` /
     `aten::nonzero` stall. Only those are candidates.
  4. Post the audit table in the issue (site → device → hot? → evidence). That alone helps maintainers close it.
  5. If one real hotspot remains, make a one-site PR with before/after trace and benchmark numbers.

### Considered and not chosen (all checked on 2026-11-14)
#28808 (TRTLLM backend refactor: several claimants, open PR #49594), #12562 (both TODOs claimed),
#5979 (open PR #34714), #6357 (open PR #38228), #3642 (assigned; reportedly fixed by #9391),
#5948 (docs follow-up claimed; whether a startup flag is still needed is unresolved), #18427
(docs dedup largely done), #6622 (assigned, PRs), #19612 (largely superseded), #50221, #50137,
#49550 and #50100 (each already has the reporter's fix PR).

**None of these issues was commented on, and no PR or other write was made to any SGLang repository in this work.**
