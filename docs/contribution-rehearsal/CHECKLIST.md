# Submitting the #10492 slice yourself: checklist

Everything here is for **you** (the owner) to do by hand. The rehearsal itself wrote nothing to
any `sgl-project` repository. Do the steps in order; don't skip step 1.

Patch: [`10492-joint-schema-layout.patch`](10492-joint-schema-layout.patch). Made on SGLang
`bd4783409967b186ecb95956a6ab11256ce6d296`. Background: [`README.md`](README.md).

## 0. Before anything: re-check the ground (5 min)

- [ ] Open https://github.com/sgl-project/sglang/issues/10492 and read any comments newer than 2026-11-14.
- [ ] Search open PRs for overlap. If any already moves the layout helpers or makes
      `joint_schema_head` lazy, stop and review that PR instead:
      `is:pr is:open joint_schema` and `is:pr is:open 10492` in the SGLang PR search box.
- [ ] Check that `python/sglang/srt/layers/joint_schema_head.py` on current `main` still has
      `pack_decision_layout` / `parse_decision_layout` at the top, and that `tokenizer_manager.py`
      still imports them from the head. If upstream already split them, the work is done.

## 1. Say what you'll do in the issue thread first, and wait a little

Post in #10492 **in your own words**, short, with your own numbers (re-measure first, step 4).
Something like:

> I profiled the frontend import path on a CPU box (`-X importtime`). The tokenizer manager and
> `/v1/systemone` import `layers/joint_schema_head.py` only for the plain-Python decision-layout
> helpers, and that import reaches the `layers/quantization` registry → linear → MoE →
> `torch._inductor` (~1 s for `entrypoints.systemone.serving`). I'd like to take this slice: move the
> layout helpers to a torch-free `layers/joint_schema_layout.py`, with a fresh-interpreter test. It
> doesn't overlap #30226 / #38177 / #39425 / #43485 / #49914.
>
> Heads-up on what it doesn't fix: `tokenizer_manager` still reaches the same registry via
> `managers/utils.py` → `logits_processor` and `state_capturer/base.py` → `memory_pool`. The common hub
> is `layers/quantization/__init__.py` eagerly importing every method. Would a lazy quantization
> registry be welcome as a follow-up, or is someone already on it?

- [ ] Wait 1–3 days for objections, especially from the assignee. If someone says "already doing it", don't open the PR.

## 2. Fork and branch

```bash
# on GitHub: Fork sgl-project/sglang to <you>/sglang
git clone https://github.com/<you>/sglang.git && cd sglang
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream
git checkout -b perf/joint-schema-layout upstream/main      # never work on main
```

## 3. Apply the patch and rebase onto latest main

```bash
git am --3way /path/to/mini-sglang/docs/contribution-rehearsal/10492-joint-schema-layout.patch
# If it conflicts (likely if a decision-model PR touched joint_schema_head.py):
#   resolve by keeping upstream's new code and re-doing the *move*: layout helpers to
#   joint_schema_layout.py, head-only code stays. Then: git add -A && git am --continue
git fetch upstream && git rebase upstream/main
# re-grep for any NEW importer added upstream since bd47834:
grep -rn "joint_schema_head import" python test | grep -v "JointSchemaHead\|JointSchemaPooler"
```
- [ ] Every remaining importer of `joint_schema_head` imports only `JointSchemaHead` / `JointSchemaPooler`.
- [ ] Commit author is you (`git log -1 --format='%an <%ae>'`). The commit message is in the patch; edit it
      if your wording differs. Keep `Refs #10492`, not `Fixes`: the issue stays open.

## 4. Lint, test and measure (CPU is enough)

```bash
pip install pre-commit && pre-commit install
pre-commit run --files $(git diff --name-only upstream/main...)   # run twice if it auto-fixes

python3 test/registered/unit/layers/test_joint_schema_layout.py
pytest test/registered/unit/layers/test_joint_schema_head.py \
       test/registered/unit/entrypoints/systemone/test_joint_schema.py \
       test/registered/unit/entrypoints/openai/test_serving_decisions.py \
       test/registered/unit/managers/ -q

# Import-time A/B against main (two worktrees, interleaved; see ab_import_time.py)
git worktree add ../sglang-main upstream/main
python /path/to/mini-sglang/docs/contribution-rehearsal/ab_import_time.py \
  --base ../sglang-main/python --patched ./python -n 9 \
  sglang.srt.entrypoints.systemone.serving sglang.srt.managers.tokenizer_manager sglang.srt.entrypoints.http_server
```
- [ ] Prove the new test guards something: temporarily point `tokenizer_manager`/`serving.py` back at
      `joint_schema_head`, and see `test_systemone_serving_does_not_load_model_layers` fail. Then revert.
- [ ] Paste **your** numbers and machine into the PR description. Don't reuse the rehearsal's numbers.

## 5. Coverage (CI enforces ≥ 60% on changed lines)

```bash
pytest test/registered/unit/layers/test_joint_schema_head.py test/registered/unit/entrypoints/systemone/ \
  --cov=sglang.srt.layers.joint_schema_layout --cov-report=term-missing
```

## 6. GPU checks (do these before asking for review, or say clearly you couldn't)

On one NVIDIA GPU with a full source install:
- [ ] `pytest test/registered/unit/layers/test_joint_schema_head.py -v`. The batched-head test
      (skipped on CPU: "the batched head uses FlashAttention") must run and pass.
- [ ] If you can get a Clef or Kev checkpoint: launch it, send one valid and one malformed
      `/v1/systemone` request, and check the 200 response and the same 400 error text as on main.
      If you can't get a checkpoint, write that in the PR ("validated by unit tests only; no live Clef server").

## 7. Open the PR

- [ ] Push: `git push -u origin perf/joint-schema-layout`.
- [ ] Open the PR against `sgl-project/sglang:main`. Title:
      `[Perf] systemone: Validate decision layouts without importing the joint schema head`.
- [ ] Body: start from [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md), replace the numbers, keep the
      *Scope note*, and delete the HTML comments.
- [ ] Don't spam CI. GPU CI needs the `run-ci` label from a trusted user; ask the merge oncall once.
      As the author you may comment `/rerun-failed-ci` after reading the logs (code failure vs flaky infra).

## 8. Respond to review

- [ ] Reply yourself, briefly, with evidence (Part 0 etiquette). No AI-written replies.
- [ ] Likely questions and honest answers:
  - *"No startup gain for the server?"* Correct, and say so. This removes one route. Point to the
    route map and offer the registry follow-up. If they'd rather have one PR that delivers the gain,
    offer to close this one and fold it into that.
  - *"Keep `pack_decision_layout` importable from the head?"* That conflicts with the repo's
    "no import through a re-export" pre-commit hook, so ask which they prefer.
  - *"Move it to `entrypoints/systemone/` instead of `layers/`?"* Fine either way. The head imports it
      too, so a neutral spot next to the head avoids layers → entrypoints imports. Follow their preference.
- [ ] Each new push invalidates old green CI, so re-run the tests in step 4 after every change.
- [ ] Once approved by one CODEOWNER per touched area (managers, layers, entrypoints) with green CI,
      ask the merge oncall to merge.
