# Contribution rehearsal: one slice of SGLang #10492 (import time)

This is a full rehearsal of an SGLang contribution, done in a local clone and **never submitted**.
Nothing was pushed, commented, or opened on any `sgl-project` repository. The owner submits it
(or doesn't) by following [`CHECKLIST.md`](CHECKLIST.md).

| file | what |
|---|---|
| [`10492-joint-schema-layout.patch`](10492-joint-schema-layout.patch) | `git format-patch` of the SGLang commit (applies cleanly on the pinned commit) |
| [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) | PR body in SGLang's template, ready to adapt |
| [`CHECKLIST.md`](CHECKLIST.md) | exact steps to submit, plus what to say in the issue thread first |
| [`ab_import_time.py`](ab_import_time.py) | the interleaved A/B import-time benchmark used below |

**Pinned SGLang commit:** `bd4783409967b186ecb95956a6ab11256ce6d296` (main, 2026-11-13).
All `file:line` references below are at this commit. Issue and PR status was checked on
2026-11-14 and will drift.

---

## 1. Which issue, and why

Section 7 of [`contributing-to-sglang.md`](../contributing-to-sglang.md#7-five-real-open-issues-with-plans)
lists five candidates. The requirement here: the work must be **genuinely doable and verifiable
on a CPU-only box**.

| issue | CPU-verifiable? | decision |
|---|---|---|
| **#10492** import time | **Yes.** `-X importtime` and wall-clock import timing are hardware-independent in what they show (which modules load, and what they cost relative to each other). | **Chosen** |
| #50206 GDS option leaks into draft loader | The repro is CPU-only, but fix PR #50207 by the reporter is already open. The right contribution is a *review* of #50207, not a second patch. | Not chosen: would duplicate |
| #50395 NGRAM nondeterminism | Repro needs a GPU, and the fix waits on a maintainer decision (reject vs document). The reporter has dibs. | Not chosen |
| #3365 gather vs all_gather | Needs TP ≥ 2 on GPUs to measure and test; design work to agree with the assignee first. | Not chosen |
| #9889 remove `torch.nonzero` | Whether a site syncs the device can only be shown with a GPU trace. The thread says not to change sites without profiling evidence. | Not chosen |

I did only one issue on purpose. The second CPU-feasible item (#50206) is a review task, and
one well-evidenced slice is worth more than two thin ones.

### Status of #10492 when checked (2026-11-14)
- Open, labels `good first issue` and `high priority`. Assigned to its original volunteer, who
  has not posted since Oct 2025. A Jul 2026 comment proposes small incremental PRs, one module at a time.
- Open PRs that already take slices (read before choosing, so as not to duplicate them):
  - #30226: top-level `import sglang` (`sglang/__init__.py`, `lang/`, HF patches).
  - #38177: torchvision `decode_jpeg`, the openai SDK via `entrypoints/openai/protocol.py`, and fastapi in the encoder receiver.
  - #39425: `compressed_tensors` in the quantization registry.
  - #43485: video decoder in `srt/utils/common.py`.
  - #49914: `transformers` reached through `arg_groups/model_overrides/ernie55.py`.
- **No open PR touches `layers/joint_schema_head.py` for import reasons.** A search for open PRs
  mentioning `joint_schema` / `systemone` finds only feature work (see *Review risks*).

## 2. How the slice was found (CPU profiling)

Environment: 4-vCPU Intel Xeon @ 2.30 GHz, Python 3.13.18, torch 2.15.0+cpu, torchvision
0.29.2+cpu, transformers 5.19.0, triton 3.9.0. SGLang was run from source (`PYTHONPATH=python`)
with only the Python dependencies these imports need. GPU-only packages were not installed (see
*What is not verified*).

```bash
PYTHONPATH=python python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler" 2> sched.log
# then sort by cumulative time and walk each heavy module's parent chain
```

What the profile showed, and why most of it was **not** a good newcomer slice:

| finding | cost here | why not chosen |
|---|---|---|
| `kernels/jit/utils/compile/loader.py:35` applies `torch.compiler.disable` at import, which imports all of `torch._dynamo` | ~0.8 s | Measured: with it removed, the Scheduler still imports `torch._dynamo` through a module-level `@torch.compile` in `managers/overlap_utils.py:91` (48 such module-level decorators exist). The scheduler compiles models anyway, so this only moves the cost. |
| `hf_transformers_patches.py` `_patch_image_processor_kwargs` (`:285`) eagerly imports `transformers.image_processing_utils` → torchvision → `torch._dynamo` | ~0.8 s | Measured: with it deferred, the same chain is re-entered by ~10 config modules in `configs/all_configs.py` (`deepseekvl2.py`, `janus_pro.py`, `dots_ocr.py`, `kimi_vl.py`, …) that import `ProcessorMixin` at module level. That is a multi-owner refactor, and it overlaps #30226's file. |
| `server_args` → `transformers` via `ernie55` | ~0.45 s | Already #49914. |
| `openai` SDK via `entrypoints/openai/protocol.py` | ~0.35–0.45 s | Already #38177. |
| **`layers/joint_schema_head.py` imported by frontend code just for plain-Python layout helpers** | **~1.0 s on `systemone.serving`** | **Chosen**: small, single-purpose, unclaimed, testable on CPU. |

## 3. Root cause

The `/v1/systemone` decision API (Clef / Kev checkpoints) validates a *decision layout*: a
flat `list[int]` describing question and option spans. Validation is pure Python. But the helpers
live in the same module as the `nn.Module` head:

- `python/sglang/srt/layers/joint_schema_head.py:29-125` holds `QUESTION_TYPES`, `LayoutQuestion` (a
  `msgspec.Struct`), `max_joint_prompt_tokens`, `pack_decision_layout` and `parse_decision_layout`:
  ints in, ints out, no torch.
- The same file, `:21-26`, imports `torch`, `torch.nn`, and `sglang.srt.layers.pooler.EmbeddingPoolerOutput`.
- `layers/pooler.py:14` → `layers/activation.py` → `layers/quantization/base_config.py` → the package
  `layers/quantization/__init__.py:8-22`, which eagerly imports every method (`awq`, `fp8`, `gguf`,
  `compressed_tensors`, …) → `quantization/awq/awq.py:10` `layers/linear.py` → `layers/moe` →
  `moe/moe_runner/deep_gemm.py:101` `@torch.compile(...)` at module level → `torch._inductor.compile_fx`.

Frontend importers that only need the layout helpers:

- `managers/tokenizer_manager.py:84-87` (used at `:1533` and `:1544` in `_validate_joint_schema_request`)
- `entrypoints/systemone/serving.py:48` (used at `:164`, `:224`), which `entrypoints/http_server.py:135` imports
- `entrypoints/systemone/joint_schema.py:21` and `entrypoints/systemone/kev.py:21` (request encoders)
- `layers/pointer_head.py:17` (model side; updated only so nothing imports a re-imported name, which a pre-commit hook rejects)

So the HTTP/tokenizer process, which never runs a model, pays for the model layer stack just
to parse a list of ints.

## 4. Design

Move the five layout definitions **verbatim** into a new torch-free module,
`python/sglang/srt/layers/joint_schema_layout.py`, and point every importer at it. The head
imports `QUESTION_TYPES`, `LayoutQuestion` and `parse_decision_layout` back for its own use. The
head-only constants `_MAX_GROUP_TOKENS` and `_MAX_GROUP_SLOTS` stay in the head.

- Behavior: unchanged. The moved code is byte-identical (checked with `diff` against the original
  lines). No signatures change. Only import paths change.
- Location: next to the head in `layers/`, which is a namespace package with no `__init__.py`, so
  importing it pulls in nothing else. The layout format is defined by the head's type embedding, so
  keeping it beside the head keeps ownership clear.
- Guard test: `test/registered/unit/layers/test_joint_schema_layout.py` follows the existing
  fresh-interpreter style of `test/registered/unit/model_executor/runner_utils/test_imports.py`.
  It checks two things: (1) the layout module does not load `torch`, and a pack/parse round trip
  works; (2) `import sglang.srt.entrypoints.systemone.serving` loads neither `joint_schema_head`
  nor `sglang.srt.layers.quantization`. **Both tests fail on unpatched main** (checked by running
  them against the base tree) and pass with the patch.

### Alternatives considered

| alternative | why not |
|---|---|
| Function-local `from ...joint_schema_head import ...` inside `_validate_joint_schema_request` and `SystemOneServing` | Fewer lines, but it hides the dependency, the first request pays ~1–3 s of imports on the serving path, and the encoders (`joint_schema.py`, `kev.py`) would need the same trick. Splitting by responsibility is cleaner. |
| `LazyImport` (from `sglang/_lazy_import.py`) for `joint_schema_head` | Same first-request latency problem, and `LayoutQuestion` is used as a type and constructed in encoders, so it would be proxied all over. |
| Keep re-exports in `joint_schema_head` for backward compatibility | The repo's pre-commit hook "reject importing a name through a module that only re-imports it" forbids importing through a re-export, and ruff `F401` would flag unused re-imports. All in-repo importers are updated instead. Out-of-tree code importing `pack_decision_layout`/`max_joint_prompt_tokens` from the head would break; `parse_decision_layout`, `LayoutQuestion` and `QUESTION_TYPES` still resolve there. |
| Fix the hub: make `layers/quantization/__init__.py` lazy | This is the real cause of the cost for many importers, but it is a large change across many CODEOWNERS that overlaps #39425. It should be proposed in the issue, not slipped into a first PR. |
| Also defer `layers/logits_processor` in `managers/utils.py:15` (the tokenizer manager's other route) | Tried: the tokenizer manager then reaches the same hub through `state_capturer/base.py:7` → `mem_cache/memory_pool.py:70` → `quantization/fp4_kv_cache_quant_method`. Each extra route removed gives zero end-to-end gain until the hub is fixed, so the experiment was reverted to keep the PR to one problem. |

## 5. Measurements

Interleaved A/B, 9 fresh interpreters per side, median. The timer starts **after** `import torch`,
so these are SGLang-attributable seconds. Command: `ab_import_time.py --base <base>/python
--patched <patched>/python -n 9 <modules>`.

| module imported | base | patched | delta | quantization / inductor loaded (base → patched) |
|---|---|---|---|---|
| `sglang.srt.entrypoints.systemone.serving` | 4046 ms | 3020 ms | **−1026 ms (−25%)** | yes, yes → **no, no** |
| `sglang.srt.managers.tokenizer_manager` | 3401 ms | 3438 ms | +37 ms (noise) | yes, yes → yes, yes |
| `sglang.srt.entrypoints.http_server` | 5025 ms | 4908 ms | −117 ms (noise) | yes, yes → yes, yes |

Layout helpers alone, without preloading torch (5 runs each):

| import | time | `torch` loaded |
|---|---|---|
| base: `from sglang.srt.layers.joint_schema_head import parse_decision_layout` | 3.47–3.73 s | yes |
| patched: `from sglang.srt.layers.joint_schema_layout import parse_decision_layout` | 0.165–0.178 s (mostly `import sglang` itself) | **no** |

**The honest reading:** the slice removes one route by which the frontend reaches the model layer
stack, and it fully helps any importer whose only route was this one (`systemone.serving`, the
encoders, the layout tests). It does **not** yet make the real server process (`http_server` /
`tokenizer_manager`) start faster on its own, because other routes (`managers/utils.py:15`
→ `logits_processor`; `state_capturer/base.py:7` → `memory_pool` → `quantization`) still load the
same hub. The PR description says this plainly, and the issue comment proposes the hub fix as the
follow-up. Reviewers decide whether that is worth merging on its own (see *Review risks*).

Absolute numbers are from a slow 4-vCPU CPU box. An H100 host will show smaller absolute
times. The module-graph change (what is or isn't loaded) is the same on any hardware.

## 6. Tests run (CPU)

```text
pre-commit (isort, ruff F401/F821/F823/UP037, ruff-format, codespell, repo-local hooks) on changed files: all Passed
python3 test/registered/unit/layers/test_joint_schema_layout.py          OK (2 tests)
pytest test/registered/unit/layers/test_joint_schema_head.py \
       test/registered/unit/layers/test_joint_schema_layout.py \
       test/registered/unit/entrypoints/systemone/test_joint_schema.py \
       test/registered/unit/entrypoints/openai/test_serving_decisions.py \
       test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
       test/registered/unit/managers/test_hidden_state_server_mode.py \
       test/registered/unit/managers/test_sampling_mask_validation.py \
       test/registered/unit/configs/test_embedding_model_spec.py
  -> 123 passed, 1 skipped, 195 subtests passed
```

The one skip is `test_joint_schema_head.py:186`, *"the batched head uses FlashAttention"*, which is GPU-only.

## 7. What is not verified (needs a GPU box or full install)

1. **The skipped batched-head test** (`test_joint_schema_head.py:186`) on CUDA. The head's code
   did not change, only where three names are imported from, but CI on GPU is the real check.
2. **An end-to-end Clef/Kev serve** on `/v1/systemone`. A layout error must still return the same
   400 message. The unit tests cover the validation paths with a mocked tokenizer manager, not a live model.
3. **A full `pip install -e python` environment.** Here GPU-only dependencies (flashinfer,
   sgl-kernel, deep_gemm, cuda-python) were absent, torch was 2.15.0+cpu rather than the pinned
   2.14.1, and installing `datasets` upgraded xgrammar to 0.2.9 (pin 0.2.7). Import graphs can differ
   slightly when optional GPU packages are present. Re-measure on the target box with the CHECKLIST
   commands.
4. The CPU CI suite `base-a-test-cpu` as a whole was not run, only the files above.

## 8. Review risks

- **"Where is the end-to-end win?"** This is the biggest risk. Be upfront: it is one route among
  several, and the route map (§3, §4 last row) is the useful part for maintainers. Offer to follow
  up on the quantization-registry hub if they agree on the approach.
- **Conflicts with in-flight decision-model PRs.** Open PRs that touch `/v1/systemone` or the decision
  heads include #45621, #48477, #44046, #45618, #44709, #45957, #42410 and #41712. Any that edit the
  layout helpers in `joint_schema_head.py` will conflict. Rebase right before opening, and say so in the PR.
- **Moved public names.** `pack_decision_layout` and `max_joint_prompt_tokens` no longer exist in
  `joint_schema_head`. In-repo code is updated, but out-of-tree users would break. A reviewer may ask
  for compatibility re-exports; that conflicts with the repo's "no import through a re-export" hook,
  so ask which they prefer.
- **CODEOWNERS (`.github/CODEOWNERS` at the pinned commit):** three owner groups are touched:
  `/python/sglang/srt/managers` (line 59: tokenizer_manager, io_struct), `/python/sglang/srt/layers`
  (line 47: the head, the new layout module, pointer_head), and `/python/sglang/srt/entrypoints` (line 31:
  systemone). Each needs one approval. That's a lot of reviewers for a small PR, so make the diff
  obviously mechanical.
- **The guard test asserts `sglang.srt.layers.quantization` is not imported** by `systemone.serving`. A
  future unrelated import could trip it. That is intended (it is the regression being guarded), but a
  reviewer may ask to narrow it to `joint_schema_head` only.

---

## Independent re-verification (orchestrating session, 2026-11-14)

A separate session re-checked the claims above from scratch on a fresh, depth-1 fetch of
`bd47834` (CPU only, torch 2.15.0+cpu):

| check | result |
|---|---|
| `git am 10492-joint-schema-layout.patch` | applies cleanly |
| moved code is verbatim | every removed line of `joint_schema_head.py` reappears unchanged in `joint_schema_layout.py`, except the rewritten `typing` import |
| `import sglang.srt.layers.joint_schema_layout` | `torch` not loaded; all five helpers exported |
| new guard test `test_joint_schema_layout.py` | **base: 2 failed**, **patched: 2 passed** |
| existing `test_joint_schema_head.py` | base and patched both: 2 passed, 1 skipped, 8 subtests passed (no regression) |
| `ab_import_time.py -n 5 sglang.srt.entrypoints.systemone.serving` | 4448 ms → 3514 ms (**−934 ms, −21%**); quantization / inductor loaded `[True, True] → [False, False]` |

Still unverified here: SGLang's GPU CI suites, and whether maintainers accept this slice. That
needs the owner to submit it (see `CHECKLIST.md`).
