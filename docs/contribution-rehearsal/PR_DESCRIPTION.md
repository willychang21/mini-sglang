<!-- Draft for the owner to adapt. Re-measure on your own machine before submitting and replace
the numbers. Title: -->

**Title:** `[Perf] systemone: Validate decision layouts without importing the joint schema head`

## Motivation

Part of #10492 (one slice, following the incremental approach discussed in the thread).

The tokenizer manager and the `/v1/systemone` entrypoints validate *decision layouts*: they pack and
parse a `list[int]` of question and option spans, and bound the prompt length. That code is plain
Python, but it lives in `layers/joint_schema_head.py` next to the `nn.Module` head. Importing it pulls
`layers/pooler.py` → `layers/activation.py` → the `layers/quantization` registry (which eagerly imports
every method) → `layers/linear.py` → `layers/moe` → `moe_runner/deep_gemm.py` (module-level
`@torch.compile`) → `torch._inductor.compile_fx`. So frontend code paid for the model layer stack
just to parse ints.

## Modifications

- New `python/sglang/srt/layers/joint_schema_layout.py` (no torch import): `QUESTION_TYPES`,
  `LayoutQuestion`, `max_joint_prompt_tokens`, `pack_decision_layout`, `parse_decision_layout`, moved
  **verbatim** from `joint_schema_head.py`.
- `joint_schema_head.py` imports `QUESTION_TYPES`, `LayoutQuestion` and `parse_decision_layout` back for the
  head; the head-only group bounds stay there.
- Importers switched to the new module: `managers/tokenizer_manager.py`, `entrypoints/systemone/{serving,joint_schema,kev}.py`,
  `layers/pointer_head.py`, a comment in `managers/io_struct.py`, and two existing tests.
- New `test/registered/unit/layers/test_joint_schema_layout.py` (CPU, fresh interpreter): the layout module
  does not load `torch`, and `entrypoints.systemone.serving` loads neither the head nor
  `sglang.srt.layers.quantization`. Both tests fail on current main.

No behavior change: the moved code is byte-identical and no signatures change.
`pack_decision_layout` and `max_joint_prompt_tokens` are no longer importable from `joint_schema_head`;
all in-repo users are updated.

**Scope note:** this removes one route from the frontend to the model layer stack. On its own it
does not speed up `http_server` / `tokenizer_manager` startup, because they still reach the same
`layers/quantization` registry via `managers/utils.py` → `logits_processor` and
`state_capturer/base.py` → `mem_cache/memory_pool.py`. I mapped those routes in #10492 and am happy to
follow up on the registry itself if that approach is welcome.

## Accuracy Tests

Not applicable: no kernel, model-forward or sampling change. The layout code is moved verbatim, and
the existing layout and decision unit tests pass unchanged (see below).

## Speed Tests and Profiling

Import time, interleaved A/B, 9 fresh interpreters per side, median. The timer starts after `import torch`.
CPU-only box (4 vCPU Xeon @ 2.3 GHz, Python 3.13, torch CPU build); absolute numbers will be lower on a GPU host.

| module imported | main | this PR | delta | `layers.quantization` / `torch._inductor.compile_fx` loaded |
|---|---|---|---|---|
| `sglang.srt.entrypoints.systemone.serving` | 4046 ms | 3020 ms | −1026 ms (−25%) | yes / yes → **no / no** |
| `sglang.srt.managers.tokenizer_manager` | 3401 ms | 3438 ms | within noise | unchanged (other routes, see scope note) |
| `sglang.srt.entrypoints.http_server` | 5025 ms | 4908 ms | within noise | unchanged |

Layout helpers alone (no torch preloaded): `from ...joint_schema_head import parse_decision_layout`
3.5–3.7 s with torch loaded → `from ...joint_schema_layout import parse_decision_layout` 0.17 s, torch not loaded.

Reproduce:
```bash
PYTHONPATH=python python -X importtime -c "import sglang.srt.entrypoints.systemone.serving" 2> imp.log
PYTHONPATH=python python -c "import sys, sglang.srt.entrypoints.systemone.serving; \
  print('sglang.srt.layers.quantization' in sys.modules)"   # main: True, this PR: False
```

Tests run:
```text
pre-commit run --files <changed files>   # all hooks passed
pytest test/registered/unit/layers/test_joint_schema_layout.py \
       test/registered/unit/layers/test_joint_schema_head.py \
       test/registered/unit/entrypoints/systemone/test_joint_schema.py \
       test/registered/unit/entrypoints/openai/test_serving_decisions.py \
       test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
       test/registered/unit/managers/test_hidden_state_server_mode.py \
       test/registered/unit/managers/test_sampling_mask_validation.py \
       test/registered/unit/configs/test_embedding_model_spec.py
# 123 passed, 1 skipped (batched head needs FlashAttention), 195 subtests passed
```
Not tested: the FlashAttention head test and a live Clef/Kev server on GPU (I'll add results here
<!-- after running CHECKLIST step 6 on a GPU box -->).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (N/A: internal module move.)
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
