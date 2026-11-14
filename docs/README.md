# From mini-sglang to SGLang

Study notes and tools that bridge this repo (Part II of
[zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)) to contributing to
[SGLang](https://github.com/sgl-project/sglang). SGLang references are pinned to
`c5d7c598049655046ccbdb8ae8d95b8e3aa3d2b3`; the course to `ba7901a`.

| | |
|---|---|
| [sglang-source-map.md](sglang-source-map.md) | every mini-sglang module → SGLang file/class, what SGLang adds, reading order |
| Part III | [Ch1 attention backends](part3/ch01-attention-backends.md) · [Ch2 quantization](part3/ch02-quantization.md) · [Ch3 hierarchical caching](part3/ch03-hierarchical-caching.md) · [Ch4 scaling out](part3/ch04-scaling-out.md) · [Ch5 PD disaggregation](part3/ch05-pd-disaggregation.md) |
| Part IV | [Ch1 deploying](part4/ch01-deploying-with-cookbook.md) · [Ch2 profiling](part4/ch02-profiling.md) + [trace walkthrough](part4/profiling-walkthrough.md) · [Ch3 PR workflow](part4/ch03-pr-workflow.md) |
| [contributing-to-sglang.md](contributing-to-sglang.md) | dev setup, tests, lint, CI, review etiquette, five researched open issues with plans |
| [self-check.md](self-check.md) | 45-question readiness quiz with answers |
| [../labs/](../labs/README.md) | runnable CPU labs used by the chapter notes |
| [tools/verify_source_map.py](tools/verify_source_map.py) | `python docs/tools/verify_source_map.py <sglang-checkout> docs/*.md docs/part*/*.md` |

Every chapter note answers all of the chapter's exercises. Numbers quoted in the notes come
from the labs and from `benchmarks/profile.py` run on a 4-thread CPU (no GPU was available);
GPU-only steps are marked as such.
