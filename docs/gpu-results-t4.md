# GPU results: Tesla T4 (Colab)

First run of [`notebooks/gpu_benchmarks.ipynb`](../notebooks/gpu_benchmarks.ipynb) on a real GPU.
Raw outputs: [`gpu_benchmarks_t4_results.ipynb`](../notebooks/gpu_benchmarks_t4_results.ipynb) (first run:
benchmarks) and [`gpu_benchmarks_t4_rerun.ipynb`](../notebooks/gpu_benchmarks_t4_rerun.ipynb) (second
run: full test suite, serving).

- **Hardware/software:** Tesla T4 (sm_75, 15 GB), CUDA 13.0, torch 2.14.1+cu130, triton 3.8.0
- **Model and dtype:** Qwen3-0.6B in **float16**, because the T4 has no bfloat16
- **Repo version:** `main` at `2314074`

## Status

| Part | Status |
|---|---|
| `tests/test_gpu.py` | **6 passed** on the T4 (details below) |
| full test suite on the GPU | **246 passed, 4 skipped** (the 4 are slow labs behind `MINISGL_RUN_SLOW_LABS=1`) |
| benchmarks: attention, KV cache, batching, prefix cache, speculative | ran, see below |
| `bench_serving.py` | all three configurations ran (section 7) |
| `torch.profiler` traces | not yet: the cell could not import `minisgl` in the notebook kernel (an editable install needs a kernel restart). Fixed in the notebook; needs a re-run. |
| multi-GPU tensor parallelism (NCCL) | not run: a Colab T4 has one GPU |

What `tests/test_gpu.py` proves on real hardware:

| test | checks |
|---|---|
| `test_cuda_graph_fp32_matches_hf[1]`, `[16]` | captured CUDA Graph decode (Triton backend, page_size 1 and 16, batch shrinking 6 → 1 with dummy-request padding) reproduces HuggingFace greedy output token for token |
| `test_eager_fp32_matches_hf[torch]`, `[triton]` | both attention backends on CUDA reproduce HuggingFace |
| `test_cuda_graph_replay_equals_eager_bf16` | graph replay produces the same tokens as eager execution in half precision |
| `test_real_model_triton_cuda_graph_fp32` | real Qwen3-0.6B, Triton + CUDA Graph, fp32 == HuggingFace |

## 1. CUDA Graph is the biggest single win

End-to-end decode through the scheduler (`bench_attention.py`, TPOT in ms):

| backend | CUDA Graph | bs=1 | bs=8 | bs=32 |
|---|:---:|---:|---:|---:|
| torch (gather + SDPA) | off | 47.8 | 49.7 | 50.7 |
| triton | off | 56.2 | 42.1 | 57.5 |
| triton | **on** | **12.7** | **21.4** | 56.6 |

Without graphs, TPOT is almost flat from bs=1 to bs=32 (~48–51 ms). A 0.6B model on a GPU is
**launch/CPU-overhead bound**: each step issues hundreds of small kernels from Python, and the GPU
waits on the CPU. Replaying a captured graph removes that overhead, giving **4.4×** at bs=1. At bs=32
the step becomes GPU-bound and the gain disappears. This is the Part I Ch3 point about where the
bottleneck sits, now measured.

The same effect end to end (`bench_batching.py`, output tok/s):

| mode | torch eager | triton eager | triton + CUDA Graph |
|---|---:|---:|---:|
| one request at a time | 21.9 | 23.0 | 81.3 |
| continuous batching | 239.5 | 242.4 | **530.2** |

Continuous batching alone is about 11× faster than one request at a time on the GPU (only 3.5× on
the CPU). It also cuts mean TTFT from 14.8 s to 0.10 s.

## 2. Scheduler features on the GPU

- **Radix prefix cache:** 88.3% hit rate (744 of 6376 prompt tokens computed). Mean TTFT
  0.092 → 0.051 s (triton + graph), 0.067 → 0.050 s (torch eager), 0.252 → 0.051 s (triton eager).
- **Optimistic admission vs conservative:** 13–32% more throughput, with 3 retractions on the
  long-output workload:

  | workload (triton + graph) | conservative | optimistic |
  |---|---:|---:|
  | exact lengths | 361.7 | 416.7 |
  | long exact outputs | 323.9 | 373.6 |
  | chat, stops at EOS | 232.5 | 307.0 |

- **Paged KV:** 16 concurrent requests in a 2048-token budget, vs 2 with contiguous pre-allocation.
  Same as on the CPU: it is pure bookkeeping.

## 3. Speculative decoding (`bench_speculative.py`, k=4, 64 tokens)

| draft | open-ended | copy-heavy |
|---|---:|---:|
| n-gram prompt lookup | 1.50–1.90× | **2.53–3.26×** |
| first 7 / 14 layers of the target | 0.4–0.8× | 0.4–0.7× |
| target itself (100% accepted) | 1.06–1.61× | 0.82–1.31× |

On the GPU, verifying 5 tokens in one forward costs about the same as verifying 1, because decode is
overhead/memory bound. That is why the n-gram draft gains much more here than on the CPU (1.6×).
Truncated drafts are still a loss: they are untrained, so 1–5% acceptance does not pay for their own
forwards.

## 4. KV cache: little gain at these lengths

`bench_kv_cache.py` shows only 1.0–1.2× (vs 4–6× on the CPU). With a 128-token prompt and up to 128
new tokens, recomputing ≤256 tokens is almost free on a GPU, and every step is dominated by
the launch overhead from section 1. The KV cache matters once sequences get long or batches get large
enough to make the step compute bound. This shows how the same optimization can be decisive on one
machine and negligible on another.

## 5. The Triton kernels are correct but slow

Per-layer attention times (ms) from `bench_attention.py`:

| case | torch (gather + SDPA) | triton (best split) |
|---|---:|---:|
| decode bs=1, kv=8192 | 1.679 | **1.276** (16 splits) |
| decode bs=8, kv=2048 | 1.188 | 1.565 |
| decode bs=64, kv=8192 | 28.9 | 42.6 |
| prefill bs=1, 2048 | 7.6 | 20.6 |
| prefill bs=8, 2048 | 30.1 | 165.8 |

Split-KV (flash-decoding) helps exactly where theory says it should: a single long sequence, where
one block per head would leave most SMs idle. Everywhere else, PyTorch's SDPA (cuDNN/FlashAttention
kernels) wins. The prefill kernel is 3–5× slower. The likely causes are untuned `BLOCK_M`/`BLOCK_N`/`num_warps`
for sm_75 and no autotuning. Making these kernels competitive is a good next exercise
([attention-backends.md](attention-backends.md), Part III Ch1).

## 6. Exactness in float16

Some fp16 benchmark rows report `identical: False` or `lossless: False`:
- batch size 32 vs torch
- optimistic vs conservative admission
- one-at-a-time vs batched
- n-gram verification

In fp16, different batch sizes and verify widths run GEMMs of different shapes. Those round
differently, and a greedy near-tie can flip. The fp32 GPU tests above show that the mechanisms
themselves are exact on this GPU: Triton kernels, CUDA Graph padding, and graph replay all reproduce
HuggingFace token for token. So these rows are floating-point batch non-invariance, not bugs. (Real
engines handle this with "deterministic inference" modes, such as SGLang's
`--enable-deterministic-inference`.)

## 7. HTTP serving (`bench_serving.py`, 32 streaming requests)

| configuration | concurrency 1: TPOT ms / tok/s | concurrency 4 | concurrency 16 |
|---|---:|---:|---:|
| torch, eager | 47 / 21.2 | 53 / 74.4 | 60 / 252.2 |
| triton, eager | 44 / 21.1 | 47 / 83.9 | 49 / 313.7 |
| triton + CUDA Graph | **12 / 75.8** | **15 / 231.4** | **25 / 540.8** |

Through the whole multi-process HTTP stack (API server → tokenizer → scheduler → detokenizer →
SSE), CUDA Graph keeps TPOT at 12–25 ms. That gives 540.8 output tok/s and 16.9 req/s at
concurrency 16, vs 252 tok/s for the eager torch backend. Mean TTFT stays at 0.05–0.17 s.
