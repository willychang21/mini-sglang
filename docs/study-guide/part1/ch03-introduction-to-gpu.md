# Part I · Chapter 3 — Introduction to GPU

> Tutorial: `part1/Chapter3_Introduction_to_GPU.md` · Lab: `labs/part1_ch02_roofline.py`
> (sections 1, 3 and 5) · Related code: `python/minisgl/engine/engine.py`, `config.py`,
> `scheduler/scheduler.py`, `attention/torch_backend.py`

## Key ideas

- **Design philosophy.** A CPU spends its chip area on control and cache (branch
  prediction, out-of-order execution) so that *one* thread finishes as fast as
  possible. It is optimized for latency. A GPU spends its area on thousands of simple
  ALUs and keeps many threads in flight. When one group waits for memory, another
  runs. It is optimized for throughput. Matrix multiplication is huge, regular and
  free of branches, which is exactly what the GPU is built for. The CPU is still
  needed to orchestrate the work: tokenize, schedule, launch kernels.
- **A100 hierarchy (PCIe 80GB):** GPC (7) → TPC (54) → SM (108) → per SM: 64 FP32
  CUDA cores (6,912 in total) + 4 third-generation Tensor Cores (432 in total).
  Tensor Cores do matrix multiply-accumulate (MMA) at warp level. Dense BF16/FP16 peak
  is **312 TFLOPS**. TF32 is 156, INT8 is 624 TOPS, and FP64 is 19.5. 2:4 structured
  sparsity doubles the dense figures (except FP64) by skipping the zero half of the
  weights.
- **Execution model (SIMT):** a kernel launches a grid of **blocks**. Each block is
  placed on one SM and stays there, sharing that SM's shared memory and registers. A
  block is split into **warps** of 32 consecutive threads. A warp is the scheduling
  unit and runs one instruction at a time for all 32 threads. Up to 64 warps can be
  resident per SM, handled by 4 schedulers, which hides memory latency. **Warp
  divergence:** if threads in a warp take different branches, the paths run one after
  the other under a mask.
- **Memory hierarchy:** registers (256 KB/SM, private to a thread) → L1/shared memory
  (192 KB/SM, up to 164 KB as shared; ~20–30 cycles) → L2 (40 MB, shared by all SMs,
  ~200 cycles) → HBM (80 GB, ~2 TB/s, ~290 cycles). Fast memory is small and close to
  the SM. A fast kernel loads a **tile** from HBM into shared memory once, reuses it
  many times from registers, then writes the result back. That reuse is what raises
  arithmetic intensity (Ch2).
- **Memory wall:** compute has grown faster than bandwidth. A100: 312 TFLOPS against
  ~2 TB/s gives a ridge point of ~156 FLOP/B. Any kernel that does fewer FLOPs per
  byte than that waits on HBM.
- **LLM inference on a GPU:** CPU preprocessing (tokenize, build the inputs, launch
  kernels) → host-to-device copy of token IDs over PCIe → **prefill** (GEMM, Tensor
  Cores, usually compute-bound) → **decode** (GEMV-like, reads all weights and KV each
  step, memory-bound) → device-to-host copy of the sampled tokens → detokenize.

## Where in mini-sglang

mini-sglang is pure PyTorch: it contains no CUDA kernels of its own (README:
"no overlap loop, CUDA graph, or custom kernels"). The GPU ideas show up as where the
host/device boundary sits and how work is batched:

| Concept | Code | Note |
|---|---|---|
| Device and precision choice | `python/minisgl/config.py:_default_device`, `_default_dtype` | bf16 on CUDA (Tensor Core format); fp32 on CPU |
| One process/device per TP rank | `python/minisgl/engine/engine.py:Engine.__init__` (`cuda:{rank}`), `Engine._init_communication` | NCCL on GPU, Gloo on CPU (Part II Ch9) |
| CPU-side scheduling, H2D copy | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch` | index tensors are built on the host, then `.to(self.device)` |
| GPU work = big batched matmuls | `python/minisgl/models/qwen3.py:Qwen3Model.forward`, `layers/linear.py` | packed `[N, C]` layout: one GEMM per linear layer for *all* tokens in the step |
| D2H sync every step | `python/minisgl/engine/engine.py:Engine.forward_batch` (`next_tokens.to("cpu")`) → `Scheduler._process_last_data` (`.tolist()`) | the CPU waits for the GPU here; the real SGLang overlaps this (overlap scheduler) |
| Padding waste (like idle lanes) | `python/minisgl/attention/torch_backend.py:TorchAttnBackend.forward` | one padded SDPA over `[B, max_q, max_k]`; padded positions are computed and then masked |
| Memory capacity planning | `python/minisgl/engine/engine.py:Engine.__init__` (KV pool allocated up front) | HBM capacity → KV tokens (Ch4) |

## Common misconceptions

**Misconception:** A GPU is a faster CPU.
**Actually:** Each GPU thread is slower and simpler. The GPU wins by keeping tens of
thousands of threads in flight and hiding latency with throughput. Branchy, serial
code (like a scheduler loop) belongs on the CPU.

**Misconception:** More CUDA cores means faster LLM inference.
**Actually:** Matmuls run on Tensor Cores. On an A100 they give 312 TFLOPS BF16,
while the CUDA cores give 19.5 TFLOPS FP32 (NVIDIA datasheet; the tutorial does not
quote the CUDA-core figure). Decode is limited by HBM bandwidth, not by core count.

**Misconception:** Data moves from HBM straight into the compute units.
**Actually:** All HBM traffic goes through L2. Good kernels stage tiles in shared
memory and registers so each byte from HBM is used many times.

**Misconception:** Prefill is always compute-bound (the Ch3 summary table).
**Actually:** That is true for long or batched prompts. Ch2's own example (N=100 on an
A100, AI 100 < 195) is memory-bound. The ridge point decides, not the stage name.

**Misconception:** Sparse 624 TFLOPS applies to normal models.
**Actually:** It needs weights pruned to the 2:4 pattern (and usually retrained). Dense
checkpoints get the dense 312 TFLOPS.

## Lab

The roofline lab (shared with Ch2) puts numbers on the GPU ideas. Section 5 measures
the memory wall on this machine's CPU:

```bash
python labs/part1_ch02_roofline.py
```

```text
== 1. ridge point = peak FLOP/s / memory bandwidth (numbers quoted by the tutorial)
|      A100 40GB |          312 | 1.60 |          195 | Ch2 Sec.4.3/5.2: 312 TFLOPS, 1.6 TB/s
| A100 80GB PCIe |          312 | 2.00 |          156 | Ch3 Sec.2.2.5/3.3.1: 312 TFLOPS, ~2 TB/s

== 5. measured on this CPU: fp32 [N, 2048] x [2048, 2048]
(25 distinct weight matrices of 16 MiB cycled, 400 MiB total, to defeat the CPU caches)
|   N | ms / matmul | GFLOP/s | GB/s | AI (fp32) FLOP/B |
|   1 |        0.43 |    19.6 | 39.1 |             0.50 |
|   8 |        0.88 |    76.2 | 19.2 |             3.97 |
|  64 |        1.56 |   343.1 | 11.4 |            30.12 |
| 512 |        9.03 |   475.7 |  2.8 |           170.67 |
time N=8 / N=1 = 2.05x for 8x the FLOPs  -> memory-bound: the weight read dominates
GFLOP/s N=512 / N=1 = 24.3x            -> weight reuse moves toward the compute roof
rough CPU ridge = best GFLOP/s / GEMV GB/s = 12 FLOP/B (fp32)
```

What to look for:

- **N = 1 is a GEMV.** It is limited by streaming the 16 MiB weight (~39 GB/s here,
  against ~2,000 GB/s HBM on an A100). Eight rows cost only about 2× the time:
  decode-style batching is almost free while you are memory-bound.
- **N = 512 is a GEMM.** The same bytes give 24× more FLOP/s. That is the
  prefill / large-batch regime.
- The weights are cycled through 400 MiB because this Xeon has a 260 MiB L3. A single
  matrix would stay in cache and you would measure cache bandwidth, not DRAM: the
  same "hit in L2 is not HBM traffic" caveat as Ch2 Sec. 5.1.
- The CPU ridge (~12 FLOP/B, fp32) is much lower than an A100's (~156–195), because
  the GPU's compute advantage (~650×: 312 TFLOPS BF16 against this CPU's
  476 GFLOP/s fp32) is much larger than its bandwidth advantage (~50×: 2 TB/s against
  39 GB/s). GPUs need *more* reuse before they stop waiting on memory.

## Exercises

**1. What are the stages of LLM inference, what characterizes each, and why did the
GPU become the main tool?**

Stages: (a) **preprocessing on the CPU**: tokenize, build input/position tensors,
choose the batch, launch kernels. The weights were already copied to HBM at load
time, and token IDs go over PCIe. (b) **Prefill**: all prompt tokens in one forward
pass. GEMMs, Tensor Cores, high parallelism, writes the KV cache, produces the first
token. Usually compute-bound. (c) **Decode**: one token per request per step. Reads
all weights and the whole KV cache, appends one KV entry. GEMV-like, memory-bound and
serial across steps. (d) Copy the sampled IDs back to the host and detokenize.

Why the GPU: (1) inference is almost entirely matmul, which thousands of CUDA cores
and Tensor Cores do in parallel (312 TFLOPS BF16 on an A100); (2) it needs huge
bandwidth to move weights and KV every step, and HBM gives ~2 TB/s against hundreds of
GB/s for server DRAM (39 GB/s measured on this 4-core VM); (3) latency hiding through
warp switching fits large homogeneous batches; (4) the software stack is mature
(CUDA, cuBLAS/cuDNN, PyTorch, FlashAttention, FlashInfer, SGLang).

**2. Essential architectural difference between CPU and GPU; why is the GPU suited
to deep-learning matrix operations?**

The CPU optimizes latency: a few (4–64) powerful cores, most of the chip spent on
control (branch prediction, out-of-order and speculative execution) and large caches,
and coherent caches managed automatically. The GPU optimizes throughput: thousands of
simple ALUs grouped into SMs, very little control logic per ALU, SIMT execution in
32-thread warps, a large register file and fast warp switching to hide latency, plus
a memory hierarchy under explicit control (shared memory). Matrix multiplication is
billions of independent multiply-adds with no data-dependent branches (so no warp
divergence) and with a regular access pattern that tiles well into shared memory and
registers. That is exactly the work a GPU does well. Tensor Cores go further and do a
small matrix MMA per instruction.

**3. Prefill vs. decode: computation type, bottleneck resource, typical
optimizations.**

| | Prefill | Decode |
|---|---|---|
| Computation type | GEMM over all prompt tokens (`[N, C]×[C, C']`), plus `N×N` causal attention | GEMV per request (`[1, C]×[C, C']`), attention of 1 query over S cached keys |
| Parallelism | tokens × requests; one big pass | across requests only; strictly serial over steps |
| Bottleneck resource | Tensor Core compute (once N > ridge) | HBM bandwidth (weights + KV), and HBM capacity (KV size) |
| AI | ≈ N | ≈ 1 (≈ B for the linears when batched) |
| Typical optimizations | larger and batched prefill, chunked prefill, fused kernels and FlashAttention, prefix caching (skip work) | continuous batching, KV cache, quantization of weights and KV, GQA/MLA, CUDA graphs (launch overhead), speculative decoding |
