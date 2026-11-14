# Part III · Ch1 — Attention Backends

Course text: `course-material/eng/part3/Chapter1_Attention-Backends.md` (zero-to-sglang @ `ba7901a`).
SGLang references are at `c5d7c59` (see [source map](../sglang-source-map.md)).
Lab: [`labs/part3/attention_tiling.py`](../../labs/part3/attention_tiling.py).

## Key ideas

- **Mechanism vs. kernel vs. backend.** The *mechanism* is the model's math (MHA/GQA/MLA). A
  *kernel* is one GPU program. The *backend* is the engine adapter: it builds per-batch
  metadata, writes new K/V into the pool, and picks the right kernel for the phase. Changing
  the backend changes how the math runs, not the model. It can still change floating-point
  results, so validate numerically.
- **Prefill vs. decode shapes.** Prefill has many queries per request, so it is compute-bound
  and K/V tiles are reused across queries. Decode has one query per request, so it is
  memory-bound and the GPU can be under-filled. That is why decode kernels use **split-KV**:
  they cut a long KV sequence into segments and merge the partial results using each
  segment's log-sum-exp.
- **FlashAttention.** It tiles Q/K/V into SRAM and uses an online softmax (running max,
  running sum, rescaled accumulator), so the L×L score matrix never reaches HBM. It saves
  memory traffic, not FLOPs: prefill stays O(L²).
- **FlashInfer** focuses on serving: ragged lengths, paged KV read through index lists, and a
  plan/run split (plan once per batch, run per layer). **Triton** kernels are readable and
  hackable but depend on compile and tuning. **FA3** uses Hopper TMA/WGMMA pipelines.
  **FlashMLA** reads MLA latent caches in decode.

## How this extends mini-sglang

`python/minisgl/attention/torch_backend.py::TorchAttnBackend` already does the
backend's two non-kernel jobs:

| backend responsibility (§2.2) | mini-sglang | SGLang (`AttentionBackend`) |
|---|---|---|
| metadata | `prepare_metadata`: `kv_indices` from `page_table`, causal mask shifted by `cached_len`, `last_indices` | `init_forward_metadata` (e.g. FlashInfer `kv_indptr`/`kv_indices` + `plan`) |
| KV write | `kvcache.store_kv(k, v, batch.out_loc, layer_id)` **before** reading | `forward_batch.token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)` inside `forward_extend`/`forward_decode` |
| kernel call | one padded SDPA for any phase | separate `forward_extend` / `forward_decode` paths |

What mini-sglang *cannot* show is the kernel side. It **gathers** each request's history
into a dense padded tensor (`flat_k_cache(layer)[md.kv_indices]`) every layer, every step.
Real paged kernels read KV in place through the index list. In our profile, that gather
(`aten::index`) costs as much as SDPA itself in decode (47 ms vs 45 ms per step; see
[profiling walkthrough](../part4/profiling-walkthrough.md)). The closest SGLang relative is
`srt/layers/attention/torch_native_backend.py::TorchNativeAttnBackend`, which also loops
over sequences and calls SDPA.

Lab output (`python labs/part3/attention_tiling.py`):

```
L= 32768: one head, one layer, 16-bit scores =     2.00 GiB
Tiled + online softmax == naive: max |diff| = 8.94e-08; peak score storage 16x less
Split-KV decode, splits=64: max |diff| vs unsplit = 5.22e-08
max |diff| tiled vs SDPA (mini-sglang's mask) = 1.12e-07
batch=1, ctx=32768: 16 (request, head) tasks vs ~132 SMs -> under-fills the GPU -> split the KV
batch=64, ctx=1024: 1024 tasks -> enough parallelism already
```

## SGLang pointers

- Interface: `srt/layers/attention/base_attn_backend.py::AttentionBackend`.
- Readable kernels first: `srt/layers/attention/triton_backend.py::TritonAttnBackend.forward_decode`.
  Then the split-KV merge in `srt/layers/attention/merge_state.py::merge_state`.
- `srt/layers/attention/flashinfer_backend.py::FlashInferAttnBackend.init_forward_metadata` (plan) and `forward_decode` (run).
- Registry: `srt/layers/attention/attention_registry.py`. The choices list is in `srt/arg_groups/choices.py`.
- A real refactor issue in this area: [#28808](https://github.com/sgl-project/sglang/issues/28808)
  (the TRTLLM MHA backend inherits from FlashInfer but reuses almost nothing). Several
  people have already claimed it, but it is a good read on backend structure.

## Exercises — worked answers

**1. Responsibilities of mechanism, kernel and backend; why switching backends leaves the architecture unchanged.**
The mechanism (MHA/GQA/MLA, masks, head counts) is fixed by the trained weights and
config. The kernel is one implementation of a computation (tiling, fused softmax, split-KV)
for given shapes and hardware. The backend sits between the engine and the kernels:
it builds per-batch metadata (which tokens belong to which request, history lengths, page
indices, masks), writes new K/V to the pool slots the allocator chose, picks the
prefill or decode path, and returns the output. Switching backends swaps the adapter and
kernels but computes the same function on the same weights. The only differences are the
order of operations and precision, so outputs can differ by float rounding and should be
validated.

**2. Query count per request in prefill vs. decode; effect on kernel design.**
Prefill processes all new tokens of the request (or of a chunk) at once: many queries.
Ordinary decode processes 1 query per request. With many queries, each K/V tile loaded
into SRAM is reused by many query rows, so prefill kernels optimise for compute throughput
(big tiles, tensor cores, causal-mask skipping). With one query, each K/V byte is used
once, so decode is bound by bandwidth. It also has too little parallel work when the batch is small
(our lab: 16 tasks for 132 SMs). Decode kernels therefore split the KV sequence across
CTAs (split-KV) and merge with log-sum-exp, and they optimise the KV read path (paged
reads, GQA head grouping so one KV load serves several query heads).

**3. Why FlashAttention avoids the full matrix; is prefill then linear?**
Softmax needs a row's global max and sum, but these can be maintained *online*. Process
K/V in tiles, keep the running max `m`, the running denominator `l` and an accumulator
`o`, and rescale `l` and `o` by `exp(m_old − m_new)` whenever the max grows. After the
last tile, `o / l` equals the exact result (lab §2: max diff 9e-8). Only an
`[Lq, block]` tile exists at any time, so the L×L matrix never touches HBM. The FLOPs
are unchanged: every query still multiplies with every visible key, so full prefill
stays **quadratic** in sequence length. What became linear is the *extra memory* (and the HBM
traffic for scores).

**4. What each backend emphasises; why FlashMLA can't run any MHA model.**
FlashInfer targets serving: variable lengths, paged KV via index lists, load-balanced
plan/run, and reuse of the plan across layers. Triton targets ease of writing and adapting
kernels (readable, portable, tunable, but compile cost and tuning-dependent speed). FA3 targets
peak utilisation on Hopper (TMA, async WGMMA, warp specialisation, overlapping softmax
with matmul, FP8). FlashMLA targets MLA decode: it reads the compressed latent cache (512
latent + 64 rope dims per token, shared by all heads) and computes attention in that
space. A plain MHA/GQA model has no latent cache: its cache holds per-head K and V of
`head_dim` each, with different shapes and semantics. FlashMLA's kernel assumes the
MLA layout, head dims and absorbed projections, so it cannot run on that cache. MLA
compression comes from the model architecture, not from the kernel.

**5. Lower TTFT on long inputs but lower throughput at high concurrency and long outputs: what to check?**
That backend's prefill path is faster, but its decode path (or its overhead at large
batch) is worse. Check: (a) which kernel actually ran in each phase. A backend can mix
kernels; `--prefill-attention-backend` / `--decode-attention-backend` can combine two
compatible ones. (b) TPOT/ITL and throughput at the real concurrency and output length,
not just TTFT. (c) Per-step metadata/plan overhead and split-KV workspace at large batch,
and whether CUDA graphs are used in decode. (d) Memory: a larger workspace can mean a
smaller KV pool, smaller batches and lower throughput. (e) Correctness on the same inputs.
Then weigh the service's real mix. A long-input/short-output service values TTFT; a
long-output service values decode throughput. Measure with the production length
distribution, not just one point.
