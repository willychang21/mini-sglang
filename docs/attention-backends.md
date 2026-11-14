# Attention backends: from gather + SDPA to paged kernels and CUDA Graphs

Companion to zero-to-sglang **Part III Chapter 1 (Attention Backends)** and the
GPU parts of Part II (Ch4 KV cache, Ch7 paged memory, Ch9 TP). Every idea is
tied to a file in this repo and to its counterpart in real SGLang
(`python/sglang/srt/...`). Section 9 answers the chapter's exercises.

| Here | Real SGLang |
|---|---|
| `attention/base.py` (`prepare_metadata`, `forward`) | `layers/attention/base_attn_backend.py` (`init_forward_metadata`, `forward_extend` / `forward_decode`) |
| `attention/torch_backend.py` | `layers/attention/torch_native_backend.py` |
| `attention/triton_backend.py`, `attention/triton_kernels.py` | `layers/attention/triton_backend.py`, `layers/attention/triton_ops/decode_attention.py`, `triton_ops/extend_attention.py` |
| `attention/flashinfer_backend.py` (optional, untested) | `layers/attention/flashinfer_backend.py` |
| — | `layers/attention/flashattention_backend.py` (FA3), `flashmla_backend.py` |
| `engine/graph.py` | `model_executor/cuda_graph_runner.py` |
| `kvcache/mha_pool.py` | `mem_cache/memory_pool.py` (`MHATokenToKVPool`) |
| `engine.page_table` (token slots per request) | `ReqToTokenPool.req_to_token` |
| `--attention-backend {torch,triton,flashinfer}` | `--attention-backend` (+ `--prefill-attention-backend` / `--decode-attention-backend`) |

---

## 1. What a backend is responsible for

The model never sees the KV cache layout. `RopeAttn.forward` (`models/utils.py`)
computes `q, k, v` for the packed tokens of this step and calls

```python
o = ctx.attn_backend.forward(q, k, v, layer_id, batch)
```

The backend owns three things:

1. **Store**: write this step's `k, v` into the pool at `batch.out_loc`
   (`MHAKVCache.store_kv`). Store comes *before* attend: the new token must see
   itself.
2. **Metadata**: once per step, before any layer runs, turn the batch
   (`cached_len`, `device_len`, `table_idx` of each request) into whatever the
   kernel needs (`prepare_metadata`). The same metadata serves all layers.
3. **Attend**: per layer, find each request's history in the pool and compute
   causal attention for its new queries.

Swapping backends changes only (2) and (3). The model, the scheduler and the
pool layout stay the same, which is why `--attention-backend` is a flag and
not a different model.

## 2. Paged attention vs gather

Request `r`'s K/V is scattered over pages; `page_table[r.table_idx, j]` is the
pool slot of its `j`-th token (Ch4/Ch7). There are two ways to attend over it.

**Gather (the `torch` backend).** Build `kv_indices [B, max_k]` (padded),
gather `k_hist = k_cache[kv_indices]` into a new `[B, max_k, Hkv, D]` tensor,
build a `[B, 1, max_q, max_k]` mask, call SDPA. Simple and runs anywhere, but
every layer of every step:

* copies the *entire* history of every request (the cache is read once to
  gather, then the copy is read again by SDPA);
* pads every request to the longest one (`max_k`), so one long request makes
  the whole batch pay;
* has shapes that change every step (`max_k` grows), so it cannot be put in a
  CUDA Graph.

**Paged (the `triton` backend).** The kernel gets only `rows[b]`, `seq_lens[b]`
(and for extend `qo_start`, `extend_lens`, `prefix_lens`) and resolves
addresses itself, a tile at a time (`triton_kernels.py`):

```python
page_start = page_table[row, (j // PAGE_SIZE) * PAGE_SIZE]  # base slot of j's page
slot       = page_start + j % PAGE_SIZE
k_tile     = k_cache[slot]                                   # straight from the pool
```

No copy, no padding to `max_k` (each program loops to its own request's
length), and the per-step metadata is a few `[B]` tensors that can live in
static buffers. `page_size` only changes the address arithmetic; `page_size=1`
degenerates to one lookup per token. SGLang's Triton backend does the same with
`req_to_token[req_pool_indices[b], j]` (it first flattens those rows into a CSR
`kv_indptr / kv_indices` pair with a small Triton kernel).

## 3. Prefill vs decode: different query shapes

| | queries per request | K/V read per request | bound by |
|---|---|---|---|
| prefill / extend | `extend_len` (often hundreds+) | `cached_len + extend_len` | compute (Q·Kᵀ is a real matmul) |
| decode | **1** | the whole history | memory bandwidth |

So the kernels are organized differently:

* **Extend** (`_extend_kernel`): one program per `(request, query head, block
  of BLOCK_M queries)`. Each program streams key tiles `[0, prefix_len +
  last query offset]` with the causal rule `key j visible to query t ⇔
  j <= cached_len + t` (prefix offset = `cached_len`). Q·Kᵀ is a
  `[BLOCK_M, BLOCK_N]` matmul: tensor cores are busy.
* **Decode** (`_decode_stage1_kernel`): a single query row would make a
  `[1, BLOCK_N]` "matmul", wasting the tensor cores. With **GQA** the `GROUP =
  Hq / Hkv` query heads that share one KV head *are* a natural M dimension, so
  one program handles `(request, KV head)` and loads each K/V tile once for
  the whole group (`BLOCK_H` rows, padded to 16 for `tl.dot`). SGLang's
  `_fwd_grouped_kernel_stage1` is the same idea.

The backend dispatches on the data, not on `batch.phase`: if every request has
exactly one new token (`max_extend_len == 1`) the decode kernel runs.

## 4. Online softmax and flash-decoding (split-KV + merge)

**Online softmax.** For one query row, process keys in tiles. Keep the running
max `m`, the running denominator `l = Σ exp(s - m)` and the running unnormalized
output `acc = Σ exp(s - m) v`. For a new tile with scores `s`:

```
m'   = max(m, max(s))
α    = exp(m - m')                 # rescale everything accumulated so far
p    = exp(s - m')
l    = α·l + Σ p
acc  = α·acc + p @ V_tile
m    = m'
```

At the end `o = acc / l`. The `n × n` score matrix never exists in memory; this
is the FlashAttention idea and it is exactly the loop body of both kernels.

**Why split.** Decode launches `B × Hkv` programs. With `B = 1` and Qwen3-0.6B
(`Hkv = 8`) that is 8 programs on a GPU with 40–132 SMs: most of the chip idles
while 8 programs each walk thousands of keys. Flash-decoding cuts each request's
keys into `S = num_kv_splits` chunks and runs `B × Hkv × S` programs.

**Merge (stage 2).** Split `s` returns its normalized output `o_s` and its
log-sum-exp `lse_s = m_s + log l_s`. Since `exp(lse_s)` *is* split `s`'s share
of the full softmax denominator,

```
M = max_s lse_s
o = Σ_s exp(lse_s - M) · o_s  /  Σ_s exp(lse_s - M)
```

is exactly the single-pass result (`_decode_merge_kernel`). An empty split
(`seq_len < S`) reports `lse = -inf` and gets weight 0;
`test_decode_more_splits_than_tokens` checks that. With `num_kv_splits = 1`
stage 1 writes straight into the output and no merge kernel runs.

Trade-off: more splits = more parallelism but an extra kernel and
`[B, Hq, S, D]` fp32 of partial results. SGLang picks the split count per
request from its length (`get_num_kv_splits`, bounded by `max_kv_splits`);
here it is a fixed `MINISGL_TRITON_KV_SPLITS` (default 8), because a fixed `S`
is also what a CUDA Graph needs. `benchmarks/bench_attention.py` sweeps it.

## 5. FlashInfer: plan / run

FlashInfer separates **planning** from **running**:

* `wrapper.plan(indptr, indices, last_page_len, ...)` runs once per *step* on
  the host. It looks at the actual lengths (CPU copies) and decides the work
  partition: which requests to split, into how many chunks, which CTA does
  what (load balancing a 128-token request next to an 8k-token one). The
  schedule is uploaded to the GPU.
* `wrapper.run(q, (k_cache, v_cache))` runs once per *layer* and only launches
  kernels following that schedule, then merges split results by LSE as above.

All layers share lengths and page lists, so one plan serves ~28 runs. The KV
cache is described page-granularly in CSR form: `indptr` (cumulative pages per
request), `indices` (page ids), `last_page_len` (valid tokens in each last
page). `attention/flashinfer_backend.py` builds those from our token-slot page
table (`page_id = page_table[row, i * P] // P`); it is registered only if
`flashinfer` imports and has not been run in CI. In SGLang the per-step plan is
done by `FlashInferIndicesUpdaterDecode` / `...Prefill` inside
`flashinfer_backend.py`; for CUDA Graphs SGLang uses
`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`, whose index buffers are static.

Compared to our Triton backend: Triton's split count is static and the kernel
derives its chunk from `seq_len` on the GPU (nothing to plan, cheap, but no
global load balancing). FlashInfer pays host-side planning for better
balancing on skewed batches.

## 6. FlashAttention 3 and FlashMLA

**FA3** (`flashattention_backend.py`, `flash_attn_with_kvcache`) takes a
page table of *page ids* plus `cache_seqlens` / `cu_seqlens_q`, like the
reference mini-sglang's `attention/fa.py`. What it does differently is inside
the kernel, on Hopper: TMA loads tiles asynchronously, warp-specialized
producer warps fetch K/V while consumer warps run `wgmma` and the softmax of
the previous tile overlaps the matmul of the next ("ping-pong"). Same math as
section 4, but the memory pipe and the tensor cores are both kept busy. The
price is narrow support (sm90, particular head dims / dtypes).

**FlashMLA** serves DeepSeek's Multi-head Latent Attention. The *cache layout*
is the main difference:

| | stored per token per layer | DeepSeek-V3 example |
|---|---|---|
| MHA / GQA (this repo: `[2, L, pages, P, Hkv, D]`) | `2 × Hkv × D` | — (MHA would be 128 heads × (192 + 128) ≈ 41k values) |
| MLA | one latent `c_kv` (`kv_lora_rank`) + one shared RoPE key (`qk_rope_head_dim`) | 512 + 64 = **576** values, no head dimension, no separate V |

In decode the K/V up-projections are absorbed into the query and output
projections, so attention runs directly against the 576-wide latent: it looks
like MQA with one KV "head" of size 576 whose first 512 dims also serve as V.
SGLang's `MLATokenToKVPool` stores exactly that single buffer. A kernel for
MHA cannot read it (there are no per-head K/V to read), hence FlashMLA. SGLang
uses FlashMLA mainly for decode; extend goes through another implementation.

## 7. CUDA Graphs and the "+1 dummy page"

Decode of a small model is hundreds of tiny kernels; launching them from
Python costs more than running them. `engine/graph.py` captures one decode
forward per batch size in `determine_graph_bs` (1, 2, 4, 8, 16, …) and replays
it:

* **Static inputs**: `GraphInputBuffers` (`input_ids`, `positions`, `out_loc`)
  and the backend's capture buffers (`rows`, `seq_lens` in
  `TritonAttnBackend.init_capture_graph`). Before replay we only `copy_` into
  them (`prepare_for_replay`); the graph sees new values at the same addresses.
* **Padding**: a batch of 5 replays the graph for 8 (`pad_size`). Rows 5–7
  are the **dummy request**: token 0, position 0, page-table row
  `max_running_req` (the extra row allocated in Ch4), `seq_len` 1, and
  `out_loc` = first slot of the **dummy page**, the `num_pages + 1`-th page
  of the pool that the allocator never hands out. Padding rows therefore
  compute garbage but can only write garbage into memory nobody reads.
* Only decode is captured; the torch backend opts out
  (`supports_cuda_graph = False`) because its shapes change every step.
* Padding happens inside the `Engine`, so the scheduler needed no change.

SGLang's `CudaGraphRunner` does the same (capture sizes, `bs → padded bs`,
`init_forward_metadata_capture_cuda_graph` / `..._replay_cuda_graph` on the
backend, a `seq_len_fill_value` for padded rows); its token pool reserves slot
0 as the write target of padded tokens instead of an extra page.

**What is verified here (CPU, no GPU):** bucketing, dummy page / row
bookkeeping, buffer padding, and full generations whose every decode step goes
through `GraphRunner.replay` with an eager stand-in for the graph
(`tests/test_cuda_graph.py`). **The capture itself is only exercised by
`tests/test_gpu.py`** (run it with `notebooks/gpu_benchmarks.ipynb`).

## 8. How the Triton kernels are tested without a GPU

`TRITON_INTERPRET=1` runs `@triton.jit` functions in a numpy interpreter
(`tests/conftest.py` sets it when CUDA is absent; the backend sets it itself on
CPU). `tests/test_triton_attention.py` compares both kernels with the torch
backend on random K/V scattered over shuffled pages (page_size 1/4/16, GQA
groups 1/4/6, head_dim 16/24/32, split counts 1/3/4/8, empty splits, bf16) and
runs the scheduler end to end against HuggingFace greedy output.

One interpreter quirk: numpy has no bfloat16, and the interpreter's `tl.dot`
returns garbage for bf16 inputs. Under the interpreter only, bf16 tiles are
upcast to fp32 before `tl.dot` (`DOT_FP32`); on a GPU they go to the tensor
cores as bf16. All accumulation is fp32 everywhere, and `tl.dot` uses
`input_precision="ieee"` so fp32 runs are not silently TF32 on Ampere+.

## 9. Exercises (Part III Chapter 1, §6.2)

**1. Mechanism vs kernel vs backend.** The *mechanism* is the math (causal
softmax attention, GQA, MLA…) and is fixed by the model. A *kernel* is one GPU
program that computes part of it for a given data layout (`_decode_stage1_kernel`).
The *backend* is the adapter between the engine and kernels: it stores K/V,
turns the batch into kernel metadata, picks the kernel per phase, and
cooperates with CUDA Graphs (`TritonAttnBackend`). Switching backends leaves
the architecture unchanged because the model only calls
`attn_backend.forward(q, k, v, layer_id, batch)`; the pool layout, the page
table and the math are the same. In this repo all backends must and do give
the same greedy tokens as HuggingFace.

**2. Query count in prefill vs decode.** Prefill has `extend_len` queries per
request (all prompt tokens, or a chunk); decode has exactly one. Prefill
kernels tile the query dimension (`BLOCK_M`) and are compute-bound; decode
kernels have nothing to tile in M, are memory-bound, and instead (a) use the
GQA group as the M dimension and (b) parallelize over the KV length with
split-KV + LSE merge, because `B × Hkv` programs cannot fill the GPU at small
batch.

**3. Why FlashAttention does not store the attention matrix, and is prefill
then linear?** Tiles of Q, K, V are loaded on chip and the softmax is computed
online (running max, running denominator, rescaling of the accumulated
output), so each score tile is consumed right after it is produced. This
removes the O(n²) *memory* and HBM traffic, not the O(n²) *compute*: every
query still dots with every visible key, so full prefill stays quadratic in
sequence length (causal halves the constant).

**4. What each backend emphasizes; why FlashMLA does not work for MHA.**
FlashInfer: serving-shaped inputs — variable lengths, paged KV via CSR
indices, host-side plan for load balancing, plan once / run per layer.
Triton: kernels that are easy to read and change, compiled for many GPUs
(this repo's choice for exactly that reason). FA3: hardware utilization on
Hopper (TMA, warp specialization, overlapped softmax and matmul). FlashMLA:
MLA's compressed latent cache in decode. FlashMLA expects every token's cache
entry to be one 576-wide latent shared by all heads, with V living inside it;
an MHA/GQA model has per-head K and V of a different size and no absorbed
projections, so there is nothing for FlashMLA to read.

**5. Lower TTFT on long inputs, but lower throughput with high concurrency and
long outputs.** The backend has a faster prefill but a weaker decode path (or
a costlier per-step overhead). Check, with the same model, dtype, lengths and
concurrency, after warm-up: TPOT/ITL at the real concurrency, whether decode
is CUDA-Graph-captured for those batch sizes (and the padding waste), per-step
metadata/plan cost (FlashInfer plan, CSR index building), split-KV settings vs
the real context lengths, workspace memory reducing the KV pool and therefore
the max batch, and correctness. Then weigh by the real traffic mix: a
chat/agent service with long outputs is dominated by decode, so it may be the
wrong backend there even though it wins TTFT; it can still be the right
*prefill* backend in a split prefill/decode configuration
(`--prefill-attention-backend` / `--decode-attention-backend`).
`benchmarks/bench_attention.py` and `bench_serving.py` measure exactly these
axes here.
