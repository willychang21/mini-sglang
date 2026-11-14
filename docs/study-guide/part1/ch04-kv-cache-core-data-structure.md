# Part I · Chapter 4 — KV Cache: The Core Data Structure of Inference

> Tutorial: `part1/Chapter4_KV Cache The Core Data Structure of Inference.md` · Lab:
> `labs/part1_ch04_kv_memory.py` · Related code: `python/minisgl/kvcache/mha_pool.py`,
> `attention/torch_backend.py`, `scheduler/cache.py`, `scheduler/prefill.py`, `core.py`
> · Continues in Part II Ch4 (`part2/ch04-kv-cache-optimization.md`) and Ch7 (paged KV)

## Key ideas

- **Why cache K and V but not Q.** At decode step `t+1`, the attention output is
  `softmax(q_{t+1} K_{1:t+1}ᵀ/√d_h) V_{1:t+1}`. The K and V rows of earlier tokens do
  not change: causal masking means a token's hidden state never depends on later
  tokens, so its projections are fixed. They can be computed once and reused. `q` is
  needed only for the token's own step, so caching it would be useless. Each layer
  has its own K/V, because each layer has its own weights and inputs.
- **What is still computed every decode step:** the new token's embedding, its
  q/k/v in every layer, attention of that one query against *all* cached keys
  (O(S) work and bytes), the FFN, and the LM head. The cache removes the
  re-projection of the history. It does not remove reading the history.
- **Lifecycle of one request:** allocate → prefill writes K/V for all T prompt tokens
  in one pass and samples the first token from the *last* position → each decode step
  embeds the sampled token ID (not the previous hidden state), appends 1 K/V entry per
  layer, and reads all of them → on EOS, length limit or abort, the blocks are
  released (reference count → 0 → free pool), or kept in a prefix cache for reuse
  until they are evicted.
- **Memory formula** (dense, theoretical):
  `KV_bytes = Σ_b (T_b + l_b) × L × 2 × n_kv × d_h × s`. Per token, everything except
  the token count is a model constant: `2·L·n_kv·d_h·s`. It grows linearly in tokens,
  layers and KV heads. That is why GQA (fewer KV heads) and FP8 KV (smaller `s`) are
  the main ways to shrink it.
- **Scale:** Qwen3-4B needs 144 KiB per token, so 0.56 GiB at 4K tokens and 4.5 GiB at
  32K. Eight 32K requests take 36 GiB, which is far more than the 7.45 GiB of weights.
  Activations are short-lived scratch space. The KV cache is long-lived state per
  request, and it decides how many requests fit.
- **Why one contiguous buffer per request fails** when there are many requests: you
  must reserve for the worst case (max length), which wastes memory; free gaps of
  different sizes cause external fragmentation; growing a buffer means copying it;
  shared prefixes are stored twice; and batching variable-length requests needs
  padding or gather copies. The fix is to manage KV in fixed-size **pages/blocks**,
  with a page table and prefix sharing. That is Part II Ch4/7/8.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| The KV cache tensor | `python/minisgl/kvcache/mha_pool.py:MHAKVCache` | one buffer `[2, L, num_pages, page_size, n_kv, d_h]` for **all** requests |
| Pool size = the formula | `MHAKVCache.num_bytes`; sized in `engine/engine.py:Engine.__init__` from `max_total_tokens` | allocated once at startup |
| Write (prefill and decode append) | `MHAKVCache.store_kv`, called first in `attention/torch_backend.py:TorchAttnBackend.forward` | scatter into the `out_loc` slots |
| Read history | `TorchAttnBackend.forward` (`flat_k_cache(layer)[kv_indices]`) | gathered through the request's `page_table` row |
| Q is not cached | `python/minisgl/models/utils.py:RopeAttn.forward` | q is used only in this step's SDPA call |
| Per-request lengths | `python/minisgl/core.py:Req` (`cached_len`, `device_len`, `extend_len`), `Req.complete_one` | prefill: `extend_len = T`; decode: `extend_len = 1` |
| Allocation | `python/minisgl/scheduler/cache.py:CacheManager.allocate_paged` (from `Scheduler._prepare_batch`) | pages taken from `free_slots` only when needed |
| Capacity check on admission | `python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one` | reserves `extend_len + output_len + page_size` tokens of budget (accounting only, not physical) |
| Release / keep for reuse | `python/minisgl/scheduler/scheduler.py:Scheduler._free_req_resources` → `CacheManager.cache_req(finished=True)` | the radix cache keeps the prefix; the rest goes back to `free_slots` |
| Reference counts and eviction | `python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.lock_handle`, `evict` | `ref_count > 0` protects a node; LRU leaf eviction |

## Common misconceptions

**Misconception:** The KV cache makes decode O(1) per token.
**Actually:** It removes the O(S) *re-projection* of the history. Attention still reads
all S cached keys and values every step, so decode bytes (and the attention FLOPs)
still grow with S (Ch2: 1.5 GB of KV read per token at S=3000 on LLaMA-7B).

**Misconception:** The next decode step's input is the last hidden state `h_t`.
**Actually:** It is the embedding of the **sampled token ID**. The hidden state is
thrown away after the LM head.

**Misconception:** The tutorial's "14.06 MB" is decimal megabytes.
**Actually:** 14,745,600 B / 10⁶ = 14.75 MB. The tutorial divides by 2²⁰, so its
MB/GB are MiB/GiB (lab section 1).

**Misconception:** KV size depends on the number of attention (query) heads.
**Actually:** It depends on **KV** heads. Qwen2.5-1.5B has 12 query heads but 2 KV
heads, so its cache is 6× smaller than MHA would need (lab section 3).

**Misconception:** mini-sglang's admission "reserves" memory like a contiguous
allocator.
**Actually:** `PrefillAdder` only *accounts* for the worst case
(`input + max_tokens`) so a running request never runs out of slots. Physical pages
are taken from `free_slots` step by step in `allocate_paged`, and the request holds
only what it has written (plus one partly filled page).

## Lab

```bash
python labs/part1_ch04_kv_memory.py         # ~4 s; --budget-gib 16 --max-seq-len 32768 --page-size 16
```

What to look for: section 1 reproduces every number in the chapter. Section 2 answers
exercise 3. Section 4 answers exercise 4 quantitatively. Section 5 checks the formula
against the engine's real pool.

```text
== 4. a 16 GiB KV budget, Qwen2.5-1.5B: contiguous max-length reservation vs paged
budget = 599,186 token slots; request length (prompt+output) 274..3971, mean 2131
|                            allocation | concurrent requests | slots holding real KV |
| contiguous, reserve max_seq_len=32768 |                  18 |                  6.2% |
|                   paged, page_size=16 |                 278 |                 99.7% |

== 5. formula vs. mini-sglang's MHAKVCache (tiny model)
|                        pool | _kv_buffer shape [2,L,pages,page,kv_heads,dim] | B/token | slots | num_bytes | formula |
| Engine(tiny), torch.float32 |                          (2, 3, 128, 4, 2, 16) |     768 |   512 |    393216 |  393216 |
|      Qwen3-0.6B shape, bf16 |                         (2, 28, 64, 1, 8, 128) |  114688 |    64 |   7340032 | 7340032 |
```

The same formula explains the server log in Part 0 (`kv_tokens=32768 (7168.0 MiB)`):
in fp32 on CPU, Qwen3-0.6B needs 2×28×8×128×4 = 229,376 B/token, and × 32,768 = 7 GiB.

## Exercises

**1. In full-attention inference, which computations produce the KV cache, and what
is it for?**

In every layer, each token's normalized hidden state `h` is projected:
`k = h W_K`, `v = h W_V` (and, in Qwen3, `k` goes through RMSNorm and RoPE before it
is stored). *Prefill* computes K and V for all T prompt tokens in one batched pass
(`K_{1:T} = X W_K`) and writes them. Every *decode* step computes `k_t, v_t` for the
single new token and appends them: `K_{1:t} = concat(K_{1:t−1}, k_t)`. Its purpose is
the decode step `o_t = softmax(q_t K_{1:t}ᵀ/√d_h) V_{1:t}`, which needs every earlier
key and value. Without the cache, each step would recompute the projections of the
whole prefix in every layer, which is O(t) extra work per step and O(n²) over a
generation. With it, each step projects one token and only *reads* the history. In the
tutorial's toy example the decode query `q_4 = [0.82, 0.726, 0.548, 0]` scores
`[0.684, 0.637, 0.773, 0.750]` against the four cached keys. That gives weights
`[0.243, 0.232, 0.266, 0.260]` and `o_4 = [0.721, 0.686, 0.617, 0.266]` (checked with
PyTorch; the tutorial leaves `a_4` symbolic). Rows 1–3 were reused, not recomputed.

**2. Full lifecycle of the KV cache from request arrival to completion.**

1. *Admission / allocation.* The scheduler checks that the request fits (slots, token
   budget, KV capacity). With a prefix cache, it first matches the longest cached
   prefix and locks it (ref count +1), so only the uncached suffix needs new slots.
   In mini-sglang: `PrefillAdder._try_allocate_one`, then `CacheManager.allocate_paged`.
2. *Prefill (bulk write).* One forward pass over the uncached prompt tokens. Every
   layer writes their K/V into the allocated slots (`store_kv`), then attends. The
   last position's logits give the first output token. Matched prefix tokens are
   skipped.
3. *Decode (read + append), repeated.* Embed the sampled ID, compute q/k/v per layer,
   append k/v (one new slot, and a new page only at a page boundary), read the whole
   history for attention, sample. The cache grows by `2·L·n_kv·d_h·s` bytes per token.
4. *Completion.* EOS, `max_tokens`, or an abort. The request's slots are released:
   reference counts drop, and slots whose count reaches 0 go back to the free pool.
   With prefix caching, the request's KV is inserted into the cache (radix tree), so
   later requests can reuse it until LRU eviction frees it
   (`Scheduler._free_req_resources` → `CacheManager.cache_req(finished=True)`).

**3. Memory for Qwen2.5-1.5B, T = 200, l = 100.**

Config (HuggingFace `Qwen/Qwen2.5-1.5B/config.json`; the chapter does not list it):
`num_hidden_layers L = 28`, `num_key_value_heads n_kv = 2` (12 query heads),
`hidden_size 1536 / 12 heads → d_h = 128`, bf16 `s = 2`.

- Per token: `2 × 28 × 2 × 128 × 2 = 28,672 B = 28 KiB`.
- End of prefill: `200 × 28,672 = 5,734,400 B ≈ 5.47 MiB`.
- Each decode token adds: `28,672 B = 28 KiB` (`≈ 0.027 MiB`).
- End of generation: `(200 + 100) × 28,672 = 8,601,600 B ≈ 8.20 MiB`
  (= prefill + 100 × increment).

```text
== 2. Ch4 exercise 3: Qwen2.5-1.5B, T=200, l=100 (bf16)
per token = 2 x 28 layers x 2 kv_heads x 128 head_dim x 2 B = 28,672 B
|              end of prefill (T=200) |     5,734,400 | 5.47 MiB |
|                   each decode token |        28,672 |   28 KiB |
|         end of generation (T+l=300) |     8,601,600 | 8.20 MiB |
| model weights (1.54e9 params x 2 B) | 3,080,000,000 | 2.87 GiB |
KV at the end = 0.28% of the weights; a single request needs 107,422 tokens before its KV outweighs the weights.
```

What it shows: (a) growth is exactly linear. Every token costs the same 28 KiB
whether it came from the prompt or from decode, so the total is set by `T + l` and
generating 100 tokens adds 50% to the prefill footprint. (b) For one short request
the KV cache is tiny next to the 2.87 GiB of weights. It only dominates through
*scale*: long contexts (a 32K request is 0.88 GiB) or many concurrent requests
(16 GiB of KV holds ~600K tokens, which is 278 typical requests in lab section 4).
(c) GQA matters: with MHA (12 KV heads) the same request would need 6× more
(49.2 MiB).

**4. Problems with allocating contiguous GPU memory per request when many requests
run concurrently.**

- *Reservation waste:* the final length is unknown, so each request must reserve
  `max_seq_len`. In lab section 4 (16 GiB budget, 32K max length, real lengths
  274–3971), only **18** requests fit and **6.2%** of the reserved slots ever hold KV.
  With 16-token pages, **278** requests fit at 99.7% utilization (15.4× more). The
  same effect in this repo's README (`bench_batching.py`): with a 2048-token budget
  and `max_seq_len=1024`, contiguous pre-allocation fits 2 requests, paged ran 16.
- *External fragmentation:* requests of different sizes finish at different times and
  leave holes. A new request may not fit in any single hole even when total free
  memory is enough.
- *Growth means reallocation:* a buffer that outgrows its reservation must be moved
  (alloc + copy + free), which stalls the step.
- *No sharing:* identical system prompts or conversation prefixes are stored once per
  request.
- *Batching overhead:* variable-length contiguous buffers either get padded to a
  rectangle (wasted compute and memory) or gathered/compacted every step (copies).
  Requests that join and leave every iteration (continuous batching) make this worse.
- *Scheduling complexity:* preemption, resumption and eviction have to move or track
  whole buffers.

Paged allocation (fixed pages + a page table; Part II Ch7) removes all of these. The
only waste left is the last partly filled page per request.

**5. Could some or all of the KV cache live in CPU memory? Benefits and costs.**

*Benefits.* Host DRAM is usually several times larger and much cheaper than HBM, so
(a) more or longer contexts can be kept alive, (b) KV for idle conversations,
preempted requests or popular prefixes can be *kept* instead of evicted and
recomputed, and (c) GPU memory goes to the batch that is decoding right now, which
means higher concurrency.

*Costs.* The link is slow compared with HBM. PCIe 4.0 x16 gives ~32 GB/s per
direction (spec value; Ch3 names the interface but not the speed), against ~2 TB/s of
HBM, about 60× less. Decode reads the *whole* KV every step. So a 32K-token Qwen3-4B
context (4.5 GiB) would take ~150 ms per token to stream from the host, against
~2.4 ms from HBM. KV that decode needs must therefore be on the GPU. Offloading also
adds copy scheduling, pinned memory, synchronization, and complexity (consistency
with the prefix tree, eviction policies across tiers).

*What works in practice:* a **hierarchy**, not a split read. Keep active requests'
KV in HBM, put cold or reusable KV in host memory (and on disk), and **prefetch** it
back before it is needed, overlapping with compute (often layer by layer). Reloading
usually beats recomputing. Re-prefilling 32K tokens of Qwen3-4B costs
≈ 2 × 4×10⁹ × 32,768 ≈ 2.6×10¹⁴ FLOPs ≈ 0.84 s at 312 TFLOPS, while copying its
4.5 GiB back over PCIe takes ≈ 0.15 s. This is SGLang's HiCache (Part III Ch3,
tutorial reference "HiCache").
