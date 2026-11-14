# Part II · Chapter 7 — Paged KV Cache and Memory Management

> Tutorial: `part2/Chapter7_Paged-KV-Cache-and-Memory-Management.md` (concept only, no `_code` file; the reference's paging lives in `scheduler/cache.py`) · Lab: `labs/part2_ch07_paged_memory.py` · Bench: `benchmarks/bench_batching.py` (second table)

## Limitation → mechanism → validation

| Limitation (before paging) | Mechanism | Metric | Validated here by |
|---|---|---|---|
| A contiguous KV region reserved per request at `max_seq_len` wastes most of it (internal fragmentation), leaves unusable holes (external fragmentation), and cannot grow without copying | Paged KV: fixed-size pages from a free list (`CacheManager.free_slots`), a per-request block table (`page_table` row), pages allocated as the sequence grows (`allocate_paged`) | memory utilization, concurrency | **test**: `tests/test_scheduler.py::test_continuous_batching_matches_hf` (parametrized `page_size` 1 and 4: tokens == HuggingFace), `test_eviction_under_memory_pressure`, `tests/test_real_model.py` (page_size 4) · **bench**: `benchmarks/bench_batching.py` · **lab**: sections 1-5 |

README (`bench_batching.py`, Qwen3-0.6B): with a 2048-token KV budget and `max_seq_len=1024`,
contiguous pre-allocation fits **2** concurrent requests; paged allocation (page_size 16) ran **16**.
Lab section 1 reproduces this on the tiny model: peak decode batch 16, 1184 slots allocated, 89.5 %
of them holding real tokens.

## Key ideas

* **The contiguous dilemma.** A request's final length is unknown, so a contiguous design reserves
  the maximum up front. Three wastes follow: *internal* (reserved but never used, e.g. 2048 reserved
  for 200 used ≈ 90 % idle), *external* (enough free memory in total, but no single hole large
  enough), and *growth* (extending means reallocating and copying; sharing a prefix between
  sequences means duplicating it).
* **The tension paging resolves:** memory wants on-demand, scattered allocation; attention wants a
  logically contiguous K/V sequence. Paging keeps the logical order in a table and lets the
  physical storage scatter.
* **Three abstractions.** A *page/block* is the allocation unit (all equal, all interchangeable).
  A *slot* is one token's position inside a page. The *block table* maps a request's logical
  pages to physical pages. Address translation:
  `position p → (logical page p // b, offset p % b) → physical page → slot = page·b + offset`.
* **Waste becomes bounded.** Internal waste ≤ one partially filled page per request (≤ b−1
  tokens); external fragmentation disappears because any free page serves any request; growth is
  "append one page", no copy.
* **Page size trade-off.** Larger pages: more tail waste, fewer table entries / bigger contiguous
  reads for the kernel. Smaller pages: less waste, more indirection.
* **PagedAttention = blockwise attention + online softmax.** The kernel reads K/V one block at a
  time through the table and keeps a running max `m`, sum `ℓ` and unnormalized output `O`. Merging
  block j: `m' = max(m, m_j)`, `ℓ' = ℓ·e^{m−m'} + ℓ_j·e^{m_j−m'}`, `O' = O·e^{m−m'} + O_j·e^{m_j−m'}`,
  output `O/ℓ`. The common factor `e^{−m}` cancels, so the result equals softmax over all keys;
  only the floating-point summation order differs.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| physical pages | `python/minisgl/kvcache/mha_pool.py:MHAKVCache._kv_buffer` | dim 2 = page, dim 3 = slot in page |
| number of pages | `python/minisgl/engine/engine.py:Engine.__init__` | `config.num_pages or div_ceil(max_total_tokens, page_size)`; `EngineConfig.page_size` defaults to 1 |
| block table | `Engine.page_table`, one row per request (`TableManager` rows) | stores a **token slot per logical position**, not a page id per logical page |
| free page list | `python/minisgl/scheduler/cache.py:CacheManager.free_slots` | first slot of each free page; FIFO: freed pages are appended at the end |
| page → token slots | `CacheManager._page_to_token` | `page_start + arange(page_size)` |
| allocate on growth | `CacheManager.allocate_paged` | pages `ceil(cached_len/b) .. ceil(device_len/b)`; a partial last page is reused next step |
| take pages / evict | `CacheManager._allocate` | evicts radix leaves when short; `RuntimeError` if still short |
| fill the table | `cache.py:_write_page_table` | writes the new slots into each request's row |
| free | `CacheManager._free`, `lazy_free_region`, `cache_req` | `indices[::page_size]` turns slots back into page starts |
| capacity for admission | `CacheManager.available_size` | free pages × b + evictable cached tokens |
| page rounding in admission | `scheduler/prefill.py:PrefillAdder._try_allocate_one`, `scheduler/decode.py:DecodeManager.inflight_tokens` | `+ page_size` per request (Ch6) |
| accounting check | `CacheManager.check_integrity` | free pages + cached pages == num_pages, no page freed twice; valid only when no request is running |
| reading K/V through the table | `python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_metadata`, `forward` | `kv_indices = page_table[row, :device_len]`, gather, one SDPA call (no explicit blockwise loop) |
| page-aligned prefix sharing | `python/minisgl/kvcache/radix_cache.py:RadixPrefixCache` (`page_size`) | only whole pages are shared (Ch8) |

## This repo vs. the tutorial's reference code

The reference's paging code is `scheduler/cache.py` at 9a91cfa; compared directly:

* **Same allocator design**: `free_slots` of page starts, `allocate_paged` over
  `[ceil(cached_len/b), ceil(device_len/b))`, `_page_to_token`, `_write_page_table`, `_free` via
  `indices[::page_size]`, `lazy_free_region`. Dtypes are `int64` here (`int32` in the reference).
* **Attention does not read pages.** The reference's FlashAttention backend builds a per-request
  page-id table (`page_table[req, ::page_size] // page_size`) and lets the kernel walk blocks
  with online softmax. `TorchAttnBackend` gathers per-token slots into a dense padded tensor and
  calls `scaled_dot_product_attention` once. Paging is real at the allocator level; the blockwise
  online-softmax kernel is not re-implemented (lab section 4 demonstrates it in plain torch).
* **Pool sizing:** from `max_total_tokens` / `num_pages`, no profiling of free GPU memory, no
  dummy page.
* `_allocate` raises `RuntimeError` instead of `assert`; `check_integrity` also asserts that no page
  appears twice in `free_slots`; it is not called by the scheduler loop.
* **Duplicate-prefix fix** in `cache_req` (README "Differences"; Ch8): the reference frees the
  duplicate slots but leaves a still-running request's table pointing at them.

The tutorial's generic description also differs from mini-sglang (reference and this repo) in two
ways: the table holds **one entry per token** (a slot), not one physical block number per logical
block, and there is **no separate CPU copy** of the table: the scheduler writes the device
`page_table` directly and makes its decisions from `free_slots` and `available_size`.

## Common misconceptions

**Misconception:** paging reduces the KV bytes per token.
**Actually:** bytes per token are unchanged (`2 × layers × kv_heads × head_dim × dtype`). Paging
removes *reserved-but-unused* memory, which turns into more concurrent requests (lab section 1:
2 → 16 under the same 2048-token budget).

**Misconception:** with paging there is no internal fragmentation.
**Actually:** it is bounded, not zero: each request may waste up to `page_size − 1` slots in its
last page (lab section 2: waste 46 slots at page_size 4, 302 at 16, vs 5022 for a 256-token
contiguous reservation).

**Misconception:** a request's pages are physically adjacent if it was allocated alone.
**Actually:** pages come from a shared FIFO free list in whatever order earlier requests freed them
(lab section 3: A's pages `[0, 3, 6]`, D's `[9, 10, 11, 1, 4]`).

**Misconception:** paged attention is an approximation of attention.
**Actually:** online softmax is exact up to rounding (lab section 4: max error 2.2e-16 in
float64); and engine output is identical for page_size 1, 4 and 16 (lab section 5).

**Misconception:** `page_size = 1` makes the page table smaller.
**Actually:** mini-sglang's `page_table` always has one entry per token position
(`max_running_req + 1` × `max_seq_len`); `page_size` changes allocation granularity and
`free_slots` length, not the table's shape.

**Misconception:** a corrupted table entry would crash.
**Actually:** any slot index is a valid address in the pool, so a wrong entry silently reads
another request's K/V and produces wrong tokens (lab section 5).

## Lab

```bash
python labs/part2_ch07_paged_memory.py            # tiny model, ~4-6 s unloaded
python labs/part2_ch07_paged_memory.py --model real   # ~30 s
```

Sections 2-3 use a standalone `CacheManager` with fake `Req`s (lengths only), so the numbers come
from the real `allocate_paged` / `free_slots` / `cache_req` / `check_integrity`. Sections 1 and 5
run the engine.

```text
== 1. contiguous vs paged under a fixed KV budget
|                                       allocation | concurrent requests | slots reserved/allocated | used / allocated at peak |
|             contiguous, reserve max_seq_len each |                   2 |                     2048 |                        - |
| paged (page_size 16), measured peak decode batch |                  16 |                     1184 |                    89.5% |
  contiguous: free holes (start, size) = [(512, 256), (1280, 256)], total free = 512
  new request needs 384 contiguous tokens -> first fit: None
  paged: after r2, r4 finish there are 32 free pages; the 384-token request gets pages [32, 33, 34]..[85, 86, 87] -> fits

== 2. internal fragmentation for page_size 1 / 4 / 16 (real CacheManager)
|                    page_size | used slots | allocated slots | waste | bound n*(ps-1) | utilization |
|                            1 |       3170 |            3170 |     0 |              0 |      100.0% |
|                            4 |       3170 |            3216 |    46 |             96 |       98.6% |
|                           16 |       3170 |            3472 |   302 |            480 |       91.3% |
| contiguous (max_seq_len 256) |       3170 |            8192 |  5022 |              - |       38.7% |

== 3. block table (page_table rows) after interleaved growth
| req | row | tokens | logical page -> physical page | token -> slot                                  |
|   A |   0 |     12 |                     [0, 3, 6] | [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27]  |
|   C |   2 |     12 |                     [2, 5, 8] | [8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35] |
|   D |   3 |     20 |             [9, 10, 11, 1, 4] | [36, ..., 47, 4, 5, 6, 7, 16, 17, 18, 19]     |
A's token 9: logical page 2, offset 1 -> physical page 6 -> slot 25 (page_table says 25)
```

What to look for: section 1, the paged request spans two former holes (pages 32-47 and 80-87),
which contiguous allocation cannot do. Section 2, waste stays under the `n·(b−1)` bound.
Section 3, round-robin growth (3 tokens per request per round) interleaves the requests' pages,
and D receives the three never-used pages first, then B's freed pages 1 and 4.

## Exercises

### Concept chapter

**1. Draw the data flow: tokens → logical blocks → block table → slots in physical blocks.**
Using lab section 3 (b = 4, request A, 12 tokens): tokens 0-3, 4-7, 8-11 form logical pages 0, 1, 2.
A's block table maps them to physical pages 0, 3, 6 (in mini-sglang the row stores the slots
directly: `[0..3, 12..15, 24..27]`). Token 9 → logical page 2, offset 1 → physical page 6 → slot
6·4 + 1 = 25 → in each layer, `flat_k_cache(layer)[25]` and `flat_v_cache(layer)[25]`, i.e.
`_kv_buffer[0 or 1, layer, 6, 1]`. In the engine: `CacheManager.allocate_paged` takes pages from
`free_slots` and `_write_page_table` fills the row; `Scheduler._prepare_batch` reads
`out_loc = page_table[row, positions]` (write path); `TorchAttnBackend.prepare_metadata` reads
`page_table[row, :device_len]` (read path).

**2. Roles of Block, Slot and Block Table; their hierarchy; what if one logical block's physical
number is corrupted?** Block (page) = unit of allocation and freeing, owned by one request (or by
the radix tree). Slot = one token's position inside a block, the unit `store_kv` writes. Block
table = per-request map from logical to physical block. Hierarchy: request → block table →
blocks → slots. A wrong entry makes every access to that logical block go to another physical
block: reads return someone else's K/V (or stale data) and writes overwrite it. Nothing detects
it; outputs simply change, and it can also corrupt the other owner's cache or, after frees, the
allocator's accounting. Lab section 5 swaps logical page 0 between two running requests before
their first decode step:

```text
|                         run |                                 A tokens |                              B tokens |
|                paged engine | [159, 399, 159, 124, 410, 159, 326, 343] |     [2, 266, 416, 2, 28, 28, 28, 405] |
| swap page 0 of A <-> page 0 of B | [159, 112, 457, 209, 501, 30, 445, 460] | [2, 17, 220, 466, 487, 283, 138, 193] |
|  swap pages 0 <-> 1 inside A | [159, 399, 159, 124, 410, 159, 326, 343] |     [2, 266, 416, 2, 28, 28, 28, 405] |
```

The first token is unchanged (it came from prefill, before the swap); everything after differs.
The second row is a useful subtlety: permuting pages *within* one request's fully visible history
changes nothing, because RoPE is already applied to cached K and decode attention sums over all
history keys regardless of order. The table matters for *which* data is read and for the causal
boundary (`j <= cached_len + t`), not for order among visible keys.

**3. Why is attention over non-contiguous blocks exactly standard attention? Explain with the
online-softmax update.** Scores `s_r = q·k_r/√d` depend only on q and the individual key, and the
output is a weighted sum of values, so blocks can be processed in any grouping. Each block j is
summarized relative to its own max: `(m_j, ℓ_j = Σ e^{s−m_j}, O_j = Σ e^{s−m_j} v)`. To merge,
rescale both the running pair and the block pair to `m' = max(m, m_j)` with the *same* factor for
`ℓ` and `O` (`e^{m−m'}`, `e^{m_j−m'}`), then add. After the last block, `O/ℓ =
Σ e^{s−m} v / Σ e^{s−m}`, and `e^{−m}` cancels: exactly `softmax(s)·V`. The block table only
decides where each block is read from; it does not change any q, k, v value. Fully masked blocks
contribute nothing and can be skipped. Lab section 4 runs this over K/V stored in shuffled
physical blocks:

```text
max |paged online-softmax - full softmax| = 2.22e-16
causal query at position 19 (blocks 3, 4 fully masked, skipped): max error 1.67e-16
without the e^(m - m') rescaling (averaging block outputs): error 2.74e-01
```
