# Part II · Chapter 4 — KV Cache Optimization

> Tutorial: `part2/Chapter4_KV-Cache-Optimization.md`, `part2/Chapter4_KV-Cache-Optimization_code.md` · Lab: `labs/part2_ch04_kv_cache.py` · Bench: `benchmarks/bench_kv_cache.py`

## Limitation → mechanism → validation

| Limitation (Ch3 version) | Mechanism | Metric | Validated here by |
|---|---|---|---|
| Every decode step re-runs the whole sequence (`llm/naive.py`), so per-token cost grows with length | KV cache: `MHAKVCache` pool + `page_table` + per-step `out_loc`, store-then-attend | computation, TPOT | **test**: `tests/test_scheduler.py` and `tests/test_model.py::test_incremental_extend_matches_full_prefill` (cached extend == full prefill, tokens == HuggingFace) · **bench**: `benchmarks/bench_kv_cache.py` · **lab**: sections 3, 5, 6 |

README numbers (`bench_kv_cache.py`, Qwen3-0.6B, 128-token prompt, outputs identical): TPOT without a
cache grows 325.6 → 356.1 → 414.8 ms for 16/64/128 output tokens; with the cache it stays at
80.6 / 68.3 / 69.1 ms (4.0x → 6.0x). Growing vs. flat TPOT is the signature of the O(n²) → O(n)
change in K/V work.

## Key ideas

* **The forward pass is stateless.** Without help, step *t* must re-project every earlier token
  into K and V in every layer, although those K/V never change. The KV cache moves that state
  *outside* the model: the engine keeps it and passes it back each step.
* **What is cached:** per layer, per KV head, the K (after RoPE) and V of every computed token.
  Not Q (only the current token's query is needed), not attention weights (recomputed for each
  new query), not logits, not hidden states.
* **Cost accounting.** K/V projections: Σ T_t = O(n²) without a cache → O(n) with it. Attention:
  each step goes from a T×T score matrix to a 1×T row, so per-step O(T²) → O(T), but the
  cumulative decode attention is still O(n²) because every new query reads the whole history.
  Memory goes from 0 to O(T) per request.
* **Prefill vs. decode.** Prefill computes and writes K/V for all uncached prompt tokens at once
  (`extend_len` = prompt length; many queries, causal triangle; compute-heavy). Decode writes
  exactly one new K/V row per request and reads all of its history (`extend_len` = 1;
  memory-bound).
* **Three length fields drive everything** (`core.py:Req`): `[0, cached_len)` already in the pool,
  `[cached_len, device_len)` computed this step, `extend_len = device_len - cached_len`.
  `complete_one()` after each forward: `cached_len = device_len; device_len += 1` (the sampled
  token occupies the next position but has no K/V until the next step processes it).
* **Three objects, three jobs.** `MHAKVCache` holds numbers only. `page_table[row, pos]` maps a
  request's logical position to a physical **token slot**. The prefix cache (`naive` / `radix`)
  decides which slots may be reused by other requests. `CacheManager` owns the free list.
* **Order inside one step:** allocate pages (`allocate_paged`) → read `out_loc =
  page_table[(row, position)]` → in every layer `store_kv(k, v, out_loc)` *then* attend, so the
  new token can see itself.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| pool interface | `python/minisgl/kvcache/base.py:BaseKVCachePool` | `k_cache`, `v_cache`, `store_kv`, `device`, `dtype`, `num_layers` |
| physical pool | `python/minisgl/kvcache/mha_pool.py:MHAKVCache.__init__` | `_kv_buffer` `[2, layers, pages, page_size, local_kv_heads, head_dim]`; KV heads split with `div_even(..., allow_replicate=True)` |
| per-layer views | `MHAKVCache.k_cache`, `v_cache`, `flat_k_cache`, `flat_v_cache` | views, no copy; flat view is `[pages*page_size, heads, dim]` |
| write new K/V | `MHAKVCache.store_kv` | plain advanced-index assignment at `out_loc` |
| pool + policy factory | `python/minisgl/kvcache/__init__.py:create_kvcache_pool`, `create_prefix_cache`, `SUPPORTED_CACHE_MANAGER` | `"naive"` / `"radix"` are reuse policies, not layouts |
| pool + page table creation | `python/minisgl/engine/engine.py:Engine.__init__` | `num_pages = config.num_pages or div_ceil(max_total_tokens, page_size)`; `page_table` `(max_running_req + 1, max_seq_len)` int64 |
| length bookkeeping | `python/minisgl/core.py:Req.extend_len`, `Req.complete_one` | `complete_one` is called by `Engine.forward_batch` for every req in the batch |
| free pages | `python/minisgl/scheduler/cache.py:CacheManager.free_slots` | first token slot of each free page |
| page → token slots | `CacheManager._page_to_token` | slot 8, page_size 4 → `[8, 9, 10, 11]` |
| allocate pages for `[cached_len, device_len)` | `CacheManager.allocate_paged`, `_allocate`, `_write_page_table` | pages only on page boundaries |
| prefix match (never the last prompt token) | `CacheManager.match_req` | prefill must still compute the last token's logits |
| admit + copy matched prefix | `python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one` | copies token ids to `token_pool`, matched slots to `page_table` |
| copy uncached token ids | `PrefillAdder._add_one_req` | pages are *not* allocated here |
| positions, `out_loc`, metadata | `python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch` | `allocate_paged` first, then `out_loc = page_table[input_mapping]` |
| RoPE then attention | `python/minisgl/models/utils.py:RopeAttn.forward` | K is cached *after* rotation |
| store-then-attend, mask | `python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_metadata`, `forward` | `kv_indices` gathered from `page_table`; key `j` visible to query `t` iff `j <= cached_len + t` and `j < device_len` |
| release / keep | `Scheduler._free_req_resources`, `CacheManager.cache_req` | freed slots are not zeroed |
| no-cache baseline | `python/minisgl/llm/naive.py:generate_without_kv_cache` | runs every step as a full prefill (`cached_len = 0`) |

## This repo vs. the tutorial's reference code

Checked against the reference at commit 9a91cfa.

* **No `attention/fa.py`.** The backend is `TorchAttnBackend`: it gathers each request's K/V with
  the *token slots* in `page_table` (`kv_indices`) and runs one padded
  `scaled_dot_product_attention` with an explicit mask. The reference converts the table to
  page IDs (`page_table[req, ::page_size] // page_size`) because FlashAttention's paged kernel
  reads whole pages; this repo never needs page IDs.
* **No `store_cache` kernel.** `store_kv` is `flat_k_cache(layer)[out_loc] = k`.
* **Pool size comes from `max_total_tokens`** (or `num_pages`), not from measured free GPU memory,
  and there is **no `+1` dummy page** (no CUDA graphs). `page_table` is `int64`, width
  `max_seq_len` (not aligned up to 32), and `max_seq_len` is also capped by
  `max_position_embeddings`. `free_slots` is `int64` too.
* `MHAKVCache` allocates with `torch.zeros` (reference: `torch.empty`) and adds `flat_k_cache` /
  `flat_v_cache` / `num_bytes`.
* `_prepare_batch` builds positions and `(row, position)` indices inline; there is no
  `graph_runner.pad_batch`, no `_make_positions` / `_make_input_tuple` helpers. Same result.
* `CacheManager.match_req` takes the token-id tensor, not a `PendingReq`; `cache_req` also repoints a
  running request's page table at the cached copy when it frees duplicate slots (README
  "Differences"; Chapter 8).

## Common misconceptions

**Misconception:** with a KV cache a decode step is O(1).
**Actually:** only the K/V projection of the new token is O(1). Attention still reads all T
cached keys (lab section 6: the decode row for request 0 has 6 visible keys); cumulative decode
attention stays O(n²).

**Misconception:** `out_loc` is a request's page-table row.
**Actually:** it is the list of write slots for *this step's new tokens only*, concatenated over
the batch; its length is Σ `extend_len` (lab section 6: 8 slots in prefill, 2 in decode).

**Misconception:** `page_table` stores page numbers.
**Actually:** in both this repo and the reference it stores a raw token slot per logical position.
Only the reference's FlashAttention path derives page IDs right before the kernel call.

**Misconception:** the token sampled by prefill already has K/V in the cache.
**Actually:** its K/V is computed by the next step (lab section 3: the first decode has
`positions = [6]` for a 6-token prompt). The final output token of a request is never run
through the model at all.

**Misconception:** `naive` and `radix` are two cache layouts.
**Actually:** both use the same `MHAKVCache`; they differ only in whether finished slots can be
matched by later requests.

**Misconception:** freeing a slot erases its K/V.
**Actually:** `_free` only appends the page to `free_slots`; the values stay until a new owner
overwrites them, which is why a stale `page_table` entry silently corrupts output instead of
crashing (lab section 4; Chapter 7 lab section 5).

## Lab

```bash
python labs/part2_ch04_kv_cache.py            # tiny model, ~4-6 s
python labs/part2_ch04_kv_cache.py --model real
```

What to look for:

1. Pool shape `(2, 3, 64, 4, 2, 16)` and `bytes per token = 2 x layers x kv_heads x head_dim x dtype`.
2. `free_slots` value 8 → page 2 → slots `[8, 9, 10, 11]`.
3. The per-step trace: prefill `extend_len = 6`, then decode `extend_len = 1` with
   `positions = [cached_len]`; free pages drop only when `device_len` crosses a page boundary.
4. An unallocated row gives `out_loc = [0, 0, 0, 0, 0, 0]`.
5. Same tokens with and without the cache; 216 vs 21 token positions computed.
6. Two requests in one batch: packed `positions`, `out_loc`, and the per-request mask.

```text
|   phase | cached_len | device_len | extend_len |          positions |            out_loc | ... | free pages |
| prefill |          0 |          6 |          6 | [0, 1, 2, 3, 4, 5] | [0, 1, 2, 3, 4, 5] | ... |         62 |
|  decode |          6 |          7 |          1 |                [6] |                [6] | ... |         62 |
|  decode |          7 |          8 |          1 |                [7] |                [7] | ... |         62 |
|  decode |          8 |          9 |          1 |                [8] |                [8] | ... |         61 |

== 6. two requests in one batch: packed out_loc and the attention mask
prefill: reqs (uid, cached_len, device_len) = [(3, 0, 5), (4, 0, 3)]
  positions = [0, 1, 2, 3, 4, 0, 1, 2]
  out_loc   = [36, 37, 38, 39, 40, 44, 45, 46]   (len = sum of extend_len = 8)
decode: reqs (uid, cached_len, device_len) = [(3, 5, 6), (4, 3, 4)]
  positions = [5, 3]
  out_loc   = [41, 47]   (len = sum of extend_len = 2)
  req 0: t=0 (pos 5): 111111
  req 1: t=0 (pos 3): 1111..
slot 36: layer-0 K != layer-1 K -> True
```

Note request 1's prefill starts at slot 44 although slot 41 is free: page 10 (slots 40-43) belongs
to request 0, and the next free *page* is 11. Request 0's decode then writes slot 41 in its own
partially filled page.

## Exercises

### Concept chapter

**1. Why are historical K/V recomputed at every decode step without a cache?**
Because the model's forward pass keeps no state: its only input is the token sequence, so to
predict token *t+1* it must embed and project all *t* tokens again in every layer. The K/V of
token *i* depend only on tokens `≤ i`, so they are identical every time, which makes the work
pure waste. `llm/naive.py:generate_without_kv_cache` does exactly this (`cached_len=0` every
step); lab section 5 counts 216 token positions for a 6-token prompt and 16 outputs
(Σ_{i=0}^{15} (6+i)) versus 21 (6 + 15) with the cache, with identical tokens.

**2. Why cache only K and V, not Q, the attention matrix or logits?**
A future step's output depends on old tokens only through their keys (to score against the new
query) and values (to be mixed). An old token's query was used once, to compute that old token's
own output, and is never needed again. Attention weights depend on the *new* query, so they
cannot be reused. Logits exist only for the last position and are consumed by sampling.
In the code, `RopeAttn.forward` computes q, k, v for this step's tokens only and
`store_kv` persists only k and v.

**3. How do prefill and decode read and write the cache?**
Prefill writes `extend_len` rows (the whole uncached prompt) and every prompt query reads the
keys up to itself, plus any matched prefix (`j <= cached_len + t`): many queries, a matrix-matrix
problem, compute-bound. Decode writes one row per request and its single query reads the request's
whole history: a vector-matrix problem dominated by reading weights and K/V, memory-bound. Lab
sections 3 and 6 show `extend_len` 6 vs 1 and the triangular vs single-row masks.

**4. What does "O(n²) → O(n)" refer to, and why is cumulative attention still O(n²)?**
It counts K/V projections: Σ_{t} T_t = O(n²) without a cache, Σ 1 = O(n) with it (or, equally,
token positions run through the MLP/projections; lab section 5's 216 vs 21). Attention per step
drops from a T×T score matrix to a 1×T row, but over n steps Σ T_t is still O(n²) because each
new query must read every cached key. Only changing attention itself (sliding window, sparse,
linear) removes that term.

### Code walkthrough

**1. What are dims 0-3 of `_kv_buffer`, and why is `layer_id` still needed?**
`MHAKVCache._kv_buffer` is `[2, num_layers, num_pages, page_size, local_kv_heads, head_dim]`:
dim 0 selects K or V, dim 1 the layer, dim 2 the page, dim 3 the slot inside the page. A token
slot (`page * page_size + offset`, the value stored in `page_table`) is the same number in every
layer, but each layer has its own K/V for that token, so a read is
`flat_k_cache(layer_id)[slot]`. Lab section 1 prints the shape; section 6 shows slot 36 holds
different K in layer 0 and layer 1.

**2. `page_size = 4`, a `free_slots` value is 8: which page and slots?**
Page 2 (8 // 4); `CacheManager._page_to_token` expands it to `[8, 9, 10, 11]`. Lab section 2:
`free_slots value 8 -> page 2 -> token slots [8, 9, 10, 11]`.

**3. Why must `allocate_paged` run before `out_loc = page_table[input_mapping]`?**
`out_loc` is *read* from the page table, and only `CacheManager.allocate_paged` *writes* the new
entries for `[cached_len, device_len)`. Reading first returns whatever the row holds: zeros for a
fresh row (every K/V written to slot 0) or slots of the row's previous owner (which may now
belong to another request or to the radix tree), silently corrupting another request's cache.
Lab section 4: `unallocated page_table row 1, positions 0..5 -> out_loc = [0, 0, 0, 0, 0, 0]`.
In this repo the order is in `Scheduler._prepare_batch`.
