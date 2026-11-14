# Part II · Chapter 8 — RadixAttention and Prefix Caching

> Tutorial: `part2/Chapter8_RadixAttention-and-Prefix-Caching.md`, `part2/Chapter8_RadixAttention-and-Prefix-Caching_code.md` · Lab: `labs/part2_ch08_radix_tree.py` · Bench: `benchmarks/bench_prefix_cache.py`

## Limitation → mechanism → validation

| Limitation | Mechanism | Validation here |
|---|---|---|
| Requests that share a prefix (system prompt, tool list, earlier turns) each recompute its K/V; KV pages die with the request | KV ownership moves from the request to a shared radix tree of `token ids → slots`; match → lock → allocate miss → insert → unlock → evict/free | **Correctness:** `tests/test_radix_cache.py` (split, lock, LRU, cascade, page alignment), `tests/test_scheduler.py` (radix vs naive outputs equal HF, `check_integrity` after every run). **Benefit:** `benchmarks/bench_prefix_cache.py`; README: 12 requests sharing a 512-token prompt, computed prefill tokens 6376 → 744 (hit rate 88.3%), mean TTFT 1.203 s → 0.304 s. **Mechanics:** the lab prints the tree. |

## Key ideas

* **Only prefixes are reusable.** Causal attention makes the K/V at position *i*
  a function of tokens `0..i` (and of position *i* through RoPE). Two requests can
  share K/V only for an identical run of tokens starting at position 0. Equal
  suffixes in different contexts share nothing.
* **RadixAttention is a cache layer, not new attention math.** The attention
  kernel is unchanged; it just reads some K/V from slots another request filled.
* **Tree = index, pool = data.** A node stores `key` (token ids) and `value`
  (physical slots) of equal length. The full sequence of a node is the
  concatenation of keys on the root path; its slots are the concatenation of
  values. The K/V tensors themselves stay in `MHAKVCache`.
* **Path compression.** A branch-free run of tokens is one node. Nodes are
  created only at real divergence points (`split_at`), so a 4k-token system
  prompt costs one node, not 4k.
* **Split moves no data.** `split_at(pos)` slices the key/value index tensors.
  The new parent inherits `ref_count`, because anyone protecting the old node
  also protects its prefix.
* **Matched ≠ owned.** A match result is only an address. Until `lock_handle`
  increments `ref_count` on every node up to the root, the path may be evicted.
  Locking moves those tokens from `evictable_size` to `protected_size`.
* **Lifecycle:** `match` (find reusable prefix) → `lock` (protect it) →
  `allocate` (only the miss part) → forward writes K/V → `insert` (publish;
  duplicates are freed) → `unlock` → later `evict` under memory pressure.
* **evict vs free.** `evict` removes unreferenced *leaves* of the tree, LRU
  first, and returns their slots. `free` returns slots straight to the page
  allocator: duplicates found at insert time and the unaligned/never-inserted tail
  of a finished request. Shared prefixes are never freed when a request ends;
  they are only unlocked.
* **Whole pages only.** With `page_size > 1` matching and insertion are aligned
  down to page multiples and children are keyed by their first full page. A
  partial page is private to its request.
* **The last prompt token is never matched**, so prefill always has at least one
  token to run and produce the first output token's logits.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| node (key, value, ref_count, timestamp) | `python/minisgl/kvcache/radix_cache.py:RadixTreeNode` | `set_key_value`, `set_parent`, `is_leaf`, `get_match_len`; `__lt__` orders by timestamp for the eviction heap |
| node split | `radix_cache.py:RadixTreeNode.split_at` | new parent gets `key[:pos]`, inherits `ref_count`; `self` keeps the suffix |
| child key | `radix_cache.py:_get_key_fn` | first token (`page_size == 1`) or tuple of the first page |
| common-prefix length | `radix_cache.py:_fast_compare_key` | pure PyTorch (`torch.nonzero`) instead of the reference's C++ helper |
| longest-prefix walk | `radix_cache.py:RadixPrefixCache._tree_walk` | aligns the match down to `page_size`, splits when it stops inside a node, refreshes timestamps on the path |
| match → handle | `RadixPrefixCache.match_prefix`, `RadixCacheHandle` | handle = `(cached_len, node)` |
| slots of a match | `RadixCacheHandle.get_matched_indices` | walk to root, reverse, `torch.cat` |
| lock / unlock | `RadixPrefixCache.lock_handle(handle, unlock=...)` | updates `ref_count`, `evictable_size`, `protected_size` up to the root |
| insert | `RadixPrefixCache.insert_prefix` | returns `(prefix_len already cached, handle over the aligned length)` |
| LRU leaf eviction | `RadixPrefixCache.evict`, `_collect_evictable_leaves` | heap of `ref_count == 0` leaves; a parent that becomes such a leaf is pushed during the same call |
| tree invariants | `RadixPrefixCache.check_integrity` | page alignment, child `ref_count` ≤ parent, sizes add up |
| policy switch | `python/minisgl/kvcache/__init__.py:create_prefix_cache` | `"radix"` or `"naive"` (`kvcache/naive_cache.py`: never matches, never inserts) |
| interface | `python/minisgl/kvcache/base.py:BasePrefixCache`, `BaseCacheHandle`, `SizeInfo` | |
| match without the last token | `python/minisgl/scheduler/cache.py:CacheManager.match_req` | slices `input_ids[: input_len - 1]` |
| capacity | `CacheManager.available_size` | `evictable_size + len(free_slots) * page_size` |
| allocate (evict on demand) | `CacheManager._allocate`, `CacheManager.allocate_paged` | evicts `(needed - free) * page_size` tokens, appends `evicted[::page_size]` to `free_slots` |
| publish / release | `CacheManager.cache_req(req, finished=...)` | the four regions of the page-table row are documented in its docstring |
| free | `CacheManager._free`, `CacheManager.lazy_free_region` | |
| leak check | `CacheManager.check_integrity` | free pages + cached pages == `num_pages`; no page freed twice |
| admission + lock | `python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one` | match, check, lock, check again, copy matched ids and slots into the row |
| publish after prefill / release at finish | `python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data`, `Scheduler._free_req_resources` | `cache_req(finished=False)` after a non-chunked prefill; `cache_req(finished=True)` on finish or abort |
| hit statistics | `scheduler/prefill.py:PrefillStats` (`prompt_tokens`, `cached_tokens`) | used by the benchmark and the lab |

## This repo vs. the tutorial's reference code

* **`cache_req` duplicates (code exercise 4).** Both free
  `page_indices[old_handle.cached_len : prefix_len]`. This repo additionally,
  when the request keeps running (`finished=False`), rewrites that range of its
  `page_table` row to the cached copy's slots
  (`new_handle.get_matched_indices()`). Without that, the running request's row
  still names freed slots that the allocator may hand to someone else.
* **`match_req(input_ids)`** takes the token tensor; the reference takes the
  request.
* **Admission reserves one extra page per request.** `_try_allocate_one` uses
  `estimated_len = extend_len + output_len + page_size`, `_add_one_req` adds
  `remain_len + output_len + page_size` to `reserved_size`, and
  `DecodeManager.inflight_tokens` adds `page_size` per running request. The
  reference excerpt has no `+ page_size`; it covers the partially filled last
  page.
* **`_tree_walk` stops when fewer than `page_size` tokens remain** before calling
  `key_fn`, so a short tail is never looked up.
* **`evict` raises** `RuntimeError` if asked for more than `evictable_size`, and
  `_allocate` raises "out of KV cache" if eviction still leaves too few pages.
* **`CacheManager.check_integrity`** also asserts that `free_slots` has no
  duplicates.
* **No C++ radix helper, no CUDA store kernel.** `_fast_compare_key` and
  `MHAKVCache.store_kv` are plain PyTorch.
* **`page_size` is passed explicitly** to `create_prefix_cache(device, type,
  page_size)`.
* Scheduling is FCFS in both (no `lpm`/`dfs-weight` prefix-aware ordering as in
  SGLang).

## Common misconceptions

* **Misconception:** RadixAttention changes how attention is computed.
  **Actually:** attention is the same; only which slots the page table points at
  changes. `attention/torch_backend.py` has no radix code.
* **Misconception:** the radix tree stores K/V.
  **Actually:** it stores slot indices (`RadixTreeNode.value`); the floats live in
  `MHAKVCache._kv_buffer`.
* **Misconception:** a match protects the prefix.
  **Actually:** only `lock_handle` does. `_try_allocate_one` locks immediately
  after the first capacity check; a failed second check unlocks again.
* **Misconception:** a fully cached prompt needs no prefill.
  **Actually:** `match_req` drops the last token, so `cached_len ≤ len - 1` and
  one token still runs to produce the first output logits (lab section 7).
* **Misconception:** `evict(n)` evicts exactly `n` tokens.
  **Actually:** it evicts whole leaves until at least `n` are freed (lab: `evict(1)`
  returned a 2-slot leaf).
* **Misconception:** when a request finishes its pages are freed.
  **Actually:** the aligned part of `prompt + output[:-1]` is inserted into the tree
  and merely unlocked; only duplicates and the unaligned tail are `_free`d.
* **Misconception:** matching is read-only.
  **Actually:** `_tree_walk` splits nodes and refreshes timestamps even for a
  request that is then not admitted (lab section 7: re-matching a prompt split
  `[300 301 302]` into `[300 301]` + `[302]`).

## Lab

```bash
python labs/part2_ch08_radix_tree.py            # ~4 s (tiny model); --model real ~5 s
```

Look for: the split after inserting B (`insert B returned prefix_len=3`), ref
counts along a locked path, LRU and cascade eviction, page alignment, the tree
after each `generate` with a shared 16-token system prompt, and the two
scheduler corner cases.

```
-- after insert C -> [101 11 12] split again at 2 (tutorial Figure 4)
root
└─ key=[101 11] slots=0-1 ref=0 lru#4
   ├─ key=[12] slots=2 ref=0 lru#2
   │  ├─ key=[21 22] slots=3-4 ref=0 lru#1 leaf
   │  └─ key=[31 32] slots=13-14 ref=0 lru#3 leaf
   └─ key=[99] slots=22 ref=0 lru#5 leaf
...
unlock B, evict(5) -> [13, 14, 2, 0, 1]: leaf [31 32] first, then its parent [12] became a leaf,
...
| generate | reqs | prompt tokens | cached (reused) | prefill computed |
|        1 |    1 |            19 |               0 |               19 |
|        2 |    1 |            18 |              16 |                2 |
|        3 |    2 |            37 |              32 |                5 |
```

## Exercises

### Concept chapter

**1. Why a radix tree (path compression) instead of a plain trie?**
A trie spends one node, one dict lookup and one pointer per token. LLM prefixes
are long (system prompts, multi-turn history) and branch rarely, so almost all of
those nodes have one child. A radix tree stores a branch-free run as one node:
node count is proportional to the number of *branch points*, and a match does one
dict lookup per node plus one vectorized compare (`_fast_compare_key`) per node.
Splits happen only where requests actually diverge. Lab section 6: three
generates over a 16-token system prompt produce a tree whose shared prefix is a
single 16-token node.

**2. Why can eviction only start at leaves?**
A node's tokens are the prefix of every descendant. Removing an internal node
would orphan the descendants: their slots would still be in the tree but their
sequence could never be matched (a match must walk from the root), and
`get_matched_indices` would return a path with a hole. Evicting leaves first
shrinks the tree from the tips and keeps every remaining path valid; a parent
becomes evictable once its last child is gone. `RadixPrefixCache.evict` only
heaps leaves with `ref_count == 0` and pushes a parent when it becomes one (lab
section 4: `[31 32]`, then `[12]`, then `[101 11]`).

**3. evict vs free during scheduling.**
`evict` acts on the *tree*: called from `CacheManager._allocate` only when
`free_slots` is short, it detaches LRU unreferenced leaves and returns their
slots, which `_allocate` appends to `free_slots`. It is lazy (only under memory
pressure) and only touches unlocked cache. `free` (`CacheManager._free`) acts on
the *allocator*: slots that are not (or not any more) owned by the tree go back to
`free_slots` immediately: duplicates found at insert (`[old.cached_len,
prefix_len)`) and, at finish, the tail beyond the aligned inserted length. Both
end in `free_slots`; the difference is ownership and timing.

### Code walkthrough

**1. Why does `match_req` use `input_ids[:input_len-1]`?**
Prefill computes logits only for positions it runs, and the first output token
is predicted from the last prompt position. If the whole prompt matched,
`extend_len = 0`: nothing to run, no logits, and the request could not start
decoding. Dropping one token guarantees `extend_len ≥ 1`. Code:
`scheduler/cache.py:CacheManager.match_req`. Lab section 7:

```
prompt len=19: prefix_cache.match_prefix -> 19, CacheManager.match_req -> 18
re-running the same prompt: cached_len=18 = len-1; prefill still runs 1 token ...
```

With `page_size > 1` the cached length is further aligned down to a page
multiple, so more than one token may run.

**2. Why check `available_size` before and after `lock(handle)`?**
`available_size = evictable_size + free pages × page_size` counts the matched
prefix as reclaimable while it is unlocked. Locking moves it to `protected_size`,
so `available_size` drops by up to `cached_len`. The first check is cheap and
avoids locking a hopeless request; the second checks the budget that is actually
left once the prefix may no longer be evicted to make room for the request's own
miss. If it fails, the handle is unlocked again. In this repo:
`scheduler/prefill.py:PrefillAdder._try_allocate_one` (same structure as the
reference, with `+ page_size` in `estimated_len`). Lab section 8 builds a case
where check 1 passes and check 2 fails:

```
cached_len=16 estimated_len=13 reserved_size=243
available_size before lock=256 (free=222 + evictable=34); after lock=240
check 1: 13+243=256 <= 256  -> passes
check 2: 13+243=256 <= 240  -> fails (the matched prefix moved from evictable to protected)
_try_allocate_one -> None, and the handle was unlocked again (protected_size unchanged)
```

With an idle engine the second check cannot fail: the scheduler caps
`output_len` so `input_len + output_len ≤ capacity − 2 pages`
(`Scheduler._process_one_msg`). It matters when running requests reserve KV
(`reserved_size = DecodeManager.inflight_tokens`) or other paths are locked.

**3. Why does `get_matched_indices` walk to the root and then reverse?**
Each node stores only its own segment's slots, and nodes only know their parent
(children are keyed by first token/page, which the handle does not have). So the
only path from a handle's node to its full sequence is upward; collecting values
upward yields them deepest-first, and `values.reverse()` + `torch.cat` restores
sequence order. Code: `kvcache/radix_cache.py:RadixCacheHandle.get_matched_indices`.

**4. What does `page_indices[old_handle.cached_len:cached_len]` represent?**
Two requests with the same uncached prefix are admitted before either has
inserted it (e.g. in the same prefill batch). Both compute it into their own
slots. The first `insert_prefix` adds the node; the second returns
`prefix_len > old_handle.cached_len`: its slots for `[old.cached_len,
prefix_len)` duplicate K/V that the tree now owns. If they are not freed they
are owned by nobody: not in the tree, not in `free_slots`, so
`check_integrity` reports a leak and the pool shrinks permanently. This repo
also repoints the still-running request's row (see differences above). Lab
section 9:

```
| uid | old_handle.cached_len | insert prefix_len | page_table row before | row after |
|   6 |                     0 |                 0 |                 37-48 |     37-48 |
|   7 |                     0 |                12 |                 49-60 |     37-48 |
outputs identical: True; check_integrity: OK (free pages + cached pages == 256)
```

**5. Why must eviction start from unreferenced leaves, and who returns pages to the free list?**
`ref_count > 0` means a running request's page table points at the node's
slots (lock covers the whole root path, so a locked leaf implies locked
ancestors). Evicting it would let the allocator hand those slots to another
request while they are still being read. Leaves-only keeps the remaining tree
consistent (exercise 2 above). `RadixPrefixCache.evict` only detaches nodes and
returns slot indices; `CacheManager._allocate` appends `evicted[::page_size]`
(first slot of each page) to `free_slots`.
