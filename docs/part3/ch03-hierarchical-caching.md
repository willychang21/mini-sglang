# Part III · Ch3 — Hierarchical Caching (HiCache)

Course text: `part3/Chapter3_Hierarchical-Caching.md`. Lab:
[`labs/part3/hicache_cost_model.py`](../../labs/part3/hicache_cost_model.py).

## Key ideas

- The paged KV cache answers "how do I allocate GPU KV without fragmentation". The radix
  prefix cache answers "how do I avoid recomputing a shared prefix". Neither answers "where
  do I keep a computed prefix when the GPU is full". HiCache adds tiers: **L1** GPU, **L2**
  host DRAM of this instance, and **L3** optional external storage (possibly shared across
  instances).
- A request matches the longest prefix across tiers, but only a **contiguous** prefix whose
  K/V is ready can be reused. Data in L2 or L3 must be copied back to the GPU first. L3 is
  looked up page by page, with hash keys that chain the page's tokens and its whole prefix.
- Lifecycle: compute in L1 → **back up** (write-through / write-back / selective) → evict from
  L1 → **restore** (load back) on a hit. Backup costs bandwidth and host capacity even when
  the prefix is never reused.
- A lower-tier hit helps only if `T_restore < T_recompute`. **Prefetching** moves the L3 read
  into the time the request spends queued (`best_effort` / `wait_complete` / `timeout`).
  **Layer-wise loading** lets layer *i* compute while layer *i+1*'s K/V is still in flight.

## How this extends mini-sglang

mini-sglang has exactly L1. `python/minisgl/kvcache/radix_cache.py::RadixPrefixCache.evict`
removes unreferenced LRU leaves and returns their slots to `CacheManager.free_slots`. The
K/V is then gone and the next match recomputes it. HiCache changes eviction from "drop" to
"demote":

| step | mini-sglang today | with an L2 tier |
|---|---|---|
| evict | node removed, slots freed | node keeps a `host_value` (host slot indices); only the device slots are freed |
| match | stops at the first missing node | continues through host-only nodes; returns `device_len` and `host_hit_len` separately |
| admit (`PrefillAdder`) | budget counts `extend_len` only | must also reserve device slots for the restored part (SGLang: `req.host_hit_length`, `needs_host_load_back()` in `PrefillAdder.add_one_req`) |
| forward | attention reads the pool | layer *i* waits until its K/V has arrived (`LayerDoneCounter.wait_until` in `srt/managers/cache_controller.py`) |

The cost model is easy to measure on CPU (lab Part A). On Qwen3-0.6B fp32, restoring a
prefix (copying all 28 layers of K/V into pool slots with `MHAKVCache.store_kv`) is
**~150-180× cheaper** than prefilling it:

| prefix tokens | KV | recompute ms | restore ms | ratio |
|---:|---:|---:|---:|---:|
| 256 | 56 MiB | 508 | 3.5 | 147× |
| 1024 | 224 MiB | 2662 | 15.0 | 177× |
| 2048 | 448 MiB | 5993 | 37.0 | 162× |

On a GPU the ratio is much less lopsided, because compute is ~1000× faster while PCIe is not.
Part B models Qwen3-8B on an H100-class GPU (989 TFLOP/s × 50% MFU, 50 GB/s H2D,
5 GB/s + 2 ms storage; illustrative, not measured):

| 8K-token prefix (1.12 GiB KV) | ms | vs recompute |
|---|---:|---:|
| recompute | 311 | 1.00× |
| L2 hit (H2D) | 24 | 0.08× |
| L3 hit (storage→host→GPU) | 268 | 0.86× |
| L3 hit with the storage read hidden by prefetch | 24 | 0.08× |

The break-even point for an un-prefetched L3 hit is about **2K tokens** under these
assumptions. Below that, recomputing is faster. Recompute grows superlinearly (attention
is O(L²)), but transfer is linear, so long prefixes always favour restoring.

## SGLang pointers

- `srt/managers/cache_controller.py::HiCacheController` — `write` (backup), `load` (restore),
  `prefetch` (L3→L2), `LayerDoneCounter` (layer-wise sync).
- `srt/mem_cache/unified_radix_cache.py::UnifiedRadixCache.init_hicache`, `load_back`,
  `write_backup_storage`, `prefetch_from_storage`, `check_hicache_events`.
- Host pools: `srt/mem_cache/pool_host/mha.py::MHATokenToKVPoolHost`. Storage API:
  `srt/mem_cache/hicache_storage.py::HiCacheStorage`, backends under `srt/mem_cache/storage/`.
- Flags: `--enable-hierarchical-cache`, `--hicache-ratio` / `--hicache-size`,
  `--hicache-write-policy {write_through, write_back, write_through_selective}`,
  `--hicache-storage-prefetch-policy {best_effort, wait_complete, timeout}`.
- Read `srt/mem_cache/README.md` first. It explains the allocator → pool → pool_host →
  storage layering.

## Exercises — worked answers

**1. Why HiCache when paged KV and prefix caching already exist? What does each solve?**
Paged KV solves *allocation*: fixed-size pages, no fragmentation, easy reclaim. The prefix
cache solves *redundant computation*: identical prefixes map to the same KV slots and are
not recomputed while they are resident. Both live inside finite GPU memory, so under
pressure the prefix cache must evict, and an evicted prefix is recomputed on its next
use. HiCache solves *capacity*: it keeps evicted states in host memory or storage and
brings them back, trading transfer time for compute. It does not raise the maximum
context length or speed up decode.

**2. What happens before execution on an L1, L2 or L3 hit? Is L3 always shared?**
L1: the K/V is already in device slots. Lock the node and point the request's page-table
row at those slots (exactly what our `scheduler/prefill.py` (`_try_allocate_one`) does with
`handle.get_matched_indices()`). L2: allocate device slots, copy host→device (possibly
layer by layer), then compute the uncached suffix. L3: read from storage into host memory
(ideally prefetched while the request is queued), then host→device as in L2. In every
case the reusable part must be a contiguous prefix from token 0. L3 is **not** necessarily
shared. It is whatever backend is configured: a local file or SSD store is private to the
instance, while Mooncake, 3FS or NIXL-backed stores can be shared across instances.

**3. Recompute 150 ms, Host restore 30 ms, storage read +200 ms: do both lower tiers help?**
L2: 30 ms < 150 ms, so it saves 120 ms. L3: 200 + 30 = 230 ms > 150 ms, so it is **slower**
than recomputing (by 80 ms). The L3 hit helps only if prefetch hides at least 80 ms of the
storage read while the request waits in the queue (lab Part C: hiding 100 ms → 130 ms, which
beats recompute), or if the prefix is longer, since recompute grows faster than transfer.

**4. How does prefetching help, and what do the three policies trade off?**
A queued request has known tokens, so its L3 pages can be read into host memory while the
GPU serves other requests. This overlaps storage latency with useful work, and if the
read finishes before the request is scheduled, only the H2D copy remains. If it has not
finished: `best_effort` starts now with whatever prefix is ready and recomputes the rest
(lowest wait, possibly less reuse). `wait_complete` delays the request until the read
finishes (maximum reuse, but it can stall behind slow storage). `timeout` waits up to a
bound, then proceeds like best_effort (bounded wait; the bound should be tuned to the cost of
recomputing that prefix).

**5. Why can computation start once layer 0's KV is ready? What dependencies remain?**
A transformer runs layer by layer, and layer *i*'s attention only needs layer *i*'s K/V
for the history. So once layer 0's history is on the GPU, the new tokens' layer-0 compute can
run while layers 1..L-1 are still being copied. Constraints: (a) before layer *i*
attention runs, *all* of layer *i*'s restored K/V must have arrived (SGLang blocks in
`LayerDoneCounter.wait_until(i)`); (b) the destination slots must be allocated and the
page table written before both the copy and the attention read; (c) the copy stream must be
ordered with the compute stream through events, and the source host buffers must not be
freed or overwritten mid-copy; (d) the step time becomes max(transfer, compute) per layer,
so overlap cannot hide a transfer longer than the compute.

**6. Hit rate up, but TTFT tail latency worse: what to investigate?**
- Are the hits L2/L3 restores that are slower than recomputing (short prefixes, slow storage)?
  Break hits down per tier and compare restore time with prefill time per request.
- Prefetch policy: `wait_complete` or a long `timeout` makes requests wait on slow storage
  reads (tail heavy).
- Bandwidth contention: backup writes (write-through) competing with restores, or with
  other PCIe traffic (PD transfers, other GPUs on the same host bridge).
- Host-side overhead: pinned-memory allocation, host-pool eviction, page-hash lookups and
  storage queue depth.
- Device capacity: reserving slots for restored prefixes shrinks the batch, so queueing
  increases (check queue time vs prefill time in the TTFT breakdown).
- Layer-wise overlap disabled or ineffective (the trace shows compute waiting on loads).
Validate with per-tier hit tokens, restore latency, host-pool usage and link utilisation,
and compare against HiCache-off under the identical workload.
