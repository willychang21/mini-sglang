# Labs

Small runnable scripts that make each mechanism of the tutorial observable. They
complement the chapter notes in [`docs/study-guide/`](../docs/study-guide/README.md).

* Every lab runs on **CPU** in seconds with the default **tiny** model: a random
  3-layer Qwen3 (hidden 64, 4 Q heads, 2 KV heads, vocab 512), created on first
  use and cached in `$TMPDIR/minisgl-labs` (override with `MINISGL_LABS_CACHE`).
* `--model real` runs the same lab on Qwen3-0.6B (`/tmp/models/Qwen3-0.6B`, or
  `MINISGL_TEST_MODEL`); `--model <path>` takes any Qwen3 checkpoint.
* Labs **assert** what they print, so a lab that finishes has checked its own
  claims. `tests/test_labs.py` runs them all as smoke tests.
* Labs only read or trace engine objects (some wrap methods on an instance);
  they never change `python/minisgl/`.

Run from the repository root, e.g. `python labs/part2_ch04_kv_cache.py`.

| Lab | Chapter | What to look for |
|---|---|---|
| `part1_ch02_roofline.py [--skip-measure]` | Part I Ch2, Ch3 | Ridge points of the cited GPUs; AI of prefill ≈ N vs decode ≈ 1 FLOP/byte (×dtype); the decode batch size where linears turn compute-bound; long-context attention staying memory-bound at any batch; measured CPU GFLOP/s for an N=1 GEMV vs N=512 GEMM. |
| `part1_ch04_kv_memory.py [--budget-gib 16]` | Part I Ch4 | `2 × layers × kv_heads × head_dim × bytes` per token; the worked Ch4 exercise 3 (Qwen2.5-1.5B, T=200, l=100); MHA vs GQA vs MQA; requests per budget and contiguous-reservation waste; the formula equals `MHAKVCache.num_bytes`. |
| `part1_ch05_metrics.py [--slots 12]` | Part I Ch5 | TTFT / TPOT / ITL / E2E / throughput from real timestamps; with fewer slots than requests, P50 TTFT stays low while P99 explodes and the mean describes nobody; ITL shows prefill-induced stalls that TPOT averages away. |
| `part2_ch02_request_path.py` | Part II Ch2 | One uid through `TokenizeMsg → UserMsg → normal_loop → DetokenizeMsg → UserReply`; where length/EOS finish; `surr_offset/read_offset` holding back a split emoji; abort returns every slot and page. Needs the real tokenizer. |
| `part2_ch03_forward_shapes.py` | Part II Ch3 | Shapes of a packed 2-request prefill (`[N, C]` → `[B, V]`); last-token indices; the residual add inside `RMSNormFused`; causal independence from future tokens (and its loss without the mask); Hkv 2 vs 1; greedy vs top-k/top-p sampling. |
| `part2_ch04_kv_cache.py [--page-size 4]` | Part II Ch4 | Pool layout; free-page → slot expansion; per-step `cached_len/device_len/extend_len`, `positions`, `out_loc`, free pages; why allocation precedes `out_loc`; 216 vs 21 computed positions with identical tokens; the mask rule `j ≤ cached_len + t` on a 2-request batch. |
| `part2_ch05_frontend.py` | Part II Ch5 | The real `FrontendManager` with in-memory queues: interleaved A/B replies routed by uid (3 `set()` → 2 wake-ups, nothing lost); SSE chunks; `finally`+`aclosing` cleanup vs the reference's leak; cancellation and disconnects; a blocking call stalling the event loop. |
| `part2_ch06_batching_timeline.py` | Part II Ch6 | Step-by-step timelines reproducing the tutorial's A/B/C table and the A..E example row by row; static vs continuous batching; chunked prefill (`ChunkedReq` commits nothing, goes back to the front); decode while the queue waits; the KV admission inequality of exercise 5. |
| `part2_ch07_paged_memory.py` | Part II Ch7 | Contiguous reservation fits 2 requests where paging runs 16; internal fragmentation for page_size 1/4/16; non-contiguous block-table rows; blockwise online softmax = full softmax; corrupting the block table. |
| `part2_ch08_radix_tree.py` | Part II Ch8 | A printed radix tree: splits, ref counts along a locked path, LRU leaf eviction cascading to parents, page alignment; the tree after real requests with a shared system prompt; `match_req` stopping at `len-1`; the double capacity check around `lock`; `cache_req` freeing duplicate pages. |
| `part2_ch09_tp_shapes.py [--run-tp]` | Part II Ch9 | Per-rank shapes of every TP layer for tp 1/2/4 (Qwen3-0.6B on the meta device); KV-head replication; memory per rank vs tp; shard-then-merge is right, merge-then-shard is wrong; `--run-tp` adds a real TP=2 Gloo run vs TP=1. |
| `part2_ch10_speculative.py` | Part II Ch10 | Round-by-round draft/verify/commit/rewind traces for a truncated draft, the n-gram drafter and self-drafting, all equal to target-only greedy; row alignment for L=5, k=3; the p=[0.3,0.7], q=[0.6,0.4] exercise (analytic + Monte Carlo); EOS/bonus edge cases; the cost model. |

Engine benchmarks that measure the *benefit* of each mechanism on the real model
live in [`benchmarks/`](../benchmarks); their results are in the top-level
README.
