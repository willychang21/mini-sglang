# mini-sglang study guide (Part 0, Part I, Part II)

This guide is the learning layer on top of the code. The tutorial
([zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang), English
edition under `course-material/eng/`) explains *why*; this repository is a
working, CPU-runnable re-implementation of the official
[mini-sglang](https://github.com/sgl-project/mini-sglang) that the tutorial walks
through; the **labs** make each mechanism observable; the chapter notes tie the
three together and answer every end-of-chapter exercise.

Part III/IV are covered elsewhere.

## Setup (CPU only)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[test]" pyzmq
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-0.6B', local_dir='/tmp/models/Qwen3-0.6B')"
git clone --depth 1 https://github.com/datawhalechina/zero-to-sglang /tmp/zero-to-sglang
python -m pytest -q                      # engine tests + lab smoke tests
```

Labs default to a tiny random 3-layer Qwen3 (created on first use, cached in
`$TMPDIR/minisgl-labs`), so they run in seconds. Add `--model real` to run them on
Qwen3-0.6B. See [`labs/README.md`](../../labs/README.md).

## How to study one chapter

For every Part II chapter, use the same five-step loop:

1. **Read the concept chapter** (`partN/ChapterX_*.md`). Before moving on, write
   down the *limitation* the chapter starts from.
2. **Read the code-walkthrough chapter** (`ChapterX_*_code.md`) with this repo
   open next to it. The file layout matches the official mini-sglang, so every
   path in the tutorial has a counterpart here. The chapter note's
   "Where in mini-sglang" table gives `file:function` pointers, and its
   "This repo vs. the tutorial's reference code" section lists the places where
   this repo deliberately differs.
3. **Run the lab** and predict its output before reading it. The lab prints the
   internal state (lengths, slots, trees, timelines) that the chapter only
   draws.
4. **Answer the exercises yourself**, then compare with the worked answers in the
   chapter note. Answers that can be checked by running code cite lab output.
5. **Run the validation**: the tests that pin the mechanism's correctness and
   the benchmark that measures its benefit (table below).

## Learning path

| # | Read (tutorial) | Then read (this repo) | Run | Notes |
|---|---|---|---|---|
| 0.1 | `part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md` | — | — | [part0/coding-ethics-and-open-source.md](part0/coding-ethics-and-open-source.md) |
| 0.2 | `part0/Part0-Deploy-Your-First-SGLang-Server.md` | `README.md` quick start | `python -m minisgl --model /tmp/models/Qwen3-0.6B --device cpu` + `curl` (GPU steps: Colab, not run here) | [part0/deploy-first-sglang-server.md](part0/deploy-first-sglang-server.md) |
| I.1 | `part1/Chapter1_Introduction_to_LLM.md` | `models/qwen3.py`, `layers/` (skim) | — | [part1/ch01-introduction-to-llm.md](part1/ch01-introduction-to-llm.md) |
| I.2 | `part1/Chapter2_Introduction to Inference.md` | — | `labs/part1_ch02_roofline.py` | [part1/ch02-introduction-to-inference.md](part1/ch02-introduction-to-inference.md) |
| I.3 | `part1/Chapter3_Introduction_to_GPU.md` | — | `labs/part1_ch02_roofline.py` (roofline part) | [part1/ch03-introduction-to-gpu.md](part1/ch03-introduction-to-gpu.md) |
| I.4 | `part1/Chapter4_KV Cache The Core Data Structure of Inference.md` | `kvcache/mha_pool.py` | `labs/part1_ch04_kv_memory.py` | [part1/ch04-kv-cache-core-data-structure.md](part1/ch04-kv-cache-core-data-structure.md) |
| I.5 | `part1/Chapter5_Introduction_to_Benchmark.md` | `llm/llm.py` (`GenerateOutput`), `benchmarks/bench_serving.py` | `labs/part1_ch05_metrics.py` | [part1/ch05-introduction-to-benchmark.md](part1/ch05-introduction-to-benchmark.md) |
| II.1 | `Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md` | `core.py` | — | [part2/ch01-what-an-inference-engine-looks-like.md](part2/ch01-what-an-inference-engine-looks-like.md) |
| II.2 | `Chapter2_Inside-SGLang.md` → `_code.md` | `message/`, `tokenizer/`, `server/launch.py`, `scheduler/scheduler.py` | `labs/part2_ch02_request_path.py` | [part2/ch02-inside-sglang.md](part2/ch02-inside-sglang.md) |
| II.3 | `Chapter3_Forward-Pass-and-Generation.md` → `_code.md` | `models/`, `layers/`, `engine/sample.py`, `llm/naive.py` | `labs/part2_ch03_forward_shapes.py` | [part2/ch03-forward-pass-and-generation.md](part2/ch03-forward-pass-and-generation.md) |
| II.4 | `Chapter4_KV-Cache-Optimization.md` → `_code.md` | `kvcache/mha_pool.py`, `attention/torch_backend.py`, `scheduler/cache.py:allocate_paged` | `labs/part2_ch04_kv_cache.py` | [part2/ch04-kv-cache-optimization.md](part2/ch04-kv-cache-optimization.md) |
| II.5 | `Chapter5_HTTP-and-Concurrent-Requests.md` → `_code.md` | `server/api_server.py`, `tokenizer/server.py` | `labs/part2_ch05_frontend.py` | [part2/ch05-http-and-concurrent-requests.md](part2/ch05-http-and-concurrent-requests.md) |
| II.6 | `Chapter6_Continuous-Batching-and-the-Scheduler.md` → `_code.md` | `scheduler/` (all) | `labs/part2_ch06_batching_timeline.py` | [part2/ch06-continuous-batching-and-scheduler.md](part2/ch06-continuous-batching-and-scheduler.md) |
| II.7 | `Chapter7_Paged-KV-Cache-and-Memory-Management.md` | `scheduler/cache.py` | `labs/part2_ch07_paged_memory.py` | [part2/ch07-paged-kv-cache.md](part2/ch07-paged-kv-cache.md) |
| II.8 | `Chapter8_RadixAttention-and-Prefix-Caching.md` → `_code.md` | `kvcache/radix_cache.py`, `scheduler/cache.py:cache_req` | `labs/part2_ch08_radix_tree.py` | [part2/ch08-radixattention-prefix-caching.md](part2/ch08-radixattention-prefix-caching.md) |
| II.9 | `Chapter9_Multi-process-and-Tensor-Parallelism.md` → `_code.md` | `layers/linear.py`, `layers/embedding.py`, `models/weight.py`, `distributed/`, `server/launch.py` | `labs/part2_ch09_tp_shapes.py` | [part2/ch09-multiprocess-tensor-parallelism.md](part2/ch09-multiprocess-tensor-parallelism.md) |
| II.10 | `Chapter10_Speculative-Decoding.md` → `_code.md` | `speculative/` | `labs/part2_ch10_speculative.py` | [part2/ch10-speculative-decoding.md](part2/ch10-speculative-decoding.md) |

Suggested pacing: Part 0 + Part I in about a week (they are conceptual; do the
calculator labs by hand first, then check with the scripts). Part II one chapter
per 1–2 sessions; Chapters 6–8 deserve the most time because the scheduler,
the page allocator and the radix tree interact (a request's KV slots are owned
first by the request, then by the tree).

### Why Part I comes first

Part I gives the vocabulary and the two numbers you will keep using in Part II:

* **KV bytes per token** = `2 × layers × kv_heads × head_dim × dtype_bytes`
  (Part I Ch4). It sizes `MHAKVCache`, bounds concurrency (Ch6 admission), and
  explains why paging (Ch7), prefix sharing (Ch8) and TP head-splitting (Ch9)
  matter.
* **Arithmetic intensity** (FLOPs per byte moved, Part I Ch2/3). Prefill
  multiplies `[N, C]` activations by weights (high intensity, compute-bound);
  decode multiplies `[B, C]` with small `B` (low intensity, memory-bound). This
  is why batching decode requests (Ch6) is so effective, and why speculative
  decoding (Ch10) can verify `k` tokens for roughly the cost of one.

## Limitation → new mechanism → validation metric

Part II Chapter 1 §4 frames every chapter with the same question: *what is the
bottleneck, what mechanism removes it, and which metric proves it?* The table
below maps each row to the code that implements it and to where it is validated
in this repo. README.md has the measured benchmark numbers (Qwen3-0.6B, fp32,
4-core CPU).

| Ch | Limitation of the previous version | New mechanism | Code | Metric (tutorial) | Validated here by |
|---|---|---|---|---|---|
| 2 | Request path through the source is unclear | Request path with uid-tagged messages | `message/`, `tokenizer/server.py`, `scheduler/io.py`, `server/api_server.py` | draw the path, find files | lab `part2_ch02_request_path.py`, `tests/test_detokenizer.py` |
| 3 | No working generation | Forward pass, generation loop, sampling | `models/qwen3.py`, `models/utils.py`, `layers/`, `engine/sample.py`, `llm/naive.py` | output correctness | `tests/test_model.py` (token-for-token vs HuggingFace), `tests/test_real_model.py`, lab `part2_ch03_forward_shapes.py` |
| 4 | Decode recomputes previous tokens | KV cache (`MHAKVCache`, `page_table`, `out_loc`, store-then-attend) | `kvcache/mha_pool.py`, `attention/torch_backend.py`, `scheduler/scheduler.py:_prepare_batch` | computation, TPOT | `benchmarks/bench_kv_cache.py` (TPOT flat vs growing), lab `part2_ch04_kv_cache.py` (computed positions, identical tokens) |
| 5 | Only callable from Python | HTTP, concurrency (`ack_map`/`event_map`), SSE streaming, cancellation | `server/api_server.py`, `tokenizer/server.py`, `server/launch.py` | end-to-end latency | `tests/test_server.py`, `benchmarks/bench_serving.py` (TTFT/TPOT vs concurrency), lab `part2_ch05_frontend.py` |
| 6 | One request at a time / static batches | Continuous batching, `PrefillAdder`, chunked prefill, `DecodeManager` | `scheduler/scheduler.py`, `scheduler/prefill.py`, `scheduler/decode.py`, `scheduler/table.py` | throughput, TTFT | `benchmarks/bench_batching.py` (tok/s, mean TTFT), `tests/test_scheduler.py`, lab `part2_ch06_batching_timeline.py` |
| 7 | Contiguous allocation wastes memory, fragments | Paged KV, `free_slots` page allocator | `scheduler/cache.py:CacheManager.allocate_paged` | memory utilization, concurrency | `benchmarks/bench_batching.py` (concurrent requests under a fixed KV budget), `tests/test_scheduler.py` (`page_size` 1 and 4), lab `part2_ch07_paged_memory.py` |
| 8 | Shared prefixes recomputed | Radix tree prefix cache (split/lock/insert/LRU evict) | `kvcache/radix_cache.py`, `scheduler/cache.py:cache_req` | cache hit rate, TTFT | `benchmarks/bench_prefix_cache.py` (computed prefill tokens, hit rate, TTFT), `tests/test_radix_cache.py`, lab `part2_ch08_radix_tree.py` |
| 9 | One device limits capacity and throughput | One scheduler process per rank, column/row/vocab-parallel layers | `layers/linear.py`, `layers/embedding.py`, `models/weight.py`, `distributed/`, `scheduler/io.py` | multi-GPU scaling | `tests/test_tp.py` (TP=2 on Gloo matches TP=1), lab `part2_ch09_tp_shapes.py` (shapes, memory per rank); scaling itself needs GPUs |
| 10 | Every token needs a full target pass | Draft → verify in one pass → commit → rewind | `speculative/` | acceptance rate, speed | `benchmarks/bench_speculative.py`, `tests/test_speculative.py` (lossless), lab `part2_ch10_speculative.py` |

Two habits this table should build:

* **Correctness first, then speed.** Every optimization in Part II is checked
  against HuggingFace greedy output token for token (`tests/`) *before* its
  benefit is measured (`benchmarks/`). An optimization that changes the output
  is a bug, not a speed-up (speculative sampling preserves the *distribution*,
  so its test is greedy-lossless plus a statistical check).
* **Measure the metric the mechanism targets.** KV cache → TPOT, not TTFT;
  prefix cache → computed prefill tokens and TTFT, not TPOT; continuous
  batching → throughput and TTFT, at the price of TPOT under load
  (`bench_serving.py` shows TPOT rising with concurrency).

## Files in this guide

* `part0/`, `part1/`, `part2/`: one note per chapter: key ideas, where in
  mini-sglang, misconceptions, lab, worked answers to every exercise.
* [`ISSUES.md`](ISSUES.md): mismatches between the tutorial and this repo, and
  bugs found while answering the exercises (with reproductions). Engine code is
  not changed by the study guide.
* [`../../labs/`](../../labs): the runnable labs and their index.

## Going further toward SGLang

The official mini-sglang is a deliberately small subset of SGLang. Once a Part II
chapter is clear here, read the matching SGLang component that the tutorial's
reference lists point to (scheduler, `mem_cache/radix_cache.py`,
`schedule_policy.py`, speculative workers). The concepts carry over directly;
what SGLang adds is mostly overlap scheduling, CUDA graphs, custom kernels, more
cache policies, and many models. Part IV Chapter 3 of the tutorial covers the
SGLang PR workflow.
