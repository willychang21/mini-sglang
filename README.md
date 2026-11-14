# mini-sglang

A small but complete LLM inference engine, written from scratch by following
Part II of the [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
tutorial ("Build Your Own Mini SGL"). The layout mirrors the official
[mini-sglang](https://github.com/sgl-project/mini-sglang) `python/minisgl/` tree
that the tutorial walks through. The default path is pure PyTorch and runs on **CPU**.
On CUDA there is an optional Triton paged-attention backend and CUDA Graph decode.

It serves Qwen3 models (tested with `Qwen/Qwen3-0.6B`) and reproduces
HuggingFace greedy output **token for token** in every configuration tested: paged KV,
radix cache, chunked prefill, continuous batching, optimistic admission with retraction,
the overlap loop, TP=2, and batched speculative decoding.

## Learning path: from zero-to-sglang to contributing to SGLang

The repo is also a complete study companion for the course. Start at
**[docs/README.md](docs/README.md)** and follow the steps in order:

1. **[Study guide, Part 0–II](docs/study-guide/README.md):** notes for every chapter, worked
   answers to every exercise, and runnable [labs](labs/README.md) that make each mechanism
   observable (tensor shapes, `cached_len`/`out_loc` traces, radix tree dumps, batching
   timelines, and more).
2. **Beyond the Part II baseline:**
   - [scheduler-advanced.md](docs/scheduler-advanced.md): abort routing, preemption,
     backpressure, and the overlap loop
   - [attention-backends.md](docs/attention-backends.md): paged attention, flash-decoding,
     and CUDA Graph
   - [speculative-in-scheduler.md](docs/speculative-in-scheduler.md): batched
     draft/verify/rollback inside continuous batching
3. **[SGLang source map](docs/sglang-source-map.md):** every module here mapped to the SGLang
   file and class that does the same job, at a pinned commit.
4. **Part III/IV:** notes and labs for attention backends, quantization, hierarchical caching,
   scaling out, PD disaggregation, deployment, and profiling (including a
   [real trace walkthrough](docs/part4/profiling-walkthrough.md)).
5. **[Contributing to SGLang](docs/contributing-to-sglang.md):** dev setup, tests, lint, CI,
   and five real open issues with plans. Then the [self-check quiz](docs/self-check.md).

## Quick start

```bash
pip install -e ".[test]"
```

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

Offline Python API:

```python
from minisgl import LLM, SamplingParams

llm = LLM("models/Qwen3-0.6B", device="cpu")
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16))
print(out[0].text)
```

OpenAI-style server (`/v1/chat/completions`, `/v1/completions`, SSE streaming):

```bash
python -m minisgl --model models/Qwen3-0.6B --device cpu --tp-size 1 --num-tokenizers 2
```

```bash
curl -N http://localhost:1919/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is KV Cache?"}],"max_tokens":64,"stream":true,
       "temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Tests (tiny random Qwen3 checkpoints compared against HuggingFace; the real-model
test runs when `MINISGL_TEST_MODEL` or `/tmp/models/Qwen3-0.6B` exists):

```bash
python -m pytest -q
```

## Tutorial chapter → code

| Chapter | What it adds | Where |
|---|---|---|
| 1. What an inference engine looks like | `SamplingParams`, `Req`, `Batch`, `Context` | `core.py` |
| 2. The path of a request | API server → tokenizer → scheduler → engine → detokenizer, uid-tagged messages | `message/`, `tokenizer/`, `server/` |
| 3. Forward pass & generation | Qwen3: packed `[N, C]` layout, fused residual RMSNorm, QK-norm + RoPE, gated MLP, last-position LM head; sampling; a no-KV-cache baseline | `models/`, `layers/`, `engine/sample.py`, `llm/naive.py` |
| 4. KV cache | `MHAKVCache` `[2, layers, pages, page_size, kv_heads, head_dim]`, `page_table`, `out_loc`, store-then-attend paged attention | `kvcache/mha_pool.py`, `attention/torch_backend.py`, `engine/engine.py` |
| 5. HTTP & concurrent requests | FastAPI, `ack_map` / `event_map` per uid, SSE streaming, cooperative cancellation | `server/api_server.py`, `tokenizer/server.py` |
| 6. Continuous batching & scheduler | `normal_loop`, prefill-first policy, `PrefillAdder` (slots / token budget / KV reservation), chunked prefill, `DecodeManager` | `scheduler/` |
| 7. Paged KV & memory management | `free_slots` page allocator, `allocate_paged`, `page_size > 1` | `scheduler/cache.py` |
| 8. RadixAttention & prefix caching | radix tree with split / lock / insert / LRU leaf eviction, `cache_req` | `kvcache/radix_cache.py`, `scheduler/cache.py` |
| 9. Multi-process & tensor parallelism | one scheduler process per rank, column/row/vocab-parallel layers, shard-then-merge weight loading, KV-head replication, rank-0 broadcast over Gloo | `layers/linear.py`, `layers/embedding.py`, `models/weight.py`, `distributed/`, `server/launch.py` |
| 10. Speculative decoding | draft → one-pass verification → commit → rewind; greedy and rejection-sampling acceptance; model and n-gram drafters; batched inside the scheduler / server (`--speculative-algorithm`, [docs](docs/speculative-in-scheduler.md)) | `speculative/` |

## Benchmarks (Qwen3-0.6B, fp32, 4-core CPU)

Each chapter's "primary validation" metric, measured with the scripts in
[`benchmarks/`](benchmarks). Absolute numbers are CPU numbers; the relative
effects are the point.

**Ch4: KV cache** (`bench_kv_cache.py`, 128-token prompt; outputs identical)

| output tokens | no cache: TPOT ms | KV cache: TPOT ms | speedup |
|---:|---:|---:|---:|
| 16 | 325.6 | 80.6 | 4.0x |
| 64 | 356.1 | 68.3 | 5.2x |
| 128 | 414.8 | 69.1 | 6.0x |

Without a cache TPOT grows with length; with it, TPOT stays flat.

**Ch6/7: continuous batching and paged KV** (`bench_batching.py`, 16 requests; outputs identical)

| mode | wall s | output tok/s | mean TTFT s |
|---|---:|---:|---:|
| one request at a time | 46.5 | 14.4 | 22.93 |
| continuous batching | 13.4 | 49.9 | 2.01 |

With a 2048-token KV budget and `max_seq_len=1024`, contiguous pre-allocation
fits 2 concurrent requests; paged allocation ran 16.

**Ch8: radix prefix cache** (`bench_prefix_cache.py`, 12 requests sharing a 512-token system prompt)

| cache | computed prefill tokens | hit rate | mean TTFT s |
|---|---:|---:|---:|
| naive | 6376 | 0% | 1.203 |
| radix | 744 | 88.3% | 0.304 |

**Ch10: speculative decoding** (`bench_speculative.py`, k=4, 64 tokens; all rows lossless)

| task | draft | acceptance | tokens / target forward | speedup |
|---|---|---:|---:|---:|
| open-ended | first 14 layers of the target | 5% | 1.19 | 0.29x |
| open-ended | n-gram prompt lookup | 15% | 1.37 | 0.90x |
| open-ended | target itself (upper bound) | 100% | 4.92 | 0.80x |
| copy-heavy | first 14 layers of the target | 3% | 1.11 | 0.30x |
| copy-heavy | n-gram prompt lookup | 61% | 2.42 | **1.60x** |
| copy-heavy | target itself (upper bound) | 100% | 4.92 | 0.87x |

As Chapter 10 warns, fewer target calls are not a speedup by themselves. A
truncated, untrained draft is rarely accepted. A draft as expensive as the target
can't win. A nearly free draft wins when its guesses land.

**Ch5: HTTP serving** (`bench_serving.py`, 16 streaming requests × 32 tokens)

| concurrency | mean TTFT s | mean TPOT ms | output tok/s |
|---:|---:|---:|---:|
| 1 | 0.21 | 63 | 14.7 |
| 4 | 0.38 | 114 | 32.6 |
| 16 | 1.34 | 207 | 65.9 |

This is the latency/throughput trade-off from Part I Chapter 5: higher
concurrency raises per-request latency and raises total throughput.

## Differences from the reference implementation

The tutorial points out several gaps in the reference code. This version closes them:

* **Frontend cleanup** (Ch5 §5.1): per-uid state is removed in a `finally`
  block. Callers use `aclosing`, so early exits clean up too.
* **Non-streaming disconnects** (Ch5 §5.3) are detected and aborted.
* **Detokenizer state on abort** (Ch5 §5.3): the scheduler sends a final
  `abort` message, so the detokenizer drops the uid.
* **Rejected requests** (too long, empty, tokenization failure) get an error
  reply instead of a hung connection.
* **`cache_req` duplicates** (Ch8 exercise 4): when another request inserted
  the same prefix first, the duplicate pages are freed *and* the still-running
  request's page table is pointed at the cached copy, so it never reads freed
  slots.
* **Worker processes** exit when the API server dies, even on `SIGKILL`.

The scheduler also fixes gaps found while writing the study guide
([docs/study-guide/ISSUES.md](docs/study-guide/ISSUES.md)):

* aborts can no longer overtake their request when there are several tokenizers;
* out-of-vocabulary token ids get an error reply instead of crashing the scheduler;
* a request that cannot be admitted no longer splits radix tree nodes.

## Features beyond the Part II baseline

| Feature | Flag | Doc |
|---|---|---|
| Optimistic admission with retraction (preemption) | `--admission optimistic` | [scheduler-advanced.md](docs/scheduler-advanced.md) |
| Overlap scheduling (CPU work of step N+1 overlaps step N) | `--enable-overlap` | [scheduler-advanced.md](docs/scheduler-advanced.md) |
| Per-request reply buffer limits (backpressure) | `--max-buffered-bytes` | [scheduler-advanced.md](docs/scheduler-advanced.md) |
| Triton paged attention, decode split-KV + extend | `--attention-backend triton` | [attention-backends.md](docs/attention-backends.md) |
| CUDA Graph decode with a dummy page | on by default on CUDA, `--disable-cuda-graph` | [attention-backends.md](docs/attention-backends.md) |
| Batched speculative decoding (n-gram or draft model) | `--speculative-algorithm ngram` | [speculative-in-scheduler.md](docs/speculative-in-scheduler.md) |

**First GPU run (Tesla T4, fp16):** [docs/gpu-results-t4.md](docs/gpu-results-t4.md)
- CUDA Graph cuts decode TPOT from 56 to 12.7 ms at batch size 1, because eager decode of a 0.6B
  model is launch-overhead bound.
- Continuous batching plus CUDA Graph reaches 530 output tok/s, vs 22 tok/s one request at a time.
- N-gram speculative decoding is up to 3.3× faster on copy-heavy prompts.
- The Triton kernels run correctly but are slower than PyTorch SDPA except for long single-sequence
  decode.
- The GPU test suite and multi-GPU NCCL TP have not been verified yet. [notebooks/gpu_benchmarks.ipynb](notebooks/gpu_benchmarks.ipynb) runs the tests and
benchmarks on a Colab GPU:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/willychang21/mini-sglang/blob/main/notebooks/gpu_benchmarks.ipynb)

Remaining simplifications:

* The default `torch` attention backend is one padded `scaled_dot_product_attention` call
  over K/V gathered through the page table.
* The page table stores `int64` slots.
* Request broadcast to TP ranks uses `broadcast_object_list` over Gloo instead of
  a Gloo count plus ZMQ PUB/SUB.
