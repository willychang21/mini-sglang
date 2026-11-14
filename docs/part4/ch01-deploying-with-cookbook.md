# Part IV · Ch1 — Deploying SGLang with the Cookbook

Course text: `part4/Chapter1_Deploying-SGLang-with-the-Cookbook.md`.

> **Environment note.** This session has no GPU, so SGLang itself was not launched. Below:
> (1) the launch flags mapped to concepts and to mini-sglang's own flags, (2) the
> exercise protocol as a fill-in record, and (3) the same protocol actually executed on
> mini-sglang (CPU) so the method is demonstrated end to end. Run the SGLang half on a GPU
> machine using the same record.

## Key ideas

- A Cookbook recipe is a tested starting point for one **(model variant, GPU type/count,
  precision, SGLang version)**. Check all four before copying a command.
- Every flag is a decision from earlier chapters: `--tp-size` (Part II Ch9 / III Ch4),
  `--dtype` and the checkpoint's quantization (III Ch2), radix cache on by default
  (II Ch8), and `--mem-fraction-static` (KV pool size, III Ch2 §3.3).
- Validate in three layers: a sensible response (correctness smoke test), logs and memory
  (the intended model and `max_total_num_tokens`), then a small fixed benchmark (TTFT,
  TPOT, throughput). Change **one** setting at a time.

## Flag map: SGLang ↔ concept ↔ mini-sglang

| SGLang flag | concept | mini-sglang (`python -m minisgl`, `server/args.py`) |
|---|---|---|
| `--model-path` | weights / config | `--model` / `--model-path` |
| `--tp-size` | tensor parallelism | `--tp-size` (Gloo on CPU, NCCL on GPU) |
| `--dtype bfloat16` | compute and KV dtype | `--dtype` (default fp32 on CPU, bf16 on CUDA) |
| `--quantization`, `--kv-cache-dtype` | III Ch2 | — (see `labs/part3_ch02_quant_error.py`) |
| `--mem-fraction-static` → `max_total_num_tokens` | KV pool capacity | `--max-total-tokens` (set directly) |
| `--page-size` | paged KV | `--page-size` |
| `--chunked-prefill-size` | prefill token budget | `--prefill-budget` |
| `--max-running-requests` | request slots | `--max-running-req` |
| `--disable-radix-cache` | prefix cache | `--cache-type naive` |
| `--attention-backend` | III Ch1 | — (single torch backend) |
| `--host/--port` | | `--host/--port` |

`--disable-radix-cache` (or our `--cache-type naive`) only stops *cross-request* reuse. Each
request still has its own KV cache during decode.

## Exercise 1 — record (SGLang half to fill on a GPU)

```
recipe link:        https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3  (H100 / 8B / BF16)
GPU / count:        ______
SGLang version:     python -c "import sglang; print(sglang.__version__)"  -> ______
launch:             python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --dtype bfloat16 \
                      --tp-size 1 --host 127.0.0.1 --port 30000
request:            curl ... /v1/chat/completions (chapter §4.1) -> content: ______
log check:          model path, dtype, max_total_num_tokens=______, no errors; nvidia-smi: ____ GiB
benchmark:          python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 \
                      --model Qwen/Qwen3-8B --dataset-name random --random-input-len 256 \
                      --random-output-len 64 --random-range-ratio 1 --num-prompts 80 \
                      --max-concurrency 4 --seed 42
results:            successful ___/80, mean TTFT ___ ms, mean TPOT ___ ms, output tok/s ___
```

### The same protocol, executed on mini-sglang (CPU, 4 threads, Qwen3-0.6B fp32)

```bash
python -m minisgl --model /tmp/models/Qwen3-0.6B --device cpu --tp-size 1 \
    --num-tokenizers 1 --port 19190 --max-total-tokens 8192
curl -s --fail-with-body http://127.0.0.1:19190/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain KV cache in one sentence."}],"max_tokens":48,
       "temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
# -> "The KV cache is a type of memory cache used to store key-value pairs for faster access."
#    usage: prompt 20, completion 20 (finish_reason=stop)
python benchmarks/bench_serving.py --url http://127.0.0.1:19190 --concurrency 1,4 --num-requests 12 --max-tokens 32
```

| concurrency | requests (all succeeded) | mean TTFT s | p99 TTFT s | mean TPOT ms | output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.21 | 0.29 | 68 | 13.8 |
| 4 | 12 | 0.38 | 0.49 | 119 | 31.5 |

Reading it: raising concurrency from 1 to 4 gives 2.3× throughput, but TPOT rises 1.75×. A
decode step costs more with 4 rows (attention over 4 histories, the padded SDPA mask), and
new arrivals' prefills interrupt running decodes (prefill priority). TTFT almost doubles
from queueing behind those steps. (The answer is factually weak: a 0.6B model in non-thinking
mode. A smoke test checks that inference runs, not that it is accurate.)

## Exercise 2 — one-setting A/B

Protocol: stop the old server; change **one** flag; same inputs, concurrency, warmup and
starting cache state; repeat; report mean and range; and also check correctness.

Executed on mini-sglang with [`labs/part4_ch01_ab_compare.py`](../../labs/part4_ch01_ab_compare.py):
6 requests sharing a 384-token prefix, 8 output tokens, 3 repeats, fresh engine per
repeat (empty cache), identical warmup.

| cache_type | wall s | mean TTFT s | mean TPOT ms | output tok/s |
|---|---:|---:|---:|---:|
| naive | 7.54 [7.30, 7.83] | 5.43 [5.25, 5.61] | 301.65 [292.32, 317.07] | 6.37 [6.13, 6.58] |
| radix | 6.43 [6.36, 6.48] | 5.05 [4.96, 5.10] | 196.62 [192.92, 199.73] | 7.47 [7.41, 7.55] |

Correctness: greedy outputs were identical between the two configurations.

The ranges don't overlap, so radix is faster here: +17% throughput and −7% TTFT. The gain is
far smaller than "5 of 6 prefixes skipped" suggests. All 6 requests arrive together, and with
the default 2048-token prefill budget most of them land in the *first* prefill batch,
before anything has been inserted into the tree. So they all compute the shared prefix.
Radix only helps the requests admitted later. The TPOT drop comes from that shorter second
prefill step interrupting decodes for less time. This is a valid "smaller than expected"
result, and the profile explains it (see
[profiling walkthrough](profiling-walkthrough.md), optimization #2).

On SGLang the same experiment is: launch twice (with and without `--disable-radix-cache`),
send requests with a shared long prefix from the same client script, use
`POST /flush_cache` (or a restart) for the same starting state, and repeat 3+ times.
