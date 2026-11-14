# Part I · Chapter 2 — Introduction to Inference

> Tutorial: `part1/Chapter2_Introduction to Inference.md` · Lab: `labs/part1_ch02_roofline.py` ·
> Related code: `python/minisgl/scheduler/scheduler.py`, `scheduler/prefill.py`,
> `scheduler/decode.py`, `core.py`, `llm/naive.py`, `benchmarks/bench_kv_cache.py`

## Key ideas

- **Training vs. inference.** Both compute `P(t_i | t_<i)`. Training feeds the
  *ground-truth* sequence and uses a causal mask, so all positions are computed in one
  parallel forward pass, followed by a backward pass (about 6N FLOPs per token).
  Inference feeds *its own* outputs back, so it is **sequential across steps and
  parallel within a step**. It only runs forward (about 2N FLOPs per token), and it
  keeps a KV cache instead of activations for backprop. A single request is cheap
  (8B model, 512+128 tokens: about 1×10¹³ FLOPs), but serving volume is huge, so
  yearly inference cost can exceed training cost.
- **Training fixes the ceiling for inference.** Choices made before training decide
  inference cost: layer count and widths set weight bytes, GQA versus MHA sets KV
  bytes, and QAT decides whether quantization is safe. You cannot convert an MHA
  checkpoint to GQA at deploy time.
- **Life of a request:** schedule → **prefill** (all prompt tokens in one pass; write
  their K/V; sample the first token) → **decode** (one token per step; read all
  cached K/V, append one) → detokenize and return.
- **Prefill** does a lot of math on data that is loaded once, so its arithmetic
  intensity is high. **Decode** re-reads all weights plus the growing KV cache to
  produce *one* token per request, so its AI is about 1 FLOP/byte. The speed
  bottleneck of decode is memory bandwidth. Its capacity bottleneck is the size of
  the KV cache.
- **Roofline.** Attainable FLOP/s = `min(peak, AI × bandwidth)`. The **ridge point**
  `peak / bandwidth` separates memory-bound (left) from compute-bound (right). For a
  GEMM `[M,K]×[K,N]` in bf16, `AI = MKN / (MK + KN + MN)`. It grows with the smallest
  dimension, and for a linear layer that is the number of tokens in the step.
- **A result the chapter implies but does not state:** with its byte model, prefill
  AI equals N *exactly*:
  `(24Nd² + 4N²d) / (2·(12d² + 2Nd)) = N(24d² + 4Nd) / (24d² + 4Nd) = N`.
  So prefill is compute-bound once the prompt has more tokens than the ridge point
  (≥ 195 tokens on an A100-40GB). Decode AI tends to 1 as S grows.
- **The scenarios make it concrete.** In agent loops and multi-turn chat, each turn
  re-prefills a context that keeps growing, so TTFT grows linearly with turns. A long
  context also means a large KV cache per conversation, so fewer requests fit in a
  batch and each decode step reads more bytes. The fixes come later in the course:
  prefix caching (Part II Ch8), chunked prefill and continuous batching (Part II Ch6),
  KV quantization and offload (Part III).

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| Prefill vs. decode phase | `python/minisgl/core.py:Batch` (`phase`, `is_prefill`, `num_tokens`) | one batch is either all-prefill or all-decode |
| Prefill-first scheduling | `python/minisgl/scheduler/scheduler.py:Scheduler._schedule_next_batch` | runs a prefill batch if one fits, otherwise a decode batch |
| Prefill batch (many tokens/request) | `python/minisgl/scheduler/prefill.py:PrefillManager.schedule_next_batch`, `PrefillAdder.try_add_one` | `prefill_budget` caps tokens per step (chunked prefill) |
| Decode batch (1 token/request) | `python/minisgl/scheduler/decode.py:DecodeManager.schedule_next_batch` | every running request contributes `extend_len = 1` |
| "Tokens computed this step" | `python/minisgl/core.py:Req.extend_len` (`device_len - cached_len`) | the `B` of the roofline GEMM analysis |
| Prefill writes KV, decode appends | `python/minisgl/kvcache/mha_pool.py:MHAKVCache.store_kv`, called from `attention/torch_backend.py:TorchAttnBackend.forward` | store first, then attend |
| First token comes from the prompt's last position | `python/minisgl/layers/embedding.py:ParallelLMHead.forward` | `x[last_indices]` in prefill |
| Inference without a KV cache (O(n²) recompute) | `python/minisgl/llm/naive.py:generate_without_kv_cache` | compare with `benchmarks/bench_kv_cache.py` |
| Training vs. inference compute | — | mini-sglang runs forward only (`@torch.inference_mode()` on `Engine.forward_batch`) |

Measured effect (README, `benchmarks/bench_kv_cache.py`, CPU): without a cache TPOT
grows with length (326 → 415 ms). With the cache it stays flat (≈ 70–80 ms). That is
decode re-reading cached K/V instead of recomputing the prefix.

## Common misconceptions

**Misconception:** Prefill is always compute-bound.
**Actually:** Only when N > ridge. Ch2 itself shows that LLaMA-7B prefill with N=100 on
an A100 (AI = 100 < 195) is still memory-bound. Ch3's summary table ("Prefill:
compute-intensive") describes the usual case of long prompts or batched prefill.

**Misconception:** Decode is slow because it does a lot of math.
**Actually:** It does very little math: 14.4 GFLOPs per token for LLaMA-7B, about
0.05 ms at peak. It is slow because it has to move ~15.5 GB per token (about 9.7 ms at
1.6 TB/s).

**Misconception:** Batching more decode requests always makes decode compute-bound.
**Actually:** Batching amortizes the *weights* across requests. *KV-cache* bytes grow
with the batch just as fast as the attention FLOPs do. For Qwen3-0.6B with 4K-token
contexts, the decode AI levels off at about 4.5 FLOP/B however large B gets (lab
section 4).

**Misconception:** Training and inference are separate worlds.
**Actually:** Training runs inference for validation and RL rollouts, and inference
cost is decided at training time by the architecture (GQA, layer count, QAT).

**Misconception:** The tutorial's roofline byte counts are complete.
**Actually:** They are deliberately simplified. The prefill byte formula leaves out the
KV writes that Sec. 4.2 counts, both use MHA (`2Sd` KV per layer), and the FFN width
`8/3·d = 10923` stands in for LLaMA-7B's real 11008. The conclusions still hold.

## Lab

```bash
python labs/part1_ch02_roofline.py            # ~3 s; --model tiny|<path>, --skip-measure
```

What to look for: section 2 reproduces every number in the chapter. Section 3 shows
that a linear layer's AI ≈ B (tokens in the step). Section 4 shows why long-context
decode stays memory-bound. Section 5 measures a GEMV versus a GEMM on this CPU.

```text
== 2. Ch2's LLaMA-7B worked example, recomputed
|             prefill FLOPs, N=3000 |          43.4 TFLOP | 43.4 TFLOPs |
|        decode FLOPs/token, S=3000 |          14.5 GFLOP | 14.4 GFLOPs |
| decode bytes/token (14 GB W + KV) |             15.6 GB |     15.5 GB |
|               -> time at 1.6 TB/s |              9.7 ms |      9.7 ms |
|                 AI_prefill, N=100 |        100.0 FLOP/B |        ~100 |
|                  AI_decode, S=100 |        1.000 FLOP/B |          ~1 |
-> AI_prefill(N) = N exactly, so LLaMA-7B prefill crosses the A100-40GB ridge at N >= 195 tokens

== 3. per-layer linear GEMMs of the analysed model (/tmp/models/Qwen3-0.6B/config.json)
|        layer | weight [in x out] | AI B=1 | AI B=8 | AI B=64 | AI B=256 | AI B=1024 | AI B=4096 |
|     qkv_proj |         1024x4096 |    1.0 |    7.9 |    59.4 |    195.0 |     455.1 |     682.7 |
smallest B with AI >= ridge (decode linears become compute-bound):
|      A100 40GB |   195 |      256 |    273 |          251 |       262 |

== 4. whole decode step: weights are shared by the batch, KV is not
| context S | AI B=1 | AI B=64 | AI as B->inf | B for A100-40GB ridge |
|         0 |   1.00 |    63.6 |          inf |                   199 |
|       512 |   1.05 |    16.9 |         22.3 |                 never |
|      4096 |   1.28 |     4.4 |          4.5 |                 never |
```

(14.5 vs. 14.4 GFLOPs is rounding: the tutorial rounds `24d²` to 4.0×10⁸.)

## Exercises

**1. Differences and connections between training and inference.**

| | Training | Inference |
|---|---|---|
| Parameters | updated (forward + backward + optimizer) | frozen (forward only) |
| Input for `t_i` | ground-truth prefix (teacher forcing) | prompt + the model's own outputs |
| Parallelism | all positions at once (causal mask) | sequential across steps, parallel within a step |
| FLOPs/token | ≈ 6N (2N forward + 4N backward) | ≈ 2N |
| Memory | weights + gradients + optimizer state + saved activations | weights + **KV cache** (+ small temporary activations) |
| Cost profile | one huge job (8B on 160B tokens ≈ 7.7×10²¹ FLOPs ≈ 90 H100-days at peak) | small per request (≈ 10¹³ FLOPs), but continuous and at scale |
| Metric | loss / accuracy | latency (TTFT, TPOT) and throughput per dollar |

Connections: training runs inference inside its loop (validation, RL rollouts scored
by a reward model). Inference behaviour (alignment problems, failure cases) feeds back
into training data and methods. Training-time architecture choices (GQA, depth/width,
QAT) set the upper limit of inference efficiency.

**2. Compare prefill and decode and analyze each bottleneck.**

| | Prefill | Decode |
|---|---|---|
| Tokens per request per step | whole prompt (N) | 1 |
| Kernel shape | GEMM `[N, C]×[C, C']` | GEMV for one request (GEMM `[B, C]×…` across a batch) |
| Parallelism | across tokens and requests | across requests only; steps are serial |
| KV cache | written (N entries/layer) | read in full (S entries/layer), 1 appended |
| AI (Ch2 model, bf16) | = N | → 1 |
| Bottleneck | compute (Tensor Cores) once N > ridge | HBM bandwidth (weights + KV); capacity: KV size |
| Latency metric | TTFT | TPOT / ITL |
| Typical optimizations | batching / chunked prefill, fused kernels, prefix caching to skip work | batching to amortize weights, quantization (fewer bytes), KV compression and reuse, speculative decoding |

**3. Significance of the roofline model.**

It turns "is this kernel slow?" into a comparison you can do in your head. The x-axis
is arithmetic intensity (FLOP per byte moved to and from DRAM). The y-axis is
attainable FLOP/s. A platform gives two roofs: a slanted one, `bandwidth × AI`, and a
flat one, `peak FLOP/s`. They meet at the **ridge point** `peak / bandwidth`
(A100-40GB: 312 T / 1.6 T = 195 FLOP/B). A workload left of the ridge is
memory-bound: speed it up by moving fewer bytes or reusing them (batching, fusion,
quantization). A workload right of the ridge is compute-bound: speed it up by doing
fewer or cheaper FLOPs (Tensor Cores, lower precision, less redundant math). The
model also gives an upper bound, so you know how far an implementation is from the
hardware limit.

**4. Use the roofline to identify a bottleneck in a concrete scenario.**

Scenario: Qwen3-0.6B (bf16), single-user chat decode on an A100-40GB (tutorial
numbers: 312 TFLOP/s, 1.6 TB/s → ridge 195).

1. Hardware: ridge = 312×10¹² / 1.6×10¹² = **195 FLOP/B**.
2. Workload, B = 1, short context. Weights read per step:
   `(28 layers × 15,728,640 + 151,936 × 1024 LM head) × 2 B = 595,984,384 × 2 =
   1.19 GB`. FLOPs ≈ 2 × 595,984,384 = 1.19 GFLOP. **AI ≈ 1.0 FLOP/B** (lab section 4,
   S=0, B=1).
3. Compare: 1.0 ≪ 195, so it is memory-bound. The floor is the memory time,
   1.19 GB / 1.6 TB/s = **0.745 ms/token** (at most ≈ 1340 tokens/s for one stream).
   The compute time would be 1.19 GFLOP / 312 TFLOP/s = 3.8 µs, about 195× less.
4. Prediction and what to do: batching B requests divides the weight traffic per
   token by B. The linear layers reach the ridge at B ≈ 250–270 (lab section 3:
   qkv 256, o 273, gate_up 251, down 262). But with 4K-token contexts, the KV reads
   grow with B, and the whole step tops out at AI ≈ 4.5. So for long contexts the
   levers are fewer KV bytes (GQA/MLA, KV quantization), not just bigger batches.
5. The same comparison for **prefill** of a 3000-token prompt on LLaMA-7B: AI = N =
   3000 > 195, so it is compute-bound, ≈ 43.4 TFLOP / 312 TFLOP/s ≈ 0.14 s at peak.

The CPU part of the lab shows the same effect on real hardware. A GEMV runs at
19.6 GFLOP/s while streaming 39 GB/s. The same weights with 512 rows reach
476 GFLOP/s (24× higher).

**5. RAG + LLM over financial documents, long multi-turn conversation: main
bottlenecks?**

- **Prefill grows every turn.** Each turn's prompt is system prompt + retrieved chunks
  + full history + the new question. With ~2–3K tokens of retrieved text per turn
  plus history, the prompt soon reaches tens of thousands of tokens. Without reuse,
  the whole thing is prefilled again every turn, so TTFT grows linearly with turns
  (Ch2's example: +200 tokens/turn at 1000 tok/s means 0.2 s → 2 s by turn 10), and
  the total prefill work over a conversation grows quadratically. At long N, the
  `4N²d` attention term also starts to dominate prefill FLOPs.
- **Prefix reuse is fragile.** A prefix cache (RadixAttention, Part II Ch8) only
  reuses an *identical token prefix*. If the retrieved chunks are inserted before the
  history and change every turn, everything after them is a cache miss. Putting
  stable content first (system prompt, then history, then new retrievals) keeps the
  prefix reusable.
- **KV memory and concurrency.** One 30K-token conversation on Qwen3-4B holds
  30,000 × 147,456 B ≈ 4.1 GiB of KV (Ch4 formula). A few such users fill a GPU, so
  the batch size, and with it throughput, drops.
- **Decode becomes KV-bound.** Each generated token reads the whole context's KV. With
  long S the decode AI falls towards the GQA ratio (lab section 4: 2.3 FLOP/B at
  S=32K), so TPOT rises with conversation length even at batch size 1.
- **Retrieval-side latency** (embedding the query, vector search) also adds to TTFT,
  outside the engine.
- Mitigations: prompt layout that keeps the prefix stable, prefix caching, chunked
  prefill (so long prefills do not stall other users' decode), fewer and better
  reranked chunks, summarizing or truncating history, KV quantization, and
  hierarchical KV offload to CPU memory (HiCache, Part III Ch3).
