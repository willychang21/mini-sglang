# Part III · Ch4 — Scaling Out: DP Attention, EP, PP

Course text: `part3/Chapter4_Scaling-Out.md`. Lab:
[`labs/part3_ch04_parallel_comm.py`](../../labs/part3_ch04_parallel_comm.py), with the TP rows of
[`labs/part3_ch02_kv_capacity.py`](../../labs/part3_ch02_kv_capacity.py).

## Key ideas

- **TP limits.** Each extra rank cuts per-rank FLOPs, but every layer still does 2
  all-reduces (after `o_proj` and `down_proj`). In small-batch decode, the latency of those
  2·L collectives soon outweighs the compute saved. KV cannot be split further than the
  KV-head count (GQA), and MLA's latent cache is replicated under TP.
- **DP Attention.** Each attention group owns whole requests and their KV, so nothing is
  duplicated. The FFN/MoE is still shared, so tokens are gathered before it and scattered
  after. Capacity grows with the duplication removed; throughput must still be measured.
- **EP.** Experts are spread across GPUs. Each MoE layer does **dispatch** (all-to-all of
  activations to the ranks that own each token's top-k experts), expert compute, and
  **combine** (all-to-all back plus a weighted sum). Load imbalance (hot experts) makes
  everyone wait for the slowest rank (hence EPLB).
- **PP.** Contiguous layers form stages, and activations cross stage boundaries
  point-to-point. A single micro-batch leaves bubbles, and micro-batches fill the pipeline.
  Autoregressive decode can't put a request's *future* token in the next micro-batch;
  micro-batches must be different requests.

## How this extends mini-sglang

mini-sglang implements plain TP (`python/minisgl/layers/linear.py`, `distributed/impl.py`,
`models/weight.py`). The lab confirms the cost model against the code: there are exactly 2
`RowParallelLinear` per layer, so 2 all-reduces per layer, plus one all-gather of
vocab shards for the LM head.

GQA replication already appears in our code. `QKVParallelLinear` and `MHAKVCache` use
`div_even(num_kv_heads, tp, allow_replicate=True)`, and `models/weight.py::_shard_tensor`
gives each rank `head_idx = rank * num_kv_heads // n` when `num_kv_heads < n`. With
Qwen3-0.6B (8 KV heads) at TP=16, two ranks hold identical KV. `kv_capacity.py` prints
Qwen3-32B KV bytes per token per rank: 256 → 128 → 64 → 32 → **32** KiB for TP 1, 2, 4, 8, 16.

Communication per GPU per decode step (64 tokens, bf16 activations, NVLink 3.6 Tb/s,
bandwidth only; latency is extra):

| model, GPUs | TP | DP-Attn + TP-MLP / + EP | PP (per boundary) |
|---|---:|---:|---:|
| Qwen3-32B, 8 | 140 MiB, 128 collectives | 70 MiB | 0.6 MiB × 7 hops |
| DeepSeek-V3, 8 | 187 MiB | 93 MiB (EP) | 0.9 MiB × 7 |
| DeepSeek-V3, 32 | 207 MiB | **26 MiB** (EP) | 0.9 MiB × 31 |

TP traffic per rank *grows* toward 2·act per collective as n grows, but EP traffic per rank
*shrinks*: each rank only dispatches its own tokens/n, top-k times. PP moves the least data,
but every token still crosses all stages in sequence, so per-token latency doesn't drop.

What mini-sglang cannot show: there is no MoE model and no multi-request DP group, and Gloo
on CPU has no meaningful bandwidth profile. A natural next step would be a DP wrapper:
N `Scheduler`s, each owning a subset of requests, with a router picking a rank by load
or prefix match. That is SGLang's `DataParallelController` without the attention-only part.

## SGLang pointers

- `srt/layers/dp_attention.py` (gather/scatter buffers, `DpPaddingMode`),
  `srt/managers/data_parallel_controller.py::DataParallelController` (`LoadBalanceMethod`).
- MoE: `srt/layers/moe/topk.py::TopK`, `srt/layers/moe/fused_moe_triton/layer.py::FusedMoE`,
  `srt/layers/moe/ep_moe/layer.py::DeepEPMoE`, `srt/layers/moe/token_dispatcher/deepep.py`
  (dispatch/combine), `srt/eplb/` (expert load balancing).
- PP: `srt/managers/scheduler_pp_mixin.py::SchedulerPPMixin.event_loop_pp`, and
  `PPProxyTensors` in `srt/model_executor/forward_batch_info.py`.
- Groups: `srt/distributed/parallel_state.py::initialize_model_parallel`.
- Real issue in this area: [#50137](https://github.com/sgl-project/sglang/issues/50137)
  (an abort is lost under DP attention because a control message reaches only DP rank 0).
  It is a concrete example of the "every rank must agree" invariant our `ZmqSchedulerIO`
  broadcast enforces for TP.

## Exercises — worked answers

**1. Matching.**
- Several GPUs share one linear layer's weights and compute → **TP**. Cost: an all-reduce (or all-gather) per sharded block, 2 per layer, on the critical path, and small per-rank GEMMs at high TP.
- MLA caches replicated across GPUs, limiting capacity → **DP Attention**. Cost: attention weights replicated per DP group, load imbalance across groups, and gather/scatter around the FFN/MoE (plus padding when groups have different token counts).
- Too many MoE expert weights for one GPU → **EP**. Cost: two all-to-alls per MoE layer (dispatch and combine) and hot-expert imbalance.
- Consecutive layers on different machines, batches flowing through → **PP**. Cost: pipeline bubbles, stage imbalance, inter-stage activation transfers, and no per-token latency gain.

**2. GQA with 8 KV heads: why TP 8→16 doesn't halve per-GPU KV.**
Sharding by whole KV heads gives one head per rank at TP=8. At TP=16 there are only 8 heads
for 16 ranks, so each head is replicated on 2 ranks (our `_shard_tensor` does exactly this,
`head_idx = rank * 8 // 16`). Each rank still stores one full KV head per token. The query
heads halve, but per-GPU KV stays the same (the lab: 32 KiB/token at both TP 8 and TP 16 for
Qwen3-32B). Splitting inside `head_dim` is possible in principle, but attention needs the full
head dimension for the dot products, so it would add communication inside attention.

**3. 4× capacity from removing 4× KV replication: why not 4× throughput?**
Capacity is only an upper bound on batch size. Throughput also depends on (a) whether
enough requests arrive to fill it; (b) compute and bandwidth. A 4× bigger decode batch
reads 4× more KV in attention and eventually becomes compute-bound in the GEMMs; (c) the
gather/scatter communication around the FFN, which grows with tokens; (d) load imbalance,
since the step waits for the busiest DP group; (e) memory taken by replicated attention
weights and buffers, which shrinks the real gain below 4×. Measure it.

**4. Token on GPU 0 picks experts on GPUs 1 and 3: the path; does its KV travel?**
On GPU 0: attention (using GPU 0's local KV for that request), then the router computes
top-k (E2, E7) and weights. **Dispatch**: the hidden-state vector (one `hidden`-sized
activation, possibly quantized for transport) goes to GPU 1 and GPU 3. GPU 1 runs E2 and
GPU 3 runs E7 on their received token batches. **Combine**: both outputs return to GPU 0,
which forms the routing-weighted sum (plus any shared expert) and adds the residual. The
next layer's attention then runs on GPU 0 again. The KV history **never moves**; only the
current token's activation does, and it stays where attention ran.

**5. Two experts per GPU: equal load?**
No. Equal *placement* is not equal *work*. The work per GPU is the number of tokens routed
to its experts in this batch, which depends on the input distribution and can be very
skewed (popular experts, domain-specific traffic). Measure tokens per expert and per rank
per layer over time, per-rank MoE kernel time, all-to-all wait time (fast ranks idling),
and the max/mean ratio. Remedies: redundant replicas of hot experts and rebalancing
(EPLB), capacity limits, or a different grouping.

**6. Three stages, two micro-batches A and B.**

```
time slot:   1     2     3     4
stage 1:     A     B     .     .
stage 2:     .     A     B     .
stage 3:     .     .     A     B
```

4 slots for 2 micro-batches (vs 6 without overlap). The bubbles are the 6 idle cells: stage 2
and 3 at the start (waiting for data to arrive) and stage 1 and 2 at the end (draining).
Utilisation is 6/12 = 50%, compared with 12/18 = 67% for four micro-batches. A request's next
token cannot be a separate micro-batch because it does not exist yet: its input is the
token sampled at the end of the current pass through **all** stages. Autoregressive
dependency serialises a single request, so only *different* requests can fill the pipeline.
