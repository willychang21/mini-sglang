# Part III · Ch5 — Prefill-Decode Disaggregation

Course text: `part3/Chapter5_Prefill-Decode-Disaggregation.md`. Lab:
[`labs/part3_ch05_pd_transfer.py`](../../labs/part3_ch05_pd_transfer.py). The interference itself shows up
in our own profile and A/B runs.

## Key ideas

- On shared GPUs, a long prefill blocks every running decode for one step, which causes an
  **ITL spike**. Average TPOT can hide it. Chunked prefill bounds each spike, but the two
  phases still share the GPU.
- **PD disaggregation.** P runs prefill and samples the first token. The KV (plus the first
  token and request state) is transferred to D. D decodes the rest. Each side holds a full
  model and is sized and tuned independently.
- **Costs.** `T_transfer ≈ S_KV / B_effective`, link contention between concurrent
  transfers, two model replicas, and transient double KV storage. It suits steady traffic
  with distinct phase demands, not a few short requests.
- **Optimizations.** A smaller KV dtype, faster transport (RDMA, no bounce copies),
  overlapping transfer with prefill (layer by layer), and reusing a prefix already on D so
  only the missing suffix moves.

## How this shows up in mini-sglang

mini-sglang runs both phases in one process, and `scheduler/scheduler.py` (`_schedule_next_batch`) gives
**prefill priority**. So the interference the chapter describes is measurable here. From the
profile ([walkthrough](../part4/profiling-walkthrough.md)), on CPU:

- a decode step for 8 requests takes **~256 ms**;
- a prefill step of ~450 tokens takes **~1240 ms**. Any request already decoding waits that
  long for its next token, a **~5× ITL spike**. Four back-to-back prefill steps (5.0 s) pass
  before any decode runs.

The A/B lab shows the coupling another way. Enabling the radix cache lowers mean TPOT from
**302 → 197 ms**, even though it changes nothing in decode. The second prefill step gets
shorter (cached prefix), so decoding requests are blocked for less time.

Where PD would cut in mini-sglang: after `_process_last_data` handles a prefill batch, the
`Req` (its `page_table` row and KV slots) would be serialized and sent instead of entering
`DecodeManager`. The decode side would allocate slots (`CacheManager.allocate_paged`), receive
K/V into them (`MHAKVCache.store_kv`), write the page-table row, and only then add the request
to `running_reqs`. `pd_transfer.py` measures what would be shipped: for a 512-token prompt on
Qwen3-0.6B in bf16, gathering the request's slots from all 28 layers gives **56.0 MiB**,
exactly `512 × 2·28·8·128·2` bytes.

Transfer time for Qwen3-8B (144 KiB/token, bf16):

| prompt | KV | 5 GiB/s | 20 GiB/s | 50 GiB/s |
|---:|---:|---:|---:|---:|
| 4K | 0.56 GiB | 112 ms | 28 ms | 11 ms |
| 16K | 2.25 GiB | 450 ms | 112 ms | 45 ms |
| 128K | 18 GiB | 3.6 s | 900 ms | 360 ms |

With 8 concurrent 16K transfers sharing one 20 GiB/s link, each waits ~900 ms instead of 112 ms.

## SGLang pointers

- `srt/disaggregation/prefill.py::SchedulerDisaggregationPrefillMixin`, `PrefillBootstrapQueue`
  (handshake, then send KV after prefill).
- `srt/disaggregation/decode.py::DecodePreallocQueue` (allocate KV on D *before* transfer),
  `DecodeTransferQueue` (poll until KV arrives, then join the running batch).
- Transport interface: `srt/disaggregation/base/conn.py::BaseKVManager`, `BaseKVSender`,
  `BaseKVReceiver`, `KVPoll`. Implementations: `mooncake/conn.py`, `nixl/`, `mori/`, plus
  `fake/` (useful for reading the protocol without RDMA).
- D-side HiCache: `srt/disaggregation/decode_hicache_mixin.py`.

## Exercises — worked answers

**1. A conversation pauses after a long-document request, but mean TPOT barely moves. Which
metric? How do chunked prefill and PD help?**
Look at **ITL** (inter-token latency) per request, especially its max and p99, or the
per-step timeline. TPOT averages the one long gap over many normal ones. Chunked prefill
splits B's prefill into bounded chunks, so A waits at most one chunk per step. The pause
becomes many smaller ones, and B's total prefill takes more steps. A can also share a step
with B's chunk (mixed batches). PD runs B's prefill on other GPUs entirely, so A's decode
steps on D are not interrupted. A only competes with other decodes and with the cost of
receiving new requests' KV.

**2. Trace a request from P to its second token on D. Where is the first token produced? Do
the sides split the layers?**
The router picks a P and a D. D pre-allocates KV slots for the request (and the handshake
exchanges addresses). P runs prefill through **all** layers, writing K/V into its pool,
and samples the **first token on P** from the last position's logits. The KV (all layers,
possibly sent layer by layer during prefill), the first token and the request metadata
are transferred to D's pre-allocated slots. D marks the request ready, adds it to its
running batch, feeds the first token as input, and produces the **second token** in its first
decode step. P frees its copy (or keeps it in its prefix cache). Neither side holds half the
layers: both run the full model. PD splits *phases*, not layers (that would be PP).

**3. 2 GiB at 10 GiB/s; then contention halves bandwidth.**
2 / 10 = **0.2 s = 200 ms**. At 5 GiB/s: **400 ms**. That adds 200 ms to this request's
time-to-second-token and keeps its KV slots on both sides occupied longer.

**4. Should PD be an early option?**
- Long documents arrive continuously and existing users see frequent streaming pauses: **yes**,
  a good fit (prefill interference). Check that the interconnect can carry the KV volume
  (prompt length × bytes/token × arrival rate), that there are GPUs for two replicas, and
  compare with tuning chunked prefill first.
- A few short Q&A requests and an idle GPU group: **no**. There is no interference to
  remove, and PD adds transfer latency and leaves one side idle. Keep a single co-located
  deployment.
- Continuous long responses, D saturated, P idle: PD **already exists** here and is
  imbalanced. Rebalance: move GPUs from P to D (or use fewer P instances). Check D's batch
  size, KV capacity and TPOT, and whether D is memory- or compute-bound. Consider
  quantization or larger batches on D.

**5. A published result combines PD + EP + larger batches. Is all the speedup from PD?**
No. It is a combined effect. To isolate PD, compare co-located vs PD with the **same**
hardware count, model, precision, parallelism (EP/TP degrees), batch-size limits, attention
backend, CUDA-graph settings, request distribution (input/output lengths, arrival rate),
warmup and metric definitions. Report TTFT, ITL p99 and throughput per GPU. Ablate one
factor at a time (PD on or off with EP fixed, and so on).

**6. P restores a prefix from HiCache; D has no copy. What is saved, and what moves?**
Saved: P's recomputation of the prefix (the prefill FLOPs for those tokens). P only
computes the new question after restoring the prefix into its GPU. Still moving: the
**entire** KV of the prompt (prefix + suffix) from P to D, because D needs all of it to
decode. Plus, internally on P, the host→GPU restore. To reduce transfer: D needs its own
prefix cache (or access to a shared L3) holding the same prefix, and a transfer protocol in
which D reports its matched prefix length during the handshake so that P sends only the
missing suffix. Ideally D loads the prefix itself from the shared L3 storage (the
"Prefix Reuse and KV Transfer" figure).
