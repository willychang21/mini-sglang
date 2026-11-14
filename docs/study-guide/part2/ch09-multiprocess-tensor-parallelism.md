# Part II · Chapter 9 — Multi-process and Tensor Parallelism

> Tutorial: `part2/Chapter9_Multi-process-and-Tensor-Parallelism.md`, `part2/Chapter9_Multi-process-and-Tensor-Parallelism_code.md` · Lab: `labs/part2_ch09_tp_shapes.py` · Bench: none (CPU only; correctness via `tests/test_tp.py`)

## Limitation → mechanism → validation

| Limitation | Mechanism | Validation here |
|---|---|---|
| One GPU bounds weights + KV capacity (and per-step compute) | Multi-process runtime (one scheduler + engine per rank, rank 0 does all I/O and broadcasts requests) + tensor parallelism (column / row / vocab-parallel layers, KV heads split or replicated, shard-then-merge loading) | **Correctness:** `tests/test_tp.py` runs TP=1 and TP=2 as separate processes over Gloo on CPU and requires equal prompt logits (atol 1e-4) and equal tokens, for a tied model with 2 KV heads and an untied model with 1 KV head (replicated). **Scaling:** not measured; this repo runs on CPU, where TP adds communication and no extra hardware. The lab computes the per-rank memory effect analytically from real layer shapes. |

## Key ideas

* **Two separate problems.** Multi-process execution is *organization*: API
  server, tokenizer(s), detokenizer and one scheduler process per rank, talking
  over ZMQ, so Python's GIL is not shared and I/O does not stall the GPU loop.
  Tensor parallelism is *computation*: the weights of each layer are split so
  every rank computes part of every token.
* **Control plane vs data plane.** Control plane: every rank must run the same
  batch in the same step (requests, order, lengths). Data plane: per-layer
  collectives (all-reduce, all-gather) that recombine partial results. A wrong
  control plane deadlocks or mismatches shapes; a missing collective silently
  produces wrong numbers.
* **Column parallel** splits a weight's output dim: each rank gets a slice of
  the outputs, no communication (`qkv_proj`, `gate_up_proj`).
  **Row parallel** splits the input dim: each rank produces a full-size partial
  sum, combined by all-reduce (`o_proj`, `down_proj`). Pairing column → row
  means one all-reduce per attention block and one per MLP.
* **Heads are the unit of attention sharding.** Q heads must divide by `tp`.
  KV heads are split while `num_kv_heads ≥ tp`; below that each rank keeps one
  whole KV head and ranks share copies (replication), so KV memory per rank stops
  shrinking.
* **Vocab parallel.** Embedding and LM head are split into `div_ceil(V, tp)` rows
  per rank (last rank zero-padded). The embedding looks up only its own id range
  and all-reduces; the LM head all-gathers logits along the vocab dim and trims the
  padding.
* **Replicated, not sharded:** RMSNorm weights, RoPE caches, the full-width
  hidden state `[N, hidden]` (the residual stream) and, in this engine, each
  rank's own page table, radix tree and scheduler state.
* **Shard, then merge.** Checkpoints store `q_proj`, `k_proj`, `v_proj`
  separately; the model has one fused `qkv_proj`. Each rank must slice each
  projection by heads first and then concatenate its slices. Concatenating first
  and chunking evenly gives rank 0 only Q rows and rank 1 only K/V rows.
* **Identical decisions instead of shipped batches.** Ranks receive the same
  messages in the same order and run the same deterministic scheduler, so they
  build identical batches without sending `Batch` objects around. Sampling uses
  the same seed on every rank and sees identical all-gathered logits.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| rank / size of this process | `python/minisgl/distributed/info.py:DistributedInfo`, `set_tp_info`, `get_tp_info` | layers read it in `__init__`, so it must be set before the model is built |
| collectives | `python/minisgl/distributed/impl.py:DistributedCommunicator.all_reduce`, `all_gather_last_dim`, `get_comm` | identity when `size == 1` |
| process group | `python/minisgl/engine/engine.py:Engine._init_communication` | none at tp=1; `gloo` on CPU, `nccl` + a separate Gloo CPU group on CUDA |
| set TP before building | `engine.py:Engine.__init__` | `set_tp_info(...)` is its first line; device `cuda:{rank}` only for CUDA with tp>1 |
| column parallel | `python/minisgl/layers/linear.py:ColumnParallelLinear` | weight `[sum(out_i / tp), in]` |
| fused QKV, KV replication | `layers/linear.py:QKVParallelLinear` | `local_kv_heads = div_even(num_kv_heads, tp, allow_replicate=True)` |
| row parallel | `layers/linear.py:RowParallelLinear` | weight `[out, in / tp]`, `all_reduce`, then bias |
| even split / replication rule | `python/minisgl/utils/__init__.py:div_even` | asserts divisibility; with `allow_replicate`, `a < b` → 1 per rank if `b % a == 0` |
| attention / MLP using local heads | `python/minisgl/models/utils.py:RopeAttn`, `GatedMLP` | split `qkv` by local sizes |
| vocab parallel | `python/minisgl/layers/embedding.py:VocabParallelEmbedding`, `ParallelLMHead` | mask + all-reduce; all-gather + trim to `num_embeddings` |
| KV pool heads | `python/minisgl/kvcache/mha_pool.py:MHAKVCache.__init__` | `local_kv_heads` via the same `div_even(..., allow_replicate=True)` |
| shard rules | `python/minisgl/models/weight.py:_shard_tensor` | dim 0 for q/k/v/gate/up, dim 1 for o/down, vocab rows for embed/lm_head, replicated otherwise; replicated KV head = `rank * num_kv_heads // tp` |
| streaming load + merge | `models/weight.py:load_weight`, `_get_merge_info`, `load_model_weights` | one full tensor at a time; yields `qkv_proj` / `gate_up_proj` once all slots of a group arrived |
| one process per rank | `python/minisgl/server/launch.py:start_backend`, `_run_scheduler` | `spawn` context, process names `minisgl-TP{rank}-scheduler`; waits for `num_tokenizers + 2` acks |
| per-rank config | `python/minisgl/server/args.py:ServerArgs.engine_config(rank)` | `DistributedInfo(rank, tp_size)` |
| rank-0 I/O + broadcast | `python/minisgl/scheduler/io.py:ZmqSchedulerIO.receive_msg`, `send_result` | rank 0 pulls from ZMQ, `broadcast_object_list` over the Gloo group; only rank 0 replies |
| deterministic decode order | `python/minisgl/scheduler/decode.py:DecodeManager.schedule_next_batch` | sorts the running set by `uid` |
| same random stream | `python/minisgl/engine/sample.py:Sampler.__init__` | every rank seeds the generator with `config.seed` |
| two-process test | `tests/test_tp.py`, `tests/tp_worker.py` | |

## This repo vs. the tutorial's reference code

* **No PyNCCL.** Collectives are plain `torch.distributed`; on CPU the Gloo
  backend carries both control and data. `DistributedCommunicator` has no plugin
  list (`plugins[-1]` in the reference) and exposes `all_gather_last_dim` instead
  of `all_gather`.
* **Request broadcast** uses one `broadcast_object_list` per scheduler step over
  Gloo, instead of a Gloo broadcast of the message count plus ZMQ PUB/SUB of the
  raw messages (README "Deliberate simplifications").
* **No process group at tp=1** (`_init_communication` returns `None`); the
  reference initializes Gloo even for a single GPU (for `_sync_get_memory`). This
  repo has no cross-rank memory probe: the KV size comes from
  `max_total_tokens` / `num_pages`, identical on every rank.
* **The model is built directly on the target device** with
  `create_model(config, device, dtype)` and filled by `load_model_weights`
  (shape-checked `copy_`), instead of building on `meta` and calling
  `load_state_dict`. The lab builds on `meta` itself to get shapes cheaply.
* **Dense Qwen3 only:** no MoE expert stacking in `load_weight`. It also skips
  layers `≥ num_layers` (truncated draft models, Ch10) and `lm_head` when
  embeddings are tied.
* **`num_tokenizers` defaults to 1** (`server/args.py`); the tutorial describes a
  default of 0 with a combined tokenizer/detokenizer worker. The ack count is
  `num_tokenizers + 2` in both.
* Processes are `daemon=True` and exit when the parent dies
  (`utils.exit_when_parent_dies`), see README.

## Common misconceptions

* **Misconception:** multi-process execution and TP are the same thing.
  **Actually:** TP=1 still runs four kinds of processes; TP decides only how many
  scheduler/engine processes there are and how each layer is split.
* **Misconception:** TP=2 halves memory per GPU.
  **Actually:** weights roughly halve, but RoPE caches, norms, the page table,
  activations of width `hidden` and (on GPU) CUDA graphs and NCCL buffers are
  per-rank copies, and KV stops shrinking once `num_kv_heads < tp` (lab
  section 3: Qwen3-0.6B KV/token/rank is 14 KiB at tp=8 *and* tp=16; total KV
  per token doubles to 224 KiB).
* **Misconception:** you can fuse q/k/v at load time and then split evenly.
  **Actually:** that hands rank 0 only Q rows (lab section 4: error 9.48 vs
  1.9e-6).
* **Misconception:** rank 0 sends each rank its batch.
  **Actually:** it forwards raw messages; every rank schedules for itself and
  arrives at the same batch.
* **Misconception:** row-parallel layers need an all-gather.
  **Actually:** their outputs are full-size partial sums; an all-reduce (sum)
  is needed. All-gather is for column-split outputs that must be concatenated
  (here only the LM head).

## Lab

```bash
python labs/part2_ch09_tp_shapes.py             # ~4 s; shapes, memory, shard order
python labs/part2_ch09_tp_shapes.py --run-tp    # +~4 s: real TP=2 over Gloo vs TP=1
```

Look for: column vs row shard dimensions, `vocab_range` per rank, the tiny
model's KV heads `[0],[0],[1],[1]` at tp=4, the memory table, and the
merge-order error.

```
| tp | rank |     qkv_proj | q/kv heads |       o_proj | gate_up_proj |    down_proj |          embed |      vocab_range | ...
|  2 |    1 | (2048, 1024) |        8/4 | (1024, 1024) | (3072, 1024) | (1024, 1536) |  (75968, 1024) |  (75968, 151936) | ...
...
| tp | weights/rank | ... | weight shrink | KV/token/rank | KV shrink | KV/token all ranks | ...
|  8 |   142.22 MiB | ... |         7.99x |     14.00 KiB |     8.00x |         112.00 KiB | ...
| 16 |    78.17 MiB | ... |        14.54x |     14.00 KiB |     8.00x |         224.00 KiB | ...
tp=16 qkv_proj/rank = (384, 1024) (not 4096/16=256 rows):
...
|  2 | shard q,k,v then cat (models/weight.py) |                     1.91e-06 |
|  2 |                    cat q,k,v then chunk |                     9.48e+00 |
...
prompt logits max |tp1 - tp2| = 4.05e-06; tokens tp1=[435, 33, 344, ...] tp2=[435, 33, 344, ...] (3.5 s)
```

## Exercises

### Concept chapter

**1. What do multi-process execution and TP each solve?**
Multi-process execution solves *system* problems: tokenization, detokenization,
HTTP and scheduling run in separate processes (no shared GIL, I/O overlaps
compute), and each GPU gets its own process and CUDA context. TP solves a
*model-size / per-step-compute* problem: one layer's weights (and KV heads) are
split so each GPU stores and multiplies only a slice. They work together because
TP needs one process per rank, and those processes need a control plane to stay
in lockstep. In this repo: `server/launch.py:start_backend` (processes),
`layers/linear.py` + `layers/embedding.py` (TP).

**2. Why can't each rank accept requests independently? What must stay consistent?**
Every layer ends in a collective that all ranks enter with tensors of the same
shape at the same time. If rank 1 batched different requests, `[N, hidden]`
would differ (Gloo/NCCL error or hang) or, with equal `N`, partial sums of
different tokens would be added (silently wrong). Consistent across ranks: the set
and order of requests, every admission and chunking decision (hence identical KV
capacity configuration), `cached_len`/`device_len` per request, the decode batch
order, and the sampled tokens. Mechanisms here: only rank 0 reads from ZMQ and
broadcasts each step's messages (`ZmqSchedulerIO.receive_msg`); the scheduler is
deterministic given those messages; decode batches are sorted by `uid`; the
LM head all-gathers so every rank sees full logits and samples with the same
seeded generator (`Sampler`). Correctness is checked by `tests/test_tp.py`.

**3. Why does doubling TP not halve memory or double speed?**
*Replication:* norms, RoPE caches, page table, activations of width `hidden`,
runtime buffers stay full-size on every rank. *KV heads:* once `num_kv_heads <
tp` each rank keeps a whole KV head (and the K/V projection rows), so KV/token
per rank stops shrinking and total KV grows (lab: tp=16 on Qwen3-0.6B, weight
shrink 14.5x, KV shrink stuck at 8x). Also vocab padding to `div_ceil(V, tp)`.
*Communication:* each layer adds two all-reduces of `[N, hidden]` plus embedding
all-reduce and LM-head all-gather; at small batch the matmuls get smaller while
the latency of these collectives does not, so per-step time falls sub-linearly or
even rises (on this CPU repo, TP=2 is only a correctness demo).

### Code walkthrough

**1. What does each rank's Scheduler do (multi-GPU)?**
It is a full replica of the control logic for its rank: it owns that rank's
`Engine` (weight shard, KV pool with local heads), `TableManager`,
`CacheManager` (page allocator + radix tree) and prefill/decode managers, and it
runs the same `normal_loop`. Rank 0 additionally talks to the tokenizer and
detokenizer. Running the scheduler on every rank (instead of shipping batches)
keeps per-rank state (page tables, slot numbers) local and consistent. Code:
`server/launch.py:_run_scheduler` builds `Engine` + `ZmqSchedulerIO` +
`Scheduler(config, io, engine)` per rank; only rank 0 sends the ready ack.

**2. How are weights loaded, and how do ranks process the same batch?**
Single device: `_shard_tensor` returns the full tensor (`n == 1`), and
`load_weight` only merges q/k/v and gate/up. Multi-device: every rank opens the
same safetensors files independently, reads one tensor at a time, slices its
shard (`_shard_tensor`), merges per-rank slices into fused parameters, and
`load_model_weights` copies them into the pre-sized layers (shape-checked).
Coordination: at tp=1 there is nothing to coordinate. At tp>1 rank 0 broadcasts
each step's messages over the Gloo CPU group (`ZmqSchedulerIO.receive_msg`,
`broadcast_object_list`; the reference sends a count over Gloo and the payload
over ZMQ PUB/SUB). All ranks then make identical scheduling decisions; decode
order is fixed by sorting on `uid`.

**3. Why stream weights, and why yield only after merging?**
Streaming keeps peak host memory at about one full checkpoint tensor plus the
merge buffer, instead of the whole checkpoint (`f.get_tensor(name)` → shard →
`del raw`). The model has fused parameters (`qkv_proj.weight`,
`gate_up_proj.weight`), so a single `q_proj` slice has no destination of the
right shape; `load_weight` buffers slots in `merge_buf` and yields
`torch.cat(slots)` once all 3 (or 2) per-rank slices exist, and asserts at the end
that no group is incomplete. (MoE expert stacking from the reference is not
implemented here.)
