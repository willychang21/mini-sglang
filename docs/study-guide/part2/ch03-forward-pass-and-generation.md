# Part II · Chapter 3 — Forward Pass and Generation

> Tutorial: `part2/Chapter3_Forward-Pass-and-Generation.md`, `part2/Chapter3_Forward-Pass-and-Generation_code.md` · Lab: `labs/part2_ch03_forward_shapes.py` · Bench: — (correctness chapter; the cost of the no-cache loop is measured in Ch4's `benchmarks/bench_kv_cache.py`)

## Limitation → mechanism → validation

| Limitation | Mechanism | Primary validation | Where validated here |
|---|---|---|---|
| There is no working autoregressive generation yet | Forward pass (embedding → N decoder layers → final norm → LM head), sampling, generation loop | Output correctness | `tests/test_model.py` (prefill logits vs HuggingFace; `generate_without_kv_cache` vs HF greedy; incremental extend == full prefill), `tests/test_real_model.py` (Qwen3-0.6B token-for-token), `tests/test_scheduler.py::test_sampling_is_seeded_and_respects_top_k`, lab `part2_ch03_forward_shapes.py` |

## Key ideas

* **Forward path.** ids → embedding rows (`[vocab, C]` lookup, no context) →
  N decoder layers (attention mixes tokens, MLP transforms each token) → final
  RMSNorm → LM head → logits. Position enters only through RoPE on Q and K.
* **Packed layout `[N, C]`.** The engine never builds `[B, T, C]`. All tokens
  computed this step, from all requests, are concatenated along one axis;
  request boundaries live in `batch.attn_metadata`, not in the shape. No padding
  in the linear layers, which are the bulk of the FLOPs.
* **Pre-norm with a fused residual.** Math: `h = x + Attn(Norm(x))`,
  `out = h + MLP(Norm(h))`. Code: `RMSNormFused(x, residual)` returns
  `(Norm(x + residual), x + residual)`, so each add is folded into the *next*
  norm; the last MLP's add happens in the final `model.norm`.
* **Qwen3 attention.** Fused QKV projection → split → per-head RMSNorm on Q and
  K (QK-norm) → RoPE on Q and K (not V) → attention with GQA (`Hq` query heads
  share `Hkv` K/V heads) → `o_proj` back to `C`. `head_dim` is its own config
  value; `Hq·D = C` is a coincidence of the config, not a rule.
* **Causal mask.** Query `t` may only see keys `≤ t` (in the engine: `≤
  cached_len + t`). It makes position `t`'s output a function of tokens `0..t`
  only; that is what allows prefill to compute all positions in parallel and
  what makes caching their K/V valid.
* **Only the last position predicts something new.** Position `t`'s logits score
  token `t+1`. In inference all earlier positions' next tokens are already known,
  so the prefill LM head keeps one row per request (`get_last_indices`).
* **Logits → token.** Logits are unnormalised scores. Greedy = argmax (no softmax
  needed: softmax keeps the order). Otherwise: divide by temperature → keep top-k
  → keep the smallest top-p nucleus → renormalise → draw.
* **Prefill vs decode** are two phases of one loop: prefill processes the known
  context in parallel (compute-bound), decode one new token per step
  (memory-bandwidth-bound). Stop on EOS, `max_tokens`, or the context limit.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| whole model | `python/minisgl/models/qwen3.py:Qwen3ForCausalLM.forward` | reads `get_global_ctx().batch.input_ids` |
| embedding → layers → final norm | `models/qwen3.py:Qwen3Model.forward` | `residual` starts as `None` |
| decoder layer | `models/qwen3.py:Qwen3DecoderLayer.forward` | 4 calls, no explicit `+` |
| fused residual + RMSNorm | `python/minisgl/layers/norm.py:RMSNormFused.forward`, `rms_norm` | computes in fp32 |
| QKV, QK-norm, RoPE, attention, o_proj | `python/minisgl/models/utils.py:RopeAttn.forward` | split sizes `[Hq·D, Hkv·D, Hkv·D]` |
| fused QKV weight | `python/minisgl/layers/linear.py:QKVParallelLinear` | |
| RoPE | `python/minisgl/layers/rotary.py:RotaryEmbedding.forward` | NeoX style (rotate halves), cos/sin cache by position |
| attention + causal mask | `python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_metadata` (`visible = k <= cached + q`), `TorchAttnBackend.forward` (one `scaled_dot_product_attention(..., enable_gqa=True)`) | |
| gated MLP | `models/utils.py:GatedMLP.forward` | `down(silu(gate) * up)` with gate/up fused |
| last-position selection | `python/minisgl/layers/embedding.py:ParallelLMHead.forward`, `attention/torch_backend.py:TorchAttnMetadata.get_last_indices` | skipped when `batch.return_all_logits` |
| run + sample | `python/minisgl/engine/engine.py:Engine.forward_batch` | also advances `cached_len/device_len` (`Req.complete_one`) |
| sampling | `python/minisgl/engine/sample.py:Sampler.prepare`, `Sampler.sample`, `sample_impl` | |
| stop conditions | `python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data`; context limit: `Scheduler._process_one_msg` clamps `output_len` to `max_seq_len - input_len` | |
| §4.2 no-cache loop | `python/minisgl/llm/naive.py:generate_without_kv_cache` | `cached_len=0` every step |
| all-position logits | `python/minisgl/speculative/runner.py:SequenceRunner.extend(tokens, all_logits=True)` | sets `Batch.return_all_logits` |

## This repo vs. the tutorial's reference code

* **No `layers/attention.py:AttentionLayer`.** The Q/K/V split, QK-norm and RoPE
  that the reference does in `AttentionLayer.forward` are inline in
  `RopeAttn.forward`; the backend call is `ctx.attn_backend.forward(q, k, v,
  layer_id, batch)`.
* **One pure-PyTorch backend** (`TorchAttnBackend`): K/V gathered through the
  page table, one padded SDPA call with an explicit boolean mask; the reference
  uses FlashAttention/FlashInfer kernels with ragged metadata.
* **`GatedMLP.forward`** is `gate, up = gate_up_proj(x).chunk(2)` and
  `down_proj(F.silu(gate) * up)`; no separate `act_fn` module.
* `Qwen3Model` iterates a plain `nn.ModuleList` (reference: `self.layers.op_list`)
  and the layers are called as modules, so standard forward hooks work (the lab
  relies on this).
* **Sampling** is `torch.sort` + `cumsum` + `torch.multinomial` with a seeded
  `torch.Generator` instead of `flashinfer.sampling`; greedy rows inside a mixed
  batch use `torch.where(greedy, argmax, sampled)`. Top-k is applied first and the
  top-p nucleus is computed on the top-k-renormalised distribution.
* `Batch.return_all_logits` exists here (for speculative verification); the
  tutorial's description of the reference LM head has no such switch (prefill
  always selects `get_last_indices`).

## Common misconceptions

**Misconception:** Hidden states are `[B, T, C]` in the engine.
**Actually:** `[N, C]` with `N = Σ extend_len` (lab: lengths 5 and 3 → `(8, 64)`).
`[B, T, C]` is only the teaching layout.

**Misconception:** `Qwen3DecoderLayer` forgot the residual connection.
**Actually:** It is in `RMSNormFused`; the lab verifies the returned residual of
`L0.post_attention_layernorm` equals `embedding + self_attn_out`.

**Misconception:** The causal mask is only needed during training.
**Actually:** Prefill computes all prompt positions in one pass and needs it just
as much; without it earlier positions read the future (lab section 4) and their
cached K/V would depend on tokens that a different continuation would change.

**Misconception:** GQA shrinks the attention scores.
**Actually:** Scores keep `Hq` heads (`[1,4,3,3]` for Hkv 2 and 1); only K/V, the
QKV projection width and the KV cache shrink.

**Misconception:** Temperature 0 is a division by zero.
**Actually:** `SamplingParams.is_greedy` routes `temperature <= 0` to argmax;
inside `sample_impl` greedy rows get temperature 1 for the (unused) division.

**Misconception:** The sampler stops generation at EOS.
**Actually:** The sampler only picks a token; `_process_last_data` decides
whether the request is finished.

## Lab

```bash
python labs/part2_ch03_forward_shapes.py     # ~4 s on the tiny model; --model real for Qwen3-0.6B
```

What to look for (tiny model: C=64, Hq=4, Hkv=2, D=16, V=512):

```
|                    embedding |    (8, 64) |
|          L0 qkv_proj (fused) |   (8, 128) |      (4+2·2)·16 = 128
|       q after QK-norm + RoPE | (8, 4, 16) |
|       k after QK-norm + RoPE | (8, 2, 16) |
|                  v (no RoPE) | (8, 2, 16) |
|        attention backend out | (8, 4, 16) |
|          L0 mlp.gate_up_proj |   (8, 256) |
|                   final norm |    (8, 64) |
|             lm_head (logits) |   (2, 512) |
get_last_indices() = [4, 7]   (cumsum(extend_lens) - 1 = cumsum([5, 3]) - 1)
all_logits[last_indices] == default logits: True
```

Section 4 (causal check; the manual re-implementation matches the engine first):

```
| position |         | max|logit A - B| engine (causal) | manual, NO mask |
|        0 |         |                         0.00e+00 |        3.65e+00 |
|        2 |         |                         0.00e+00 |        4.51e+00 |
|        3 | changed |                         4.13e+00 |        4.43e+00 |
```

Section 6 prints empirical token frequencies for greedy, T=0.5/1/2, top-k=2 and
top-p=0.8 on fixed logits `[3.0, 2.2, 1.9, 1.0, ...]`.

## Exercises

### Concept chapter

**1. The conceptual path of a token from input to the next sampled token.**
Text → tokenizer → ids (`TokenizeManager.tokenize_one`) → embedding row per id
(`[N, C]`) → for each layer: RMSNorm → Q/K/V projections, QK-norm, RoPE by
position → causal multi-head (GQA) attention → `o_proj` + residual → RMSNorm →
gated MLP + residual → final RMSNorm → LM head on the last position of each
request → logits `[B, V]` → (temperature, top-k, top-p, softmax) → sampled id →
appended to the sequence; the loop repeats with that token as the next input.

**2. Logits vs probabilities; why sample; effect of temperature, top-k, top-p.**
Logits are unnormalised real scores (any sign, any sum); softmax turns them into
non-negative values summing to 1. Always taking the max is deterministic and
tends to loop or be bland; open-ended text has many acceptable continuations, so
sampling draws from the distribution. Temperature `T` rescales logits before
softmax: `T < 1` sharpens, `T > 1` flattens, `T → 0` is greedy. Top-k keeps a
fixed number of best tokens; top-p keeps the smallest set whose mass reaches `p`
(its size adapts to how peaked the distribution is). Lab section 6:

```
softmax = [0.502, 0.225, 0.167, 0.068, 0.025, 0.009, 0.003, 0.0]
|         policy | tok0 | tok1 | tok2 | tok3 | ... | support |
|   greedy (T=0) | 1.00 | 0.00 | 0.00 | 0.00 | ... |       1 |
|          T=0.5 | 0.74 | 0.15 | 0.09 | 0.01 | ... |       6 |
|          T=2.0 | 0.32 | 0.22 | 0.18 | 0.12 | ... |       8 |
|   T=1, top_k=2 | 0.69 | 0.31 | 0.00 | 0.00 | ... |       2 |
| T=1, top_p=0.8 | 0.56 | 0.26 | 0.18 | 0.00 | ... |       3 |
```

(0.502 + 0.225 = 0.727 < 0.8, so the nucleus needs a third token.)

**3. Prefill vs decode; why compute-bound vs memory-bandwidth-bound?**
Prefill feeds all `L` known tokens at once, builds their K/V and yields the
first token (dominates TTFT); decode feeds 1 token per request per step and
repeats until a stop (dominates TPOT). For a weight matrix of `C×C` parameters, a
step moves the same ~`2·C²` bytes (fp16) whatever the token count, but performs
`2·N·C²` FLOPs, so arithmetic intensity ≈ `N` FLOP/byte. Prefill has `N` in the
hundreds or thousands → above the GPU's ridge point → compute-bound. Decode has
`N = batch size` (often small) and additionally reads the whole KV history each
step → below the ridge → bandwidth-bound. (See `labs/part1_ch02_roofline.py`.)

### Code walkthrough

**1. Why does the prefill LM head not project every input position?**
Position `t`'s logits predict token `t+1`, and for every position but the last
that token is already known (it is the next prompt token). Only the last
position yields a new token, so `ParallelLMHead.forward` does
`x = x[batch.attn_metadata.get_last_indices()]` before `F.linear`. This turns a
`[N, C] × [C, V]` product into `[B, C] × [C, V]` (lab: 8 → 2 rows; for a real
prompt of 2 000 tokens and V = 151 936 it avoids a 2 000 × 151 936 logits tensor).
Exceptions that need every row (`return_all_logits=True`): speculative
verification, and the lab's equality check.

**2. Where does the residual addition happen?**
In `layers/norm.py:RMSNormFused.forward`: `if residual is not None: x = x + residual`,
then it returns `(rms_norm(x), x)`. The attention residual is added by the same
layer's `post_attention_layernorm`; the MLP residual by the next layer's
`input_layernorm`, or by `Qwen3Model.norm` after the last layer (`[0]` keeps the
normalised output). The first `input_layernorm` gets `residual=None` and just
saves the embedding. Lab section 3 verifies all three cases.

**3. Which shapes in §5 change if `Hkv` goes from 2 to 1?**
Only K and V: `[1,2,3,8] → [1,1,3,8]`. Input `[1,3]`, embedding `[1,3,32]`, Q
`[1,4,3,8]`, scores `[1,4,3,3]` (each K/V head is repeated `Hq/Hkv` times), and
logits `[1,3,64]` / `[1,64]` are unchanged. In the engine the fused QKV output
width, the k/v tensors and the KV pool shrink (lab section 5, real tiny model with
`num_key_value_heads=1`):

```
|                  stage |                 Hkv=2 |                 Hkv=1 |
|    L0 qkv_proj (fused) |              (8, 128) |               (8, 96) |
| k after QK-norm + RoPE |            (8, 2, 16) |            (8, 1, 16) |
|            v (no RoPE) |            (8, 2, 16) |            (8, 1, 16) |
|  attention backend out |            (8, 4, 16) |            (8, 4, 16) |
|                KV pool | (2, 3, 256, 1, 2, 16) | (2, 3, 256, 1, 1, 16) |
```

**4. Without the causal mask, which output positions are affected by a change to future tokens?**
All of them: every position attends to every key, so changing token `j` changes
positions `0..j-1` too (in the first layer already, and it propagates). With the
mask only positions `≥ j` change. Lab section 4 shows engine logits at positions
0–2 bit-identical (difference `0.00e+00`) when token 3 changes, and differences
of 3.6–4.5 at positions 0–2 in the unmasked re-implementation. (A consequence
for this engine: without the mask, cached K/V of earlier tokens would be stale
as soon as a new token arrives, so KV caching would be wrong.)

**5. What does generation without caching recompute, and what could be saved?**
Each step re-runs the whole prefix: embedding, every layer's Q/K/V projections,
attention and MLP for all `L + t` positions, to use only the last position's
logits (`llm/naive.py:generate_without_kv_cache` sets `cached_len=0` each step).
Over `n` new tokens that is `Σ (L + t)` positions = O(n²). Because of the causal
mask, the K and V of earlier positions never change, so they can be stored once
and reused; then each decode step computes one new position (and only attention
still reads all keys). Measured in `python labs/part2_ch04_kv_cache.py` section 5
(prompt 6, 16 new tokens): 216 positions without a cache vs 21 with it, identical
tokens. That is Chapter 4.
