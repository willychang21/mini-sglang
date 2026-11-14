# Part III · Ch2 — Quantization and Low-Precision Inference

Course text: `part3/Chapter2_Quantization-and-Low-Precision-Inference.md`. Labs:
[`labs/part3_ch02_quant_error.py`](../../labs/part3_ch02_quant_error.py) and
[`labs/part3_ch02_kv_capacity.py`](../../labs/part3_ch02_kv_capacity.py).

## Key ideas

- Quantization changes the *representation*, not the architecture: x ≈ q · s, with the
  scale s shared by a group of numbers. The number of bits and the size of the group
  determine the rounding error. Outliers inflate a group's scale and wipe out the small
  values in it.
- **Two independent wins.** (1) Fewer weight bytes per decode step: small-batch decode is
  bounded by `weight_bytes / HBM bandwidth`. (2) Freed memory goes to the KV pool, which
  allows a bigger batch and higher throughput. KV-cache precision (`--kv-cache-dtype`) is a
  separate knob from weight precision.
- **W4A16 vs W8A8.** W4A16 (AWQ/GPTQ) dequantizes the weights on chip and multiplies in
  BF16, so it helps memory-bound decode but not compute-bound prefill. W8A8-FP8 also
  quantizes activations and runs on FP8 tensor cores, so it keeps part of its gain in prefill.
- Accuracy loss is task-dependent (GSM8K barely moves while AIME drops, especially with FP4
  KV). Fix the eval conditions, repeat runs, and report speed and accuracy together.

## What mini-sglang shows, and can't

mini-sglang has no quantized kernels, but its weights and KV pool are plain tensors, so we
can **fake-quantize** them (quantize → dequantize to fp32). This isolates rounding error
from kernel speed. `labs/part3_ch02_quant_error.py` does this on the real Qwen3-0.6B weights.

Weight-level relative RMS error (and the share of weights rounded to 0):

| scheme | bits/weight incl. scales | `layers.14.mlp.down_proj` |
|---|---:|---:|
| fp8-e4m3 per-channel | 8 | 0.0264 (0.0% →0) |
| int8 per-channel | 8 | 0.0111 (1.8% →0) |
| int4 per-channel | 4 | 0.2013 (30.1% →0) |
| int4 group-128 (AWQ/GPTQ-like) | 4.12 | 0.1309 (19.2% →0) |
| fp4-e2m1 group-16 (NVFP4-like) | 4.5 | 0.0996 (7.6% →0) |
| int2 group-128 | 2.12 | 0.7862 (87.4% →0) |

End to end: all 112 decoder linear weights quantized (4 merged projections × 28 layers), with 4 prompts × 32 greedy tokens
compared against fp32. The lab uses the `naive` prefix cache on purpose. With the radix
cache, the second scheme would reuse KV computed under the *previous* weights.

| scheme | greedy tokens identical to fp32 | next-token KL | top-1 agree |
|---|---:|---:|---:|
| bf16 | 128/128 | 0.0000 | 100% |
| int8 per-channel | 128/128 | 0.0055 | 100% |
| fp8-e4m3 per-channel | 84/128 | 0.0095 | 100% |
| int4 group-128 | 70/128 | 0.3379 | 75% |
| fp4-e2m1 group-16 | 68/128 | 0.0939 | 75% |
| int4 per-channel | 18/128 | 0.9579 | 75% |
| int2 group-128 | 0/128 | 18.89 | 0% |

Things to notice:

1. **Group size matters as much as bits.** Moving INT4 from per-channel (a group of 1024)
   to group-128 cuts error by 35%. NVFP4's group of 16 with an FP8 scale beats both, even
   though FP4 has only 15 distinct values. The outlier demo in the lab reproduces Figure 6:
   one 4.0 in a group sends all 7 small weights to 0.
2. **FP8-E4M3 has a larger RMS error than INT8 on weights.** It has only 3 mantissa bits
   versus INT8's uniform 256 levels over a tight range. FP8's advantage is dynamic range
   (for activations and outliers) and native tensor-core support, not weight fidelity.
3. **Greedy agreement is a harsh, noisy metric.** One flipped token changes everything after
   it. FP8 has 100% top-1 agreement on the next token but only 66% identical continuations.
   This is the chapter's point: you need task evals with repeats, not eyeballing outputs.

KV capacity (`labs/part3_ch02_kv_capacity.py`) reproduces Figure 5 and checks the formula
against the bytes mini-sglang's `MHAKVCache` actually allocates (114,688 B/token for
Qwen3-0.6B in bf16, matching `2·28·8·128·2`):

| Qwen3-32B on 96 GB × 0.85 | weights GB | KV tokens | 1,536-token requests |
|---|---:|---:|---:|
| BF16 / bf16 KV | 65.5 | 61K | 39 |
| FP8 / bf16 KV | 34.4 | 180K | 117 |
| AWQ-INT4 / bf16 KV | 19.7 | 236K | 153 |
| AWQ-INT4 / fp8 KV | 19.7 | 472K | 307 |

For Qwen3-8B, going from BF16 to INT4 grows capacity by only **16%** (as the chapter says):
the weights were a small share of memory to begin with.

To add real KV quantization to mini-sglang, `MHAKVCache` would store `float8_e4m3fn` plus
per-head or per-token scales, `store_kv` would quantize, and the attention backend would
dequantize after the gather. SGLang does exactly this in `MHATokenToKVPool.set_kv_buffer`
with `k_scale`/`v_scale`.

## SGLang pointers

- Registry: `srt/layers/quantization/__init__.py::get_quantization_config`, `BASE_QUANTIZATION_METHODS`.
- Schemes: `srt/layers/quantization/fp8.py`, `awq/`, `gptq/`, `modelopt_quant.py` (NVFP4), `mxfp4.py`, `w8a8_int8.py`.
- KV: `srt/layers/quantization/kv_cache.py`, `srt/mem_cache/kv_cache_dtype.py`, and `MHATokenToKVPoolFP4` in `srt/mem_cache/memory_pool.py`.
- How a scheme hooks into layers: `LinearBase.quant_method` in `srt/layers/linear.py`. Then read
  `docs/docs/developer_guide/quantization_contribution_guide.mdx` in the SGLang repo.

## Exercises — worked answers

**1. What W4A16, W8A8-FP8 and FP8 KV each save and speed up.**
- *W4A16*: about 4× smaller linear weights (≈3.7× with group-128 scales). This saves weight
  memory and weight reads per step, so memory-bound decode at small batch gets faster. Compute is still
  BF16 after on-chip dequant, so prefill barely improves. The freed memory goes to the KV pool.
- *W8A8-FP8*: 2× smaller weights *and* FP8 activations, so the matmul runs on FP8
  tensor cores (Ada/Hopper/Blackwell). That speeds up both decode (bytes) and prefill (FLOP/s).
- *FP8 KV cache*: halves KV bytes per token. That doubles the tokens that fit in the pool
  (larger batches, longer contexts) and halves the KV bytes attention reads each decode
  step, which helps long-context decode. Weights are untouched.

**2. Why an 8B INT4 model is 6.1 GB, not 4 GB.**
(a) Embedding and lm_head stay 16-bit: ~1.24B params ≈ 2.49 GB. (b) Quantization metadata:
AWQ stores a scale and zero point per 128 weights (≈0.14 GB). (c) Norms and other small
tensors stay 16-bit. `(8.19 − 1.24)B × 0.5 B + 2.49 GB + 0.14 GB ≈ 6.1 GB`.

**3. Why W4A16 helps single-request decode but not long-input prefill.**
At batch 1, decode reads every weight once to produce one token, so time ≈ weight bytes /
bandwidth. Quartering the weight bytes cuts that lower bound almost proportionally (the
Qwen benchmark measured 1.76×). Long prefill multiplies each weight by thousands of tokens.
It is compute-bound, and W4A16 still computes in BF16, plus dequant overhead. Its
attention cost over a long sequence is also untouched. Hence 1.02× at 129K input.

**4. Why KV quantization matters more as conversations get longer.**
KV memory per request grows linearly with context, while weight memory is fixed. In long
conversations, the KV pool (not the weights) limits how many requests fit and how many bytes
each decode step reads (attention reads all history). Halving bytes per token roughly doubles
concurrent long requests and halves attention read traffic. For short chats the KV share is
small and the gain is too. (Accuracy also needs re-checking on long contexts specifically:
quantized KV is re-read by every later token.)

**5. GSM8K drops only 0.3 points. Is the effect nil?**
No. First, 0.3 points may be within run-to-run noise. Repeat both baseline and quantized runs
and compare means and ranges. Second, GSM8K is easy and short. Check harder and longer tasks
(AIME, GPQA, LiveBench, code), long-context tasks (especially with a quantized KV cache),
thinking mode vs non-thinking, and the actual deployment workload. The chapter's table shows
FP4 KV unchanged on GSM8K but −40 points on AIME25. Also keep the evaluation conditions
identical (dataset version, few-shot count, template, sampling params) and report speed
alongside.

**6. Identify the schemes and confirm them from config files.**
Verified by downloading each repo's config on 2026-11-14:

- `Qwen/Qwen3-32B-AWQ`: **W4A16 INT4 (AWQ)**. `config.json → quantization_config` =
  `{"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": true, "version": "gemm"}`.
  SGLang switches to `awq_marlin` automatically on compatible GPUs.
- `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`: **W8A8-FP8, dynamic per-token activations**.
  `quant_method: compressed-tensors`, `format: float-quantized`. Weights:
  `num_bits: 8, type: float, strategy: channel, dynamic: false` (static per-channel scales).
  Input activations: `num_bits: 8, type: float, strategy: token, dynamic: true`.
  `ignore: ["lm_head"]`, and `kv_cache_scheme: null`, so the KV cache is *not* quantized.
- `nvidia/DeepSeek-R1-0528-NVFP4`: **NVFP4 (W4A4)**. This one has no `quantization_config` in
  `config.json`; the evidence is in `hf_quant_config.json` (ModelOpt):
  `quant_algo: NVFP4`, `group_size: 16`, `kv_cache_quant_algo: FP8`, `pre_quant_scale: true`,
  and `exclude_modules` lists `lm_head` and **every layer's `self_attn`**. So only the MLP/MoE
  weights are FP4, attention projections stay BF16, and the KV cache is FP8. It is native on
  Blackwell; SGLang falls back to W4A16-style compute on older GPUs.
- `openai/gpt-oss-20b`: no suffix, but `quantization_config` =
  `{"quant_method": "mxfp4", "modules_to_not_convert": ["model.layers.*.self_attn",
  "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]}`. So only the MoE expert
  weights are MXFP4.

General rule: the name is a hint, the config is the evidence. Note how much the configs add:
the KV dtype (FP8 for the NVFP4 model, untouched for the RedHat one) and which modules are
excluded are not visible from the name at all. Command used:
`hf_hub_download(repo, "config.json")` / `"hf_quant_config.json"`.
