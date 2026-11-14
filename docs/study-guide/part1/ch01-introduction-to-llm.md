# Part I · Chapter 1 — Introduction to LLM

> Tutorial: `part1/Chapter1_Introduction_to_LLM.md` · Lab: none (the forward pass is made
> observable in Part II Ch3: `labs/part2_ch03_forward_shapes.py`) · Related code:
> `python/minisgl/models/qwen3.py`, `models/utils.py`, `layers/`, `engine/sample.py`

## Key ideas

- **An LLM is a next-token predictor.** Given tokens `t_1..t_{i-1}`, it outputs a
  probability distribution over the vocabulary for `t_i`. All of the "capabilities"
  come from doing this well at very large scale: billions of parameters trained on
  trillions of tokens. The same mechanism explains the limits: it can hallucinate
  (fluent but wrong text), its knowledge stops at the training cutoff, and it has a
  finite context window.
- **Why inference matters (Sec. 3.4).** Every application is a stream of forward
  passes. Two numbers decide whether serving is viable: *cost* and *latency*. Cost
  per token is dominated by GPU memory, and most of that memory is the KV cache at
  high concurrency. Latency is TTFT and TPOT; the chapter cites ≤150 ms/token as the
  reading-speed target. Part I Ch2–5 and all of Part II are about these two numbers.
- **Transformer building blocks**, in the original 2017 form and in what Qwen3 (and so
  mini-sglang) actually uses:

| Block | Original Transformer (taught here) | Qwen3 / mini-sglang | Why it matters for inference |
|---|---|---|---|
| Architecture | encoder–decoder | decoder-only (`Qwen3ForCausalLM`) | one causal stack, so tokens are generated one at a time |
| Position | sinusoidal PE **added** to the embedding | **RoPE**: rotates q and k by a position-dependent angle | positions are per token; the cache stores K after RoPE |
| Attention | MHA, `h` heads, `d_k = d_model/h` | **GQA** (16 Q heads share 8 KV heads in Qwen3-0.6B) + per-head QK-RMSNorm | fewer KV heads means a smaller KV cache (Ch4) |
| Normalization | LayerNorm, Post-Norm `LN(x + F(x))` | **RMSNorm**, **Pre-Norm** `x + F(RMSNorm(x))` | Pre-Norm leaves a clean residual path; RMSNorm is cheaper |
| FFN | `ReLU(xW1+b1)W2+b2`, `d_ff = 4·d_model` | **SwiGLU** `down(silu(gate(x)) * up(x))`, no bias | three weight matrices, so most of the parameters |
| Output | LM head + softmax | LM head **tied** to the embedding; only the last position of each prompt | Part II Ch3 |

- **Sinusoidal PE in one line.** Dimension pair `i` uses frequency
  `ω_i = 10000^{-2i/d_model}` and stores `(sin(pos·ω_i), cos(pos·ω_i))`. Low `i` means
  high frequency (fine local position). High `i` means low frequency (long range).
  Because `PE(pos+k)` is a fixed rotation of `PE(pos)` for each pair, relative offsets
  are linear functions of the encoding. **RoPE keeps exactly this frequency schedule**
  (`inv_freq = base^{-2i/d}` in `layers/rotary.py`, with `base = rope_theta = 1e6` for
  Qwen3). It applies the rotation to q and k instead of adding a vector to x. As a
  result, the score `q·k` depends only on the relative position.
- **Multi-head attention shapes:** `Q,K,V: [B, S, d_model] → [B, h, S, d_k]`; per
  head `softmax(QKᵀ/√d_k)V`; concatenate the heads; `W^O`. Splitting into `h` heads
  keeps the FLOPs about the same as one big head but learns `h` subspaces. In
  mini-sglang the batch is *packed*: `[N, C]` with all tokens of all requests
  concatenated, not `[B, S, C]`.
- **Residual and norm:** the residual (`out = x + F(x)`) gives gradients and
  information an identity path; at worst a layer learns `F ≈ 0`. Normalization keeps
  the scale of the activations stable across depth.
- **Vocabulary** used later: token (an integer ID from the tokenizer), embedding,
  parameter count (1 parameter = 1 number; 2 bytes in bf16), autoregressive
  generation, context window. The newer terms (prompt and context engineering, RAG,
  skills, agents, harness) describe what sits *in* the context. For an inference
  engine they all mean the same thing: longer prompts and repeated prefixes. That is
  why Ch2's agent/multi-turn analysis and Part II Ch8's prefix cache exist.

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| Tokenizer (text → IDs) | `python/minisgl/tokenizer/tokenize.py:TokenizeManager.tokenize_one` | chat template applied first for message lists |
| Embedding | `python/minisgl/layers/embedding.py:VocabParallelEmbedding.forward` | `[N] → [N, C]` |
| Decoder-only stack | `python/minisgl/models/qwen3.py:Qwen3Model.forward` | loop over `Qwen3DecoderLayer` |
| Pre-Norm + residual | `python/minisgl/models/qwen3.py:Qwen3DecoderLayer.forward`, `layers/norm.py:RMSNormFused.forward` | the `x + residual` add is deferred into the next norm |
| RMSNorm | `python/minisgl/layers/norm.py:rms_norm` | `x · rsqrt(mean(x²) + eps) · w`, no mean subtraction, no bias |
| Positional encoding (RoPE) | `python/minisgl/layers/rotary.py:RotaryEmbedding` | `inv_freq = 1/base^(2i/d)`, same schedule as sinusoidal PE |
| Multi-head / GQA attention | `python/minisgl/models/utils.py:RopeAttn.forward` → `attention/torch_backend.py:TorchAttnBackend.forward` | `scaled_dot_product_attention(..., enable_gqa=True)` with a causal mask |
| FFN | `python/minisgl/models/utils.py:GatedMLP` | SwiGLU, gate and up fused into one matmul |
| Next-token prediction | `python/minisgl/layers/embedding.py:ParallelLMHead.forward`, `engine/sample.py:Sampler.sample` | prefill keeps only the last position per request; greedy is `argmax` |
| Autoregressive loop | `python/minisgl/llm/naive.py:generate_without_kv_cache` | the textbook loop (re-runs the whole sequence every step) |
| Context window | `python/minisgl/config.py:ModelConfig.max_position_embeddings`, `engine/engine.py:Engine.__init__` (`max_seq_len`) | 40,960 for Qwen3-0.6B; capped further by the KV pool |

## Common misconceptions

**Misconception:** Modern LLMs use the Transformer exactly as in the 2017 paper.
**Actually:** Almost every block in the table above has been replaced (decoder-only,
RoPE, GQA, RMSNorm, Pre-Norm, SwiGLU). The chapter teaches the classic form and lists
the replacements in Sec. 4.5. mini-sglang implements the modern ones.

**Misconception:** Positional encoding is added to the token embedding in Qwen3.
**Actually:** RoPE is applied inside every attention layer, to q and k only, after the
QK-norm (`RopeAttn.forward`). V is not rotated, and nothing is added to the residual
stream.

**Misconception:** The model "understands" and therefore cannot be confidently wrong.
**Actually:** It samples from a learned distribution, so a fluent and wrong
continuation can have high probability (hallucination).

**Misconception:** Multi-head attention costs `h` times as much as single-head.
**Actually:** Each head works in `d_model/h` dimensions, so the projection FLOPs are
the same. Only the score matrices are per head.

**Misconception:** "600B parameters" is a compute number.
**Actually:** It is a count of numbers. Memory is params × bytes per param (2 in
bf16), and per-token compute is about 2 × params FLOPs (Ch2).

## Lab

There is no Part I lab for this chapter. To *see* these blocks, run the Part II
forward-pass lab (owned by the Part II notes):

```bash
python labs/part2_ch03_forward_shapes.py      # packed [N, C] shapes through every block
```

A quick check you can do now: RoPE uses the sinusoidal frequency schedule.

```bash
python -c "
import torch; from minisgl.layers.rotary import RotaryEmbedding
r = RotaryEmbedding(head_dim=8, max_position=4, base=10000.0)
print(r.cos_cache[1]); print(torch.cos(1 / 10000 ** (torch.arange(0, 8, 2) / 8)))"
```

```text
tensor([0.5403, 0.9950, 0.9999, 1.0000])
tensor([0.5403, 0.9950, 0.9999, 1.0000])
```

## Exercises

**1. State the core capability of an LLM in one sentence, and explain why "predict the
next token" leads to such rich capabilities.**

An LLM estimates `P(t_i | t_1..t_{i-1})`, a distribution over the next token given
everything before it. That objective is very general. To lower the prediction loss on
web text, code, math and dialogue, the model has to model grammar, facts, the
structure of arguments, code semantics and how to follow instructions, because all of
these make the next token more predictable. With enough parameters, data and compute
(scaling), these internal models become good enough to show up as abilities in their
own right: translation, reasoning, coding, few-shot learning. Autoregressive sampling
then turns one-step prediction into long outputs. The same fact also explains the
limits: the output is the *likely* text, not the *true* text, and it can only use
what is in the weights or the context window.

**2. Why does self-attention need positional encoding, and how does sinusoidal PE
supply it?**

Attention is permutation-equivariant. `softmax(QKᵀ)V` treats the inputs as a set: if
you shuffle the tokens, the outputs shuffle the same way and the scores do not change.
So "dog bites man" and "man bites dog" would look the same. (A causal mask gives a
little order information, but no notion of distance.) Sinusoidal PE adds a fixed
vector `PE(pos)` to each embedding. Dimension pair `i` holds
`sin(pos/10000^{2i/d})` and `cos(pos/10000^{2i/d})`, so each position gets a unique
multi-scale code. Low dimensions oscillate fast (adjacent tokens are easy to tell
apart) and high dimensions oscillate slowly (long distances). Because
`[sin(a+b), cos(a+b)]` is a rotation of `[sin a, cos a]` by `b`, `PE(pos+k)` is a
linear function of `PE(pos)` that depends only on `k`. This makes relative offsets
easy to learn. It needs no parameters and is bounded in [−1, 1]. RoPE (used by Qwen3
and in `layers/rotary.py`) keeps the same frequencies but applies the rotation to q
and k. Then `q_m·k_n` depends only on `m−n`.

**3. Describe the roles of residual connections and layer normalization.**

- *Residual connection* `y = x + F(x)`: gives an identity path through every layer.
  (a) Gradients flow straight back through the `+`, so deep stacks train. (b) A layer
  only has to learn a correction `F(x) = H(x) − x`, and `F ≈ 0` is easy, so adding
  depth cannot make things much worse. (c) Information from early layers reaches late
  layers ("residual stream").
- *Layer normalization* `γ·(x−μ)/σ + β` (per token, over the hidden dimension): resets
  the scale of each token's activation vector so values do not explode or vanish as
  depth grows. Each layer then sees inputs in a stable range, which speeds up and
  stabilizes training. γ and β let the model restore whatever scale it needs.
- Together: the residual keeps the signal and gradient paths open, and the norm keeps
  their magnitudes in check. The original paper uses Post-Norm `LN(x + F(x))`. Modern
  models (Qwen3, `Qwen3DecoderLayer.forward`) use Pre-Norm `x + F(Norm(x))` with
  RMSNorm (no mean subtraction, no β). The residual path is then never normalized,
  which trains more stably at depth.
