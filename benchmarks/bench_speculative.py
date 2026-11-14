"""Chapter 10: speculative decoding.

Target: the full model. Drafts:

* the same checkpoint truncated to its first N layers (shares the vocabulary
  by construction, but is untrained as a standalone model)
* n-gram prompt lookup (no model at all: nearly free proposals)
* the target itself (upper bound: every proposal is accepted)

Fewer target calls only pay off if drafting is cheap relative to the target,
which this benchmark makes measurable.

    python benchmarks/bench_speculative.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time

from minisgl import SamplingParams
from minisgl.config import EngineConfig
from minisgl.engine import Engine
from minisgl.speculative import NgramDrafter, SpeculativeDecoder, autoregressive_generate
from minisgl.tokenizer import TokenizeManager, load_tokenizer


TASKS = {
    "open-ended": "List the first ten prime numbers and explain what a prime number is.",
    "copy-heavy": (
        "Repeat the following paragraph exactly, word for word:\n\n"
        "The scheduler decides which requests run in each step. The engine runs the model "
        "and the sampler picks the next token. The KV cache stores keys and values so that "
        "decode steps only process one new token per request."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-layers", default="14")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = dict(device=args.device, max_total_tokens=2048, max_seq_len=1024)
    target = Engine(EngineConfig(args.model, **cfg))
    vocab = target.model_config.vocab_size
    tok = TokenizeManager(load_tokenizer(args.model))
    sp = SamplingParams(max_tokens=args.max_tokens, ignore_eos=True)
    drafts = [(f"model: first {n} layers", lambda n=int(n): Engine(EngineConfig(args.model, num_layers=n, **cfg)))
              for n in args.draft_layers.split(",") if n]
    drafts += [("n-gram prompt lookup", lambda: NgramDrafter(vocab)),
               ("model: self (upper bound)", lambda: Engine(EngineConfig(args.model, **cfg)))]

    print(f"| task | draft | k | acceptance | tokens / target fwd | target fwds | draft fwds | tok/s | speedup | lossless |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for task, text in TASKS.items():
        prompt = tok.tokenize_one([{"role": "user", "content": text}], {"enable_thinking": False})
        t0 = time.perf_counter()
        ref = autoregressive_generate(target, prompt, sp)
        base = time.perf_counter() - t0
        print(f"| {task} | none (target only) | - | - | 1.00 | {len(ref)} | 0 | {len(ref) / base:.1f} | 1.00x | - |")
        for name, make in drafts:
            spec = SpeculativeDecoder(target, make(), num_draft_tokens=args.k)
            out, st = spec.generate(prompt, sp)
            print(
                f"| {task} | {name} | {args.k} | {st.acceptance_rate:.0%} | {st.tokens_per_target_forward:.2f} "
                f"| {st.target_forwards} | {st.draft_forwards} | {len(out) / st.elapsed:.1f} "
                f"| {base / st.elapsed:.2f}x | {out == ref} |"
            )
            del spec


if __name__ == "__main__":
    main()
