"""Speculative decoding in the scheduler: batched serving with and without it.

Same code path as the HTTP server (``LLM`` drives the continuous-batching
``Scheduler``), so this measures what a served batch gets from speculation:

* output throughput of the whole batch (tokens / s)
* per-request TPOT (time between a request's first and last token / tokens)
* acceptance length: tokens committed per request per verify step
  (SGLang's ``spec_accept_length``; 1.0 = no gain)

Workloads: *copy-heavy* (repeat / quote a paragraph: n-gram lookup hits) and
*open-ended* (answers that do not copy the prompt).

    python benchmarks/bench_spec_serving.py --model /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

from _common import add_engine_args, describe, engine_kwargs
from minisgl import SamplingParams
from minisgl.llm import LLM
from minisgl.tokenizer import TokenizeManager, load_tokenizer

PARAGRAPHS = [
    "The scheduler decides which requests run in each step. The engine runs the model and the sampler "
    "picks the next token. The KV cache stores keys and values so that decode steps only process one "
    "new token per request.",
    "A radix tree maps token prefixes to KV cache pages. When a new request shares a prefix with an "
    "earlier one, the matched pages are reused and only the remaining tokens are computed.",
    "Chunked prefill splits a long prompt into pieces so that decode requests are not stalled behind it. "
    "Each piece is computed in its own step and the request only samples after its last chunk.",
    "Tensor parallelism shards every weight matrix across devices. Attention heads are split between "
    "ranks and an all-reduce combines the partial results after each block.",
    "Speculative decoding proposes several tokens cheaply and verifies them with one forward pass of "
    "the target model. Accepted tokens are kept and the first rejected one is replaced.",
    "Paged attention stores the KV cache in fixed-size pages. A page table maps each request's token "
    "positions to physical slots, so memory is allocated on demand without fragmentation.",
    "Continuous batching lets requests join and leave the running batch between steps. A finished "
    "request frees its slot immediately and a waiting request takes it in the next step.",
    "CUDA graphs record a fixed sequence of kernel launches once and replay it later, which removes "
    "the CPU launch overhead that dominates small decode batches on a GPU.",
]
QUESTIONS = [
    "List the first ten prime numbers and explain what a prime number is.",
    "Write a short story about a robot who learns to paint.",
    "What are three good habits for staying healthy? Explain each briefly.",
    "Describe how a bicycle works to a ten year old.",
    "Give me a recipe for a simple tomato soup.",
    "Why is the sky blue? Answer in a few sentences.",
    "Suggest five names for a coffee shop and explain one of them.",
    "What is the difference between weather and climate?",
]


def workloads(n: int) -> Dict[str, List[List[Dict[str, str]]]]:
    copy = [f"Repeat the following paragraph exactly, word for word:\n\n{PARAGRAPHS[i % 8]}" for i in range(n)]
    open_ = [QUESTIONS[i % 8] for i in range(n)]
    return {
        "copy-heavy": [[{"role": "user", "content": c}] for c in copy],
        "open-ended": [[{"role": "user", "content": q}] for q in open_],
    }


def run(llm: LLM, prompts: List[List[int]], sp: SamplingParams):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0
    tokens = sum(len(o.token_ids) for o in outs)
    tpots = [
        (o.token_times[-1] - o.token_times[0]) / (len(o.token_times) - 1)
        for o in outs
        if len(o.token_times) > 1
    ]
    return outs, tokens / wall, statistics.mean(tpots), wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-sizes", default="1,8")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--draft-layers", default="8", help="draft_model rows: truncated target, comma list")
    ap.add_argument("--self-draft", action="store_true", help="also run the target as its own draft")
    add_engine_args(ap)
    args = ap.parse_args()
    print(describe(args))

    tok = TokenizeManager(load_tokenizer(args.model))
    sp = SamplingParams(max_tokens=args.max_tokens, ignore_eos=True)
    configs = [("none", {}), (f"ngram k={args.k}", dict(speculative_algorithm="ngram"))]
    for n in (int(x) for x in args.draft_layers.split(",") if x):
        configs.append((f"draft {n}L k={args.k}", dict(speculative_algorithm="draft_model",
                                                        speculative_draft_num_layers=n)))
    if args.self_draft:
        configs.append((f"draft self k={args.k}", dict(speculative_algorithm="draft_model")))

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    max_bs = max(batch_sizes)
    data = {
        name: [tok.tokenize_one(m, {"enable_thinking": False}) for m in msgs]
        for name, msgs in workloads(max_bs).items()
    }
    print("| workload | batch | config | tok/s | TPOT ms | accept len | acceptance | verify steps | lossless |")
    print("|---|---:|---|---:|---:|---:|---:|---:|:---:|")
    for cname, ckw in configs:
        llm = LLM(args.model, load_tokenizer_=False, max_total_tokens=4096, max_seq_len=1024,
                  speculative_num_draft_tokens=args.k, **ckw, **engine_kwargs(args))
        llm.generate([data["open-ended"][0]], SamplingParams(max_tokens=4))  # warm-up
        for wname, prompts in data.items():
            for bs in batch_sizes:
                spec = llm.scheduler.spec
                before = (spec.stats.committed, spec.stats.req_rounds, spec.stats.accepted,
                          spec.stats.proposed, spec.stats.verify_steps) if spec else None
                outs, tps, tpot, _ = run(llm, prompts[:bs], sp)
                key = (wname, bs)
                if cname == "none":
                    REF[key] = [o.token_ids for o in outs]
                lossless = [o.token_ids for o in outs] == REF.get(key)
                if spec:
                    st = spec.stats
                    c, r, a, p, v = (st.committed - before[0], st.req_rounds - before[1], st.accepted - before[2],
                                     st.proposed - before[3], st.verify_steps - before[4])
                    acc = f"{c / max(r, 1):.2f} | {a / max(p, 1):.0%} | {v}"
                else:
                    acc = "1.00 | - | -"
                print(f"| {wname} | {bs} | {cname} | {tps:.1f} | {tpot * 1000:.0f} | {acc} | {lossless} |",
                      flush=True)
        llm.shutdown()


REF: Dict = {}

if __name__ == "__main__":
    main()
