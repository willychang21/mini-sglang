"""Chapters 6 & 7: continuous batching and paged KV memory.

1. Throughput: the same workload with ``max_running_req=1`` (one request at a
   time) vs continuous batching.
2. Concurrency under a fixed KV budget: paged allocation reserves only what each
   request needs; contiguous pre-allocation would reserve ``max_seq_len`` per
   request (Chapter 7, Figure 7-1).
3. Admission policy under a tight KV budget: conservative (reserve the whole
   ``max_tokens``) vs optimistic (reserve ``max_tokens * new_token_ratio`` and
   retract on KV exhaustion). Two workloads: exact lengths (``ignore_eos``) and
   chat prompts that stop at EOS long before ``max_tokens``.

    python benchmarks/bench_batching.py --model /path/to/Qwen3-0.6B
    python benchmarks/bench_batching.py --model /path/to/Qwen3-0.6B --sections admission
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from minisgl import SamplingParams
from minisgl.llm import LLM
from minisgl.tokenizer import TokenizeManager


def workload(n: int, seed: int = 0, prompt_len=(16, 96), output_len=(16, 64)):
    rng = random.Random(seed)
    prompts = [[rng.randrange(100, 10000) for _ in range(rng.randrange(*prompt_len))] for _ in range(n)]
    sps = [SamplingParams(max_tokens=rng.randrange(*output_len), ignore_eos=True) for _ in range(n)]
    return prompts, sps


def run(llm: LLM, prompts, sps):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sps)
    elapsed = time.perf_counter() - t0
    tokens = sum(len(o.token_ids) for o in outs)
    ttft = [o.ttft for o in outs]
    return outs, elapsed, tokens, ttft


def chat_workload(n: int, max_tokens: int):
    topics = ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Kenya", "Norway", "Peru", "Chile",
              "Spain", "Italy", "Mexico", "Vietnam", "Greece", "Turkey"]
    prompts = [
        [{"role": "user", "content": f"In one short sentence, what is the capital of {topics[i % len(topics)]}?"}]
        for i in range(n)
    ]
    return prompts, [SamplingParams(max_tokens=max_tokens) for _ in range(n)]


def admission(args) -> None:
    print("## Admission: conservative vs optimistic (retraction) under a tight KV budget\n")
    print("| workload | admission | KV tokens | wall s | output tok/s | peak batch | mean decode batch "
          "| retractions | identical |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    n = args.num_requests * 2
    cases = [
        ("exact lengths (ignore_eos)", *workload(n, seed=1), 1024, False),
        ("long exact outputs (ignore_eos)", *workload(n, seed=2, prompt_len=(16, 32), output_len=(96, 160)), 1024,
         False),
        ("chat, stops at EOS, max_tokens=256", *chat_workload(n, 256), 2048, True),
    ]
    for name, prompts, sps, budget, chat in cases:
        ref = None
        for adm in ("conservative", "optimistic"):
            llm = LLM(args.model, load_tokenizer_=chat, device=args.device, max_total_tokens=budget,
                      max_seq_len=1024, max_running_req=64, admission=adm)
            if chat:
                tm = TokenizeManager(llm.tokenizer)
                prompts_ = [tm.tokenize_one(p, {"enable_thinking": False}) for p in prompts]
            else:
                prompts_ = prompts
            llm.generate([prompts_[0][:8]], SamplingParams(max_tokens=2))  # warm-up
            st = llm.scheduler.stats
            st.batch_sizes.clear()
            steps0, tokens0 = st.decode_steps, st.decode_tokens
            outs, elapsed, tokens, _ = run(llm, prompts_, sps)
            got = [o.token_ids for o in outs]
            ref = ref or got
            mean_decode = (st.decode_tokens - tokens0) / max(1, st.decode_steps - steps0)
            print(f"| {name} | {adm} | {budget} | {elapsed:.1f} | {tokens / elapsed:.1f} | {max(st.batch_sizes)} "
                  f"| {mean_decode:.1f} | {st.retracted_reqs} | {got == ref} |")
            llm.shutdown()
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sections", default="batching,paged,admission",
                    help="comma-separated subset of: batching, paged, admission")
    args = ap.parse_args()
    sections = set(args.sections.split(","))
    if "admission" in sections:
        admission(args)
    if sections & {"batching", "paged"}:
        legacy(args, sections)


def legacy(args, sections) -> None:
    prompts, sps = workload(args.num_requests)

    if "batching" in sections:
        batching(args, prompts, sps)
    if "paged" in sections:
        paged(args, prompts, sps)


def batching(args, prompts, sps) -> None:
    print("## Throughput: sequential vs continuous batching\n")
    print("| mode | max_running_req | wall s | output tok/s | mean TTFT s | p90 TTFT s | mean batch size |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    results = {}
    for name, max_running in [("one at a time", 1), ("continuous batching", 64)]:
        llm = LLM(args.model, load_tokenizer_=False, device=args.device, max_total_tokens=16384,
                  max_seq_len=1024, max_running_req=max_running, cache_type="naive")
        llm.generate([prompts[0][:8]], SamplingParams(max_tokens=2))  # warm-up
        llm.scheduler.stats.batch_sizes.clear()
        outs, elapsed, tokens, ttft = run(llm, prompts, sps)
        results[name] = [o.token_ids for o in outs]
        decode_sizes = llm.scheduler.stats.batch_sizes
        ttft_sorted = sorted(ttft)
        print(
            f"| {name} | {max_running} | {elapsed:.1f} | {tokens / elapsed:.1f} | {statistics.mean(ttft):.2f} "
            f"| {ttft_sorted[int(0.9 * len(ttft_sorted)) - 1]:.2f} | {statistics.mean(decode_sizes):.1f} |"
        )
        llm.shutdown()
    same = results["one at a time"] == results["continuous batching"]
    print(f"\nidentical outputs in both modes: {same}\n")



def paged(args, prompts, sps) -> None:
    print("## Concurrency under a fixed KV budget (paged vs contiguous)\n")
    budget, max_seq_len = 2048, 1024
    llm = LLM(args.model, load_tokenizer_=False, device=args.device, max_total_tokens=budget,
              max_seq_len=max_seq_len, max_running_req=64, cache_type="naive", page_size=16)
    llm.generate(prompts, sps)
    peak = max(llm.scheduler.stats.batch_sizes)
    need = statistics.mean(len(p) + sp.max_tokens for p, sp in zip(prompts, sps))
    print(f"KV budget = {budget} tokens, max_seq_len = {max_seq_len}, mean request needs {need:.0f} tokens")
    print(f"- contiguous pre-allocation (reserve max_seq_len each): {budget // max_seq_len} concurrent requests")
    print(f"- paged allocation (page_size=16, measured peak batch):  {peak} concurrent requests")


if __name__ == "__main__":
    main()
