"""Part I Ch5 lab: benchmark metrics computed from real engine timestamps.

What it shows
  1. Precise definitions of TTFT, TPOT, ITL, E2E latency, output/request throughput and
     Goodput, computed from ``GenerateOutput.ttft`` / ``.token_times`` returned by
     ``minisgl.llm.LLM.generate`` (python/minisgl/llm/llm.py).
  2. Scenario A: 16 concurrent requests, 16 running slots -> everybody is prefilled in
     the first step; TTFT is identical for all; mean ~= P99.
  3. Scenario B: the same 16 requests, only 12 running slots -> 4 requests queue until a
     slot frees. Their TTFT is many steps longer: P50 stays low, P99 explodes, and the
     mean sits in between and describes nobody (Ch5 Sec.5.1).
  4. ITL vs TPOT: when a queued request is admitted, its prefill runs as a separate
     step (mini-sglang is prefill-first), so every running request skips a step: a
     jitter that per-token ITL shows and the averaged TPOT hides.

    python labs/part1_ch05_metrics.py [--model tiny|real] [--num-requests 16] [--slots 12]

Timestamps are taken once per scheduler iteration, so tokens produced by the same
forward pass share one timestamp. The lab uses that to count *steps* too, which makes
its assertions independent of how fast this machine is.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import List

from _common import lab_parser, make_llm, quiet_logs, resolve_model, section, table


# ---------------------------------------------------------------- metrics
def percentile(xs: List[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default, which SGLang's
    bench_serving uses via np.percentile)."""
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class ReqMetrics:
    ttft: float  # first token time - submission time (queueing + prefill)
    e2e: float  # last token time - submission time
    n_out: int  # output tokens
    itls: List[float]  # gaps between consecutive output tokens (n_out - 1 values)

    @property
    def tpot(self) -> float:
        """Convention used by SGLang's bench_serving and benchmarks/bench_serving.py:
        decode time divided by the tokens produced *after* the first one."""
        return (self.e2e - self.ttft) / (self.n_out - 1)


def req_metrics(out) -> ReqMetrics:
    t = out.token_times
    return ReqMetrics(ttft=out.ttft, e2e=t[-1], n_out=len(t), itls=[b - a for a, b in zip(t, t[1:])])


def summarize(ms: List[ReqMetrics], ttft_slo: float):
    wall = max(m.e2e for m in ms)  # all requests were submitted at t=0
    good = [m for m in ms if m.ttft <= ttft_slo]
    return dict(
        wall=wall,
        out_tok_s=sum(m.n_out for m in ms) / wall,  # output throughput
        req_s=len(ms) / wall,  # request throughput
        goodput_req_s=len(good) / wall,  # requests/s that met the TTFT SLO
        n_good=len(good),
    )


def dist_row(name: str, xs: List[float], scale: float = 1e3):
    return [name, f"{statistics.mean(xs) * scale:.1f}", f"{percentile(xs, 50) * scale:.1f}",
            f"{percentile(xs, 90) * scale:.1f}", f"{percentile(xs, 99) * scale:.1f}", f"{max(xs) * scale:.1f}"]


def step_index(outs):
    """Every scheduler step that emitted tokens has its own timestamp; map time -> step."""
    times = sorted({t for o in outs for t in o.token_times})
    return {t: i for i, t in enumerate(times)}


def run(path, prompts, sps, slots):
    from minisgl import SamplingParams

    llm = make_llm(path, max_running_req=slots, max_total_tokens=8192, max_seq_len=512, load_tokenizer_=False)
    cold = llm.generate([prompts[0]], SamplingParams(max_tokens=4, ignore_eos=True))[0]  # warm-up
    warm = llm.generate([prompts[0]], SamplingParams(max_tokens=4, ignore_eos=True))[0]
    llm.scheduler.stats.__init__()  # count only the measured run
    outs = llm.generate(prompts, sps)
    return llm, outs, (cold.ttft, warm.ttft)


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--slots", type=int, default=12, help="max_running_req for scenario B")
    args = p.parse_args(argv)
    assert 0 < args.slots < args.num_requests
    quiet_logs()
    from minisgl import SamplingParams

    path = resolve_model(args.model)
    rng = random.Random(0)
    n = args.num_requests
    prompts = [[rng.randrange(1, 500) for _ in range(rng.randint(8, 120))] for _ in range(n)]
    sps = [SamplingParams(max_tokens=rng.randint(16, 48), ignore_eos=True) for _ in range(n)]

    results = {}
    for label, slots in (("A", n), ("B", args.slots)):
        llm, outs, (cold, warm) = run(path, prompts, sps, slots)
        results[label] = (llm, outs)
        if label == "A":
            section("0. warm-up (Ch5 pitfall 3)")
            print(f"first-ever request TTFT {cold * 1e3:.1f} ms, same request again {warm * 1e3:.1f} ms "
                  "(one-off initialisation lands in the cold number)")

    # ------------------------------------------------------------------ 1
    section("1. definitions, checked on every request")
    print("TTFT = token_times[0]            (queueing + prefill; clock starts at submission)")
    print("E2E  = token_times[-1]")
    print("ITL  = token_times[i] - token_times[i-1], i = 1..n_out-1   (one value per gap)")
    print("TPOT = (E2E - TTFT) / (n_out - 1)                            (one value per request)")
    print("output throughput = sum(n_out) / wall;  goodput = requests meeting the SLO / wall")
    for label in "AB":
        for o, sp in zip(results[label][1], sps):
            m = req_metrics(o)
            assert m.n_out == sp.max_tokens and len(m.itls) == m.n_out - 1
            assert math.isclose(m.tpot, statistics.mean(m.itls), rel_tol=1e-9, abs_tol=1e-12)
            assert math.isclose(m.e2e, m.ttft + sum(m.itls), rel_tol=1e-9)
    print("-> asserted: TPOT == mean(ITL of that request) and E2E == TTFT + sum(ITL).")

    ms = {lab: [req_metrics(o) for o in results[lab][1]] for lab in "AB"}
    steps = {lab: step_index(results[lab][1]) for lab in "AB"}
    ttft_slo = 2 * percentile([m.ttft for m in ms["A"]], 50)

    # ------------------------------------------------------------------ 2/3
    for lab, title in (("A", f"2. scenario A: {n} requests, {n} running slots"),
                       ("B", f"3. scenario B: {n} requests, {args.slots} running slots ({n - args.slots} must queue)")):
        llm, outs = results[lab]
        m = ms[lab]
        st = llm.scheduler.stats
        section(title)
        print(f"scheduler steps: {st.prefill_steps} prefill + {st.decode_steps} decode")
        print(table([dist_row("TTFT ms", [x.ttft for x in m]),
                     dist_row("TPOT ms", [x.tpot for x in m]),
                     dist_row("ITL ms (all gaps)", [g for x in m for g in x.itls]),
                     dist_row("E2E ms", [x.e2e for x in m])],
                    ["metric", "mean", "P50", "P90", "P99", "max"]))
        s = summarize(m, ttft_slo)
        print(f"wall {s['wall'] * 1e3:.0f} ms | output {s['out_tok_s']:.0f} tok/s | {s['req_s']:.1f} req/s | "
              f"goodput (TTFT <= {ttft_slo * 1e3:.1f} ms = 2 x P50 of A) {s['goodput_req_s']:.1f} req/s "
              f"({s['n_good']}/{n} requests)")

    # Scenario A: everyone is prefilled in the same first step -> identical TTFT.
    tA = [x.ttft for x in ms["A"]]
    assert results["A"][0].scheduler.stats.prefill_steps == 1 and max(tA) == min(tA)
    assert all(steps["A"][b] - steps["A"][a] == 1 for o in results["A"][1] for a, b in zip(o.token_times, o.token_times[1:]))

    # Scenario B: FCFS admission; the last n - slots requests wait for a free slot.
    outs_b, mb, idx = results["B"][1], ms["B"], steps["B"]
    ttft_steps = [idx[o.token_times[0]] for o in outs_b]
    admitted, queued = ttft_steps[: args.slots], ttft_steps[args.slots:]
    min_out = min(sp.max_tokens for sp in sps[: args.slots])
    rows = [[i, len(prompts[i]), mb[i].n_out, ttft_steps[i], f"{mb[i].ttft * 1e3:.1f}", f"{mb[i].tpot * 1e3:.2f}",
             f"{max(mb[i].itls) * 1e3:.2f}", f"{mb[i].e2e * 1e3:.1f}"] for i in range(n)]
    print()
    print(table(rows, ["req", "prompt", "n_out", "TTFT step", "TTFT ms", "TPOT ms", "max ITL ms", "E2E ms"]))
    assert set(admitted) == {0} and min(queued) >= min_out  # waited for the shortest request to finish
    assert min(mb[i].ttft for i in range(args.slots, n)) > max(mb[i].ttft for i in range(args.slots))
    tB = [x.ttft for x in mb]
    p50, p99, mean = percentile(tB, 50), percentile(tB, 99), statistics.mean(tB)
    print(f"\nTTFT: P50 {p50 * 1e3:.1f} ms, mean {mean * 1e3:.1f} ms, P99 {p99 * 1e3:.1f} ms "
          f"(P99/P50 = {p99 / p50:.1f}x). The mean is pulled up by {n - args.slots} queued requests,")
    print("yet it is far below what those requests experienced: report percentiles, not the mean.")
    assert p50 < mean < p99 and p99 > 3 * p50
    assert summarize(mb, ttft_slo)["n_good"] < n

    # ------------------------------------------------------------------ 4
    section("4. ITL jitter that TPOT averages away")
    skipped = [(i, b_idx - a_idx) for i, o in enumerate(outs_b) for a, b in zip(o.token_times, o.token_times[1:])
               if (b_idx := idx[b]) - (a_idx := idx[a]) > 1]
    print(f"{len(skipped)} token gaps in scenario B span 2+ scheduler steps: a queued request's prefill ran in")
    print("between and the running requests produced nothing in that step (prefill-first policy,")
    print("python/minisgl/scheduler/scheduler.py:_schedule_next_batch). Example gaps (req, steps):",
          skipped[:6])
    worst = max(range(n), key=lambda i: max(mb[i].itls) / mb[i].tpot)
    print(f"worst request {worst}: max ITL {max(mb[worst].itls) * 1e3:.2f} ms vs its TPOT {mb[worst].tpot * 1e3:.2f} ms")
    assert len(skipped) > 0 and all(s == 2 for _, s in skipped)


if __name__ == "__main__":
    main()
