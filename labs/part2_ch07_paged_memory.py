"""Part II Ch7 lab: paged KV memory, measured on the real allocator.

What it shows
  1. Contiguous vs paged under one KV budget: reserving ``max_seq_len`` per request
     fits 2 requests in 2048 tokens; the real scheduler (page_size 16) runs all 16
     requests of the ``bench_batching.py`` workload at once. Plus a first-fit
     contiguous allocator that fails with enough total free memory (external
     fragmentation) where pages succeed.
  2. Internal fragmentation (allocated - used slots) for page_size 1/4/16 over a
     mixed workload, read from ``CacheManager.free_slots`` after ``allocate_paged``;
     ``check_integrity`` after everything is freed.
  3. The block table: ``page_table`` rows after interleaved growth of three
     requests and a recycled allocation; physical pages are not contiguous.
  4. Online softmax: attention over K/V read block by block through a shuffled
     block table, merged with a running max / sum, equals full softmax attention.
  5. Corrupting the block table: swap one physical page between two running
     requests -> tokens change; swap two pages *inside* one request's history ->
     tokens do not (attention over a fully visible history is order-free).
     page_size 1/4/16 all produce the same tokens.

    python labs/part2_ch07_paged_memory.py [--model tiny|real]
"""

from __future__ import annotations

import gc
import math
import random

from _common import lab_parser, make_llm, quiet_logs, resolve_model, section, table


def fake_req(idx: int, length: int):
    """A ``Req`` whose next step must cover positions [0, length)."""
    import torch

    from minisgl import SamplingParams
    from minisgl.core import Req
    from minisgl.kvcache.naive_cache import NaiveCacheHandle

    return Req(input_ids=torch.zeros(length, dtype=torch.int64), table_idx=idx, cached_len=0, output_len=1,
               uid=idx, sampling_params=SamplingParams(), cache_handle=NaiveCacheHandle(torch.device("cpu")))


def finish(cm, req) -> None:
    """What the scheduler does when a request ends: cache_req(finished=True)."""
    req.cached_len = req.device_len  # everything allocated has been computed
    cm.cache_req(req, finished=True)


def new_cache_manager(num_pages: int, page_size: int, rows: int = 64, cols: int = 512):
    import torch

    from minisgl.scheduler.cache import CacheManager

    page_table = torch.zeros((rows, cols), dtype=torch.int64)
    return CacheManager(torch.device("cpu"), num_pages, page_size, page_table, "naive")


def first_fit(holes, size):
    """Contiguous allocator: index of the first free hole that fits, or None."""
    for i, (start, n) in enumerate(holes):
        if n >= size:
            return i
    return None


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    from minisgl import SamplingParams
    from minisgl.llm import generate_without_kv_cache

    path = resolve_model(args.model)

    # ------------------------------------------------------------------ 1
    section("1. contiguous vs paged under a fixed KV budget (cf. benchmarks/bench_batching.py)")
    budget, max_seq_len, ps = 2048, 1024, 16
    rng = random.Random(0)
    prompts = [[rng.randrange(4, 500) for _ in range(rng.randrange(16, 96))] for _ in range(16)]
    sps = [SamplingParams(max_tokens=rng.randrange(16, 64), ignore_eos=True) for _ in range(16)]
    llm = make_llm(path, load_tokenizer_=False, max_total_tokens=budget, max_seq_len=max_seq_len,
                   max_running_req=64, cache_type="naive", page_size=ps)
    sched, cm = llm.scheduler, llm.scheduler.cache_manager
    peak = {"reqs": 0, "used": 0, "alloc": 0}
    orig = sched._prepare_batch

    def traced(batch):
        fi = orig(batch)
        alloc = (cm.num_pages - len(cm.free_slots)) * ps
        if batch.is_decode and batch.size >= peak["reqs"]:
            peak.update(reqs=batch.size, used=sum(r.device_len for r in batch.reqs), alloc=alloc)
        return fi

    sched._prepare_batch = traced
    llm.generate(prompts, sps)
    sched._prepare_batch = orig
    cm.check_integrity()
    llm.shutdown()
    del llm, sched, cm, orig, traced
    gc.collect()  # one engine at a time: matters with --model real
    need = [len(p_) + sp.max_tokens for p_, sp in zip(prompts, sps)]
    contiguous = budget // max_seq_len
    print(f"KV budget {budget} tokens, max_seq_len {max_seq_len}; requests need {min(need)}..{max(need)} "
          f"tokens (mean {sum(need) / len(need):.0f})")
    print(table([
        ["contiguous, reserve max_seq_len each", contiguous, f"{contiguous * max_seq_len}", "-"],
        [f"paged (page_size {ps}), measured peak decode batch", peak["reqs"], peak["alloc"],
         f"{peak['used'] / peak['alloc']:.1%}"],
    ], ["allocation", "concurrent requests", "slots reserved/allocated", "used / allocated at peak"]))
    assert peak["reqs"] == 16 > contiguous
    print("-> README (Qwen3-0.6B): 'contiguous pre-allocation fits 2 concurrent requests; paged ran 16'.")
    print("   Admission still reserves extend+output+page_size per request (Ch6), but only real")
    print("   pages are taken from free_slots, and only as each request grows.")

    print("\nexternal fragmentation with exact-size contiguous regions (first fit, 2048 tokens):")
    sizes = [("r1", 512), ("r2", 256), ("r3", 512), ("r4", 256), ("r5", 512)]
    holes = [(0, budget)]
    live = {}
    for name, size in sizes:
        i = first_fit(holes, size)
        start, n = holes[i]
        live[name] = (start, size)
        holes[i] = (start + size, n - size)
    for name in ("r2", "r4"):  # two short requests finish
        holes.append(live.pop(name))
    holes = sorted(h for h in holes if h[1] > 0)
    want = 384
    print(f"  contiguous: free holes (start, size) = {holes}, total free = {sum(n for _, n in holes)}")
    print(f"  new request needs {want} contiguous tokens -> first fit: {first_fit(holes, want)}")
    assert first_fit(holes, want) is None and sum(n for _, n in holes) >= want
    cm16 = new_cache_manager(budget // ps, ps)  # the same story on the real page allocator
    reqs = {name: fake_req(i, size) for i, (name, size) in enumerate(sizes)}
    cm16.allocate_paged(list(reqs.values()))
    assert len(cm16.free_slots) == 0
    for name in ("r2", "r4"):
        finish(cm16, reqs.pop(name))
    new = fake_req(5, want)
    cm16.allocate_paged([new])
    pages = (cm16.page_table[5, :want:ps] // ps).tolist()
    print(f"  paged: after r2, r4 finish there are {len(pages) + len(cm16.free_slots)} free pages; the "
          f"{want}-token request gets pages {pages[:3]}..{pages[-3:]} -> fits")
    assert sorted(set(pages)) != list(range(min(pages), min(pages) + len(pages)))  # spans both holes
    with cm16.lazy_free_region():
        for r in [*reqs.values(), new]:
            finish(cm16, r)
    cm16.check_integrity()

    # ------------------------------------------------------------------ 2
    section("2. internal fragmentation for page_size 1 / 4 / 16 (real CacheManager)")
    rng = random.Random(1)
    lengths = [rng.randrange(1, 200) for _ in range(32)]
    used = sum(lengths)
    rows = []
    for ps in (1, 4, 16):
        cmp = new_cache_manager(8192 // ps, ps)
        reqs = [fake_req(i, n) for i, n in enumerate(lengths)]
        cmp.allocate_paged(reqs)
        allocated = (cmp.num_pages - len(cmp.free_slots)) * ps
        waste = allocated - used
        assert waste == sum(-(-n // ps) * ps - n for n in lengths) <= len(lengths) * (ps - 1)
        rows.append([ps, used, allocated, waste, len(lengths) * (ps - 1), f"{used / allocated:.1%}"])
        with cmp.lazy_free_region():
            for r in reqs:
                finish(cmp, r)
        cmp.check_integrity()  # free + cached pages == num_pages, no page freed twice
        assert len(cmp.free_slots) == cmp.num_pages
    contig = len(lengths) * 256
    rows.append(["contiguous (max_seq_len 256)", used, contig, contig - used, "-", f"{used / contig:.1%}"])
    print(f"{len(lengths)} requests, lengths {min(lengths)}..{max(lengths)}, {used} tokens in total")
    print(table(rows, ["page_size", "used slots", "allocated slots", "waste", "bound n*(ps-1)", "utilization"]))
    print("-> paged waste lives only in each request's last page; contiguous waste is everything")
    print("   between the real length and the max_seq_len reservation.")

    # ------------------------------------------------------------------ 3
    section("3. block table (page_table rows) after interleaved growth")
    ps = 4
    cm3 = new_cache_manager(12, ps)
    reqs = {n: fake_req(i, 0) for i, n in enumerate("ABC")}
    for _ in range(4):  # each request grows by 3 tokens per round, round robin
        for r in reqs.values():
            r.cached_len, r.device_len = r.device_len, r.device_len + 3
            cm3.allocate_paged([r])
    finish(cm3, reqs["B"])  # B ends; its pages go to the back of free_slots
    reqs["D"] = fake_req(3, 20)
    cm3.allocate_paged([reqs["D"]])
    rows = []
    for name, r in reqs.items():
        if name == "B":
            continue
        row = cm3.page_table[r.table_idx, : r.device_len]
        pages = (row[::ps] // ps).tolist()
        rows.append([name, r.table_idx, r.device_len, pages, row.tolist()])
    print(table(rows, ["req", "row", "tokens", "logical page -> physical page", "token -> slot"]))
    a_pages = rows[0][3]
    assert a_pages != list(range(a_pages[0], a_pages[0] + len(a_pages)))  # not contiguous
    tok = 9
    row_a = cm3.page_table[reqs["A"].table_idx]
    phys = int(row_a[(tok // ps) * ps]) // ps
    print(f"A's token {tok}: logical page {tok // ps}, offset {tok % ps} -> physical page {phys} -> "
          f"slot {phys * ps + tok % ps} (page_table says {int(row_a[tok])})")
    assert int(row_a[tok]) == phys * ps + tok % ps
    print("-> pages are handed out in free-list order as requests grow, so a request's pages")
    print("   interleave with its neighbours'; D reuses B's freed pages after the untouched ones.")
    with cm3.lazy_free_region():
        for name in "ACD":
            finish(cm3, reqs[name])
    cm3.check_integrity()

    # ------------------------------------------------------------------ 4
    section("4. online softmax over blocks read through a block table")
    g = torch.Generator().manual_seed(0)
    L, d, b = 37, 16, 8
    K = torch.randn(L, d, generator=g, dtype=torch.float64)
    V = torch.randn(L, d, generator=g, dtype=torch.float64)
    q = torch.randn(d, generator=g, dtype=torch.float64) * 2
    n_blocks = -(-L // b)
    pool_k = torch.zeros(16, b, d, dtype=torch.float64)  # physical blocks
    pool_v = torch.zeros_like(pool_k)
    block_table = torch.randperm(16, generator=g)[:n_blocks]
    for j in range(n_blocks):
        seg = slice(j * b, min(L, (j + 1) * b))
        pool_k[block_table[j], : seg.stop - seg.start] = K[seg]
        pool_v[block_table[j], : seg.stop - seg.start] = V[seg]

    def paged_attention(query, visible_len, trace=None):
        m, l_sum, O = float("-inf"), torch.tensor(0.0, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
        for j in range(n_blocks):
            valid = min(b, visible_len - j * b)
            if valid <= 0:
                break  # fully masked block: contributes nothing
            kb, vb = pool_k[block_table[j], :valid], pool_v[block_table[j], :valid]
            s = kb @ query / d ** 0.5
            m_j = float(s.max())
            l_j, O_j = torch.exp(s - m_j).sum(), torch.exp(s - m_j) @ vb
            m_new = max(m, m_j)
            alpha, beta = math.exp(m - m_new), math.exp(m_j - m_new)  # move both to reference m'
            l_sum = l_sum * alpha + l_j * beta
            O = O * alpha + O_j * beta
            m = m_new
            if trace is not None:
                trace.append([j, int(block_table[j]), valid, f"{m_j:.3f}", f"{m:.3f}", f"{float(l_sum):.4f}"])
        return O / l_sum

    def full_attention(query, visible_len):
        s = K[:visible_len] @ query / d ** 0.5
        return torch.softmax(s, dim=0) @ V[:visible_len]

    trace = []
    out_paged = paged_attention(q, L, trace)
    out_full = full_attention(q, L)
    print(f"L={L} keys, block size {b}, block table {block_table.tolist()} (shuffled physical blocks)")
    print(table(trace, ["logical block", "physical block", "valid slots", "block max m_j", "running m",
                        "running l"]))
    err = float((out_paged - out_full).abs().max())
    print(f"max |paged online-softmax - full softmax| = {err:.2e}")
    assert err < 1e-12
    causal = float((paged_attention(q, 20) - full_attention(q, 20)).abs().max())
    print(f"causal query at position 19 (blocks 3, 4 fully masked, skipped): max error {causal:.2e}")
    assert causal < 1e-12
    wrong = torch.zeros(d, dtype=torch.float64)
    for j in range(n_blocks):  # mistake: average per-block softmax outputs, no rescaling
        seg = slice(j * b, min(L, (j + 1) * b))
        wrong += torch.softmax(K[seg] @ q / d ** 0.5, 0) @ V[seg] / n_blocks
    print(f"without the e^(m - m') rescaling (averaging block outputs): error "
          f"{float((wrong - out_full).abs().max()):.2e}")
    assert float((wrong - out_full).abs().max()) > 1e-3

    # ------------------------------------------------------------------ 5
    section("5. corrupt the block table of running requests")
    ps, n_out = 4, 8
    prompts = [list(range(30, 38)), list(range(300, 308))]  # 8 tokens = 2 pages each
    llm = make_llm(path, load_tokenizer_=False, page_size=ps, cache_type="naive", max_total_tokens=256)
    eng, sched = llm.engine, llm.scheduler
    sp = SamplingParams(max_tokens=n_out, ignore_eos=True)
    ref = [o.token_ids for o in llm.generate(prompts, sp)]
    naive = [generate_without_kv_cache(eng, p_, n_out) for p_ in prompts]
    assert ref == naive

    def run_with(corrupt):
        orig = sched._prepare_batch
        done = []

        def hook(batch):
            if batch.is_decode and not done:  # before the first decode step: pages hold the prompts
                rows = [r.table_idx for r in batch.reqs]
                corrupt(eng.page_table, rows)
                done.append(True)
            return orig(batch)

        sched._prepare_batch = hook
        try:
            return [o.token_ids for o in llm.generate(prompts, sp)]
        finally:
            sched._prepare_batch = orig
            sched.cache_manager.check_integrity()

    def swap_between(pt, rows):  # logical page 0 of A <-> logical page 0 of B
        a, b_ = pt[rows[0], 0:ps].clone(), pt[rows[1], 0:ps].clone()
        pt[rows[0], 0:ps], pt[rows[1], 0:ps] = b_, a

    def swap_within(pt, rows):  # logical pages 0 and 1 of A
        a0, a1 = pt[rows[0], 0:ps].clone(), pt[rows[0], ps:2 * ps].clone()
        pt[rows[0], 0:ps], pt[rows[0], ps:2 * ps] = a1, a0

    between = run_with(swap_between)
    within = run_with(swap_within)
    print(table([
        ["reference (no KV cache, llm/naive.py)", naive[0], naive[1]],
        ["paged engine", ref[0], ref[1]],
        ["swap page 0 of A <-> page 0 of B", between[0], between[1]],
        ["swap pages 0 <-> 1 inside A", within[0], within[1]],
    ], ["run", "A tokens", "B tokens"]))
    assert between[0] != ref[0] and between[1] != ref[1]
    assert within == ref
    print("-> between requests: A's queries now read B's prompt K/V (and vice versa): wrong tokens,")
    print("   no error raised. Inside one request: decode queries see the whole history and RoPE")
    print("   is already baked into the cached K, so the order of fully visible pages does not")
    print("   matter. (It would matter for a prefill chunk, whose causal mask is position-based.)")

    llm.shutdown()
    del llm, eng, sched, run_with
    gc.collect()
    outs = {}
    for ps_ in (1, 4, 16):
        llm_ = make_llm(path, load_tokenizer_=False, page_size=ps_, cache_type="naive", max_total_tokens=256)
        outs[ps_] = [o.token_ids for o in llm_.generate(prompts, sp)]
        llm_.shutdown()
        del llm_
        gc.collect()
    print(f"page_size 1 / 4 / 16 give identical tokens: {outs[1] == outs[4] == outs[16] == ref}")
    assert outs[1] == outs[4] == outs[16] == ref


if __name__ == "__main__":
    main()
