"""Part II Ch8 lab: watch the radix tree (RadixAttention's prefix cache) work.

What it shows
  1. A radix-tree pretty printer (key tokens, value slots, ref_count, LRU order,
     evictable / protected sizes) built on ``RadixPrefixCache.root_node``.
  2. insert A, insert B sharing a prefix -> node split; insert C -> second split.
  3. match_prefix + lock_handle -> ref_count along the whole path to the root.
  4. evict(n) -> only unreferenced leaves, oldest first; a parent that becomes a
     leaf is evicted next. Locked paths survive.
  5. page_size=4: only whole pages are inserted / matched.
  6. The real scheduler (cache_type='radix'): the tree after each ``generate`` for
     requests sharing a system prompt.
  7. ``CacheManager.match_req`` never matches the last prompt token.
  8. The double ``available_size`` check around ``lock`` in
     ``PrefillAdder._try_allocate_one`` (a case where check 1 passes and check 2 fails).
  9. ``cache_req`` duplicates (Ch8 code exercise 4): two identical prompts in one
     prefill batch -> the second insert finds prefix_len > old cached_len, its
     duplicate pages are freed and its page-table row is repointed.

    python labs/part2_ch08_radix_tree.py [--model tiny|real]
"""

from __future__ import annotations

from typing import List

from _common import lab_parser, make_llm, quiet_logs, resolve_model, section, table


# --------------------------------------------------------------------------- printer
def _runs(xs: List[int]) -> str:
    """[100,101,102,7,8] -> '100-102,7-8' (slot lists are mostly contiguous runs)."""
    if not xs:
        return "-"
    out, start, prev = [], xs[0], xs[0]
    for x in xs[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ",".join(out)


def _toks(xs: List[int], limit: int = 8) -> str:
    if len(xs) <= limit:
        return "[" + " ".join(map(str, xs)) + "]"
    return "[" + " ".join(map(str, xs[:4])) + " ... " + " ".join(map(str, xs[-2:])) + f"] ({len(xs)} tok)"


def _all_nodes(cache):
    nodes, stack = [], list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children.values())
    return nodes


def dump_tree(cache, title: str = "") -> None:
    """Print the tree. ``lru#1`` is the oldest timestamp (evicted first among leaves)."""
    order = {ts: i + 1 for i, ts in enumerate(sorted({n.timestamp for n in _all_nodes(cache)}))}
    if title:
        print(f"-- {title}")
    print("root")

    def walk(node, prefix: str) -> None:
        kids = sorted(node.children.values(), key=lambda c: c.key.tolist())
        for i, child in enumerate(kids):
            last = i == len(kids) - 1
            leaf = " leaf" if child.is_leaf() else ""
            print(
                f"{prefix}{'└─' if last else '├─'} key={_toks(child.key.tolist())} "
                f"slots={_runs(child.value.tolist())} ref={child.ref_count} lru#{order[child.timestamp]}{leaf}"
            )
            walk(child, prefix + ("   " if last else "│  "))

    walk(cache.root_node, "")
    si = cache.size_info
    print(f"   evictable={si.evictable_size} protected={si.protected_size} total={si.total_size} "
          f"nodes(excl. root)={cache.num_nodes() - 1}")


def path_refs(handle) -> List[int]:
    """ref_count of every node from the matched node up to (excluding) the root."""
    refs, node = [], handle.node
    while not node.is_root():
        refs.append(node.ref_count)
        node = node.parent
    return refs


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    quiet_logs()

    import torch

    from minisgl import SamplingParams
    from minisgl.kvcache import RadixPrefixCache
    from minisgl.scheduler.prefill import PendingReq, PrefillAdder

    t = lambda xs: torch.tensor(xs, dtype=torch.int64)  # noqa: E731
    cpu = torch.device("cpu")

    # ------------------------------------------------------------------ 1-2
    section("1-2. insert A, then B sharing a prefix (split), then C (second split)")
    cache = RadixPrefixCache(cpu)
    A, B, C = [101, 11, 12, 21, 22], [101, 11, 12, 31, 32], [101, 11, 99]
    pl, _ = cache.insert_prefix(t(A), t([0, 1, 2, 3, 4]))
    assert pl == 0
    dump_tree(cache, "after insert A -> one compressed node")
    # B's request computed its own KV in slots 10..14; the tree already had 3 of them.
    pl, hb = cache.insert_prefix(t(B), t([10, 11, 12, 13, 14]))
    print(f"insert B returned prefix_len={pl}: B's slots 10-12 are duplicates the caller must free")
    assert pl == 3 and hb.get_matched_indices().tolist() == [0, 1, 2, 13, 14]
    dump_tree(cache, "after insert B -> A's node split at 3: [101 11 12] -> {[21 22], [31 32]}")
    pl, _ = cache.insert_prefix(t(C), t([20, 21, 22]))
    assert pl == 2 and cache.num_nodes() == 6
    dump_tree(cache, "after insert C -> [101 11 12] split again at 2 (tutorial Figure 4)")
    cache.check_integrity()

    # ------------------------------------------------------------------ 3
    section("3. match_prefix + lock_handle: ref_count along the path")
    h = cache.match_prefix(t(B + [77])).cuda_handle
    print(f"match_prefix(B+[77]) -> cached_len={h.cached_len}, indices={h.get_matched_indices().tolist()}")
    print(f"path ref_counts (matched node -> root) before lock: {path_refs(h)}")
    cache.lock_handle(h)
    print(f"path ref_counts after lock:                       {path_refs(h)}")
    assert path_refs(h) == [1, 1, 1] and cache.size_info.protected_size == 5
    h2 = cache.match_prefix(t(C + [5])).cuda_handle  # shares only [101 11]
    cache.lock_handle(h2)
    print(f"lock C's path as well -> shared node [101 11] now has ref=2: {path_refs(h2)}")
    assert path_refs(h2) == [1, 2]
    dump_tree(cache, "two locked paths")
    cache.lock_handle(h2, unlock=True)
    cache.check_integrity()

    # ------------------------------------------------------------------ 4
    section("4. evict: unreferenced leaves only, LRU order, parents cascade")
    # Touch C's leaf so A's leaf [21 22] is the least recently used.
    cache.match_prefix(t(C))
    dump_tree(cache, "B locked; C touched recently; A's tail is the oldest leaf")
    ev = cache.evict(1).tolist()
    print(f"evict(1) -> {ev}  (a whole leaf goes even though only 1 token was asked for)")
    assert ev == [3, 4]
    ev = cache.evict(1).tolist()
    print(f"evict(1) -> {ev}  (next unreferenced leaf: C's [99])")
    assert ev == [22]
    try:
        cache.evict(1)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        print(f"evict(1) -> RuntimeError: {e}  (everything left is on B's locked path)")
    dump_tree(cache, "only the locked path is left")
    cache.lock_handle(h, unlock=True)
    ev = cache.evict(5).tolist()
    print(f"unlock B, evict(5) -> {ev}: leaf [31 32] first, then its parent [12] became a leaf,")
    print("   then [101 11]: a parent is pushed onto the heap the moment its last child is evicted")
    assert ev == [13, 14, 2, 0, 1] and cache.num_nodes() == 1
    cache.check_integrity()

    # ------------------------------------------------------------------ 5
    section("5. page_size=4: only whole pages enter the tree")
    pc = RadixPrefixCache(cpu, page_size=4)
    pl, hh = pc.insert_prefix(t(list(range(10))), t(list(range(100, 110))))
    print(f"insert 10 tokens -> handle.cached_len={hh.cached_len} (tokens 8,9 = partial page, not inserted)")
    assert hh.cached_len == 8
    m = pc.match_prefix(t([0, 1, 2, 3, 4, 5, 99, 7, 8])).cuda_handle
    print(f"match [0..5, 99, ...] -> common prefix 6, aligned down to cached_len={m.cached_len} -> split at 4")
    assert m.cached_len == 4
    m3 = pc.match_prefix(t([0, 1, 2])).cuda_handle
    print(f"match 3 tokens (< one page) -> cached_len={m3.cached_len}")
    assert m3.cached_len == 0
    pl, _ = pc.insert_prefix(t([0, 1, 2, 3, 9, 9, 9, 9, 5]), t(list(range(200, 209))))
    assert pl == 4
    dump_tree(pc, "page_size=4 tree: every node length is a multiple of 4; children keyed by first page")
    assert all(n.length % 4 == 0 for n in _all_nodes(pc))
    pc.check_integrity()

    # ------------------------------------------------------------------ 6
    section("6. real scheduler: requests sharing a system prompt")
    path = resolve_model(args.model)
    llm = make_llm(path, cache_type="radix", page_size=1, max_total_tokens=256, load_tokenizer_=False)
    sched = llm.scheduler
    cm, pstats = sched.cache_manager, sched.prefill_manager.stats
    sp = SamplingParams(max_tokens=3, ignore_eos=True)
    system = list(range(10, 26))  # 16-token "system prompt"
    users = [[300, 301, 302], [400, 401], [450, 451, 452, 453], [460]]
    rows = []
    for i, prompts in enumerate([[system + users[0]], [system + users[1]], [system + users[2], system + users[3]]]):
        before_computed, before_cached = sched.stats.prefill_tokens, pstats.cached_tokens
        outs = llm.generate(prompts, sp)
        computed = sched.stats.prefill_tokens - before_computed
        cached = pstats.cached_tokens - before_cached
        rows.append([i + 1, len(prompts), sum(len(x) for x in prompts), cached, computed])
        dump_tree(cm.prefix_cache, f"after generate #{i + 1} ({len(prompts)} request(s)); outputs "
                                   f"{[o.token_ids for o in outs]}")
        cm.check_integrity()
    print(table(rows, ["generate", "reqs", "prompt tokens", "cached (reused)", "prefill computed"]))
    assert rows[0][3] == 0 and rows[1][3] == 16 and rows[2][3] == 32
    print("-> every finished request leaves prompt + output[:-1] in the tree (the last sampled token never")
    print("   had its K/V computed); later requests reuse the 16-token system prompt.")

    # ------------------------------------------------------------------ 7
    section("7. match_req never matches the last prompt token")
    full = system + users[0]  # fully cached (and more) after generate #1
    raw = cm.prefix_cache.match_prefix(t(full)).cuda_handle.cached_len
    via_req = cm.match_req(t(full)).cuda_handle.cached_len
    print(f"prompt len={len(full)}: prefix_cache.match_prefix -> {raw}, CacheManager.match_req -> {via_req}")
    assert raw == len(full) and via_req == len(full) - 1
    before = pstats.cached_tokens
    again = llm.generate([full], sp)[0]
    print(f"re-running the same prompt: cached_len={pstats.cached_tokens - before} = len-1; prefill still runs "
          f"1 token to get the first output's logits -> {again.token_ids}")
    assert pstats.cached_tokens - before == len(full) - 1
    cm.check_integrity()

    # ------------------------------------------------------------------ 8
    section("8. the double available_size check around lock()")
    req = PendingReq(uid=999, input_ids=t(system + [700, 701, 702, 703]), sampling_params=sp, output_len=8)
    handle = cm.match_req(req.input_ids).cuda_handle
    est = req.input_len - handle.cached_len + req.output_len + cm.page_size
    avail_before = cm.available_size
    cm.lock(handle)
    avail_after = cm.available_size
    cm.unlock(handle)
    # pick a decode reservation so that check 1 passes but check 2 fails
    reserved = avail_before - est
    print(f"cached_len={handle.cached_len} estimated_len={est} reserved_size={reserved}")
    print(f"available_size before lock={avail_before} (free={len(cm.free_slots)} + evictable="
          f"{cm.prefix_cache.size_info.evictable_size}); after lock={avail_after}")
    print(f"check 1: {est}+{reserved}={est + reserved} <= {avail_before}  -> passes")
    print(f"check 2: {est}+{reserved}={est + reserved} <= {avail_after}  -> fails (the matched prefix moved "
          "from evictable to protected)")
    adder = PrefillAdder(token_budget=512, reserved_size=reserved, cache_manager=cm, table_manager=sched.table_manager)
    prot0 = cm.prefix_cache.size_info.protected_size
    assert adder._try_allocate_one(req) is None
    assert cm.prefix_cache.size_info.protected_size == prot0, "the failed admission must unlock"
    print("_try_allocate_one -> None, and the handle was unlocked again (protected_size unchanged)")
    adder.reserved_size = reserved - (avail_before - avail_after)
    res = adder._try_allocate_one(req)
    assert res is not None
    print(f"with reserved_size={adder.reserved_size} it is admitted: table_idx={res[1]}; undo it for the lab")
    cm.unlock(res[0])
    sched.table_manager.free(res[1])
    cm.check_integrity()

    # ------------------------------------------------------------------ 9
    section("9. cache_req duplicates: two identical prompts in the same prefill batch")
    log = []
    orig_cache_req = cm.cache_req

    def traced_cache_req(r, *, finished):
        old = r.cache_handle.cached_len
        n = r.cached_len
        row_before = cm.page_table[r.table_idx, :n].tolist()
        orig_insert = cm.prefix_cache.insert_prefix
        got = {}

        def traced_insert(ids, idx):
            got["prefix_len"], h = orig_insert(ids, idx)
            return got["prefix_len"], h

        cm.prefix_cache.insert_prefix = traced_insert
        try:
            orig_cache_req(r, finished=finished)
        finally:
            del cm.prefix_cache.insert_prefix
        log.append(dict(uid=r.uid, finished=finished, old=old, prefix_len=got["prefix_len"], n=n,
                        before=row_before, after=cm.page_table[r.table_idx, :n].tolist()))

    cm.cache_req = traced_cache_req
    dup = list(range(40, 52))  # not in the tree yet
    outs = llm.generate([dup, dup], SamplingParams(max_tokens=4, ignore_eos=True))
    del cm.cache_req
    first = [e for e in log if not e["finished"]]
    print(table([[e["uid"], e["old"], e["prefix_len"], _runs(e["before"]), _runs(e["after"])] for e in first],
                ["uid", "old_handle.cached_len", "insert prefix_len", "page_table row before", "row after"]))
    a, b = first
    assert a["prefix_len"] == 0 and b["prefix_len"] == len(dup) > b["old"] == 0
    assert b["after"] == a["before"] and b["before"] != a["before"]
    print(f"-> uid {b['uid']}: page_indices[{b['old']}:{b['prefix_len']}] (slots {_runs(b['before'])}) were "
          f"duplicates and were freed; its row now points at the cached copy (slots {_runs(a['before'])}).")
    print("   The reference only frees them; the still-running request would then read freed slots.")
    assert outs[0].token_ids == outs[1].token_ids
    cm.check_integrity()
    print(f"outputs identical: {outs[0].token_ids == outs[1].token_ids}; check_integrity: OK "
          f"(free pages + cached pages == {cm.num_pages})")
    dump_tree(cm.prefix_cache, "final tree")


if __name__ == "__main__":
    main()
