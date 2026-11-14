"""Check that every SGLang path/symbol cited in docs/*.md exists in an SGLang checkout.

    python docs/tools/verify_source_map.py /tmp/sglang [docs/sglang-source-map.md ...]

Recognized citations (inside backticks):
  srt/a/b.py::Name            file python/sglang/srt/a/b.py defines Name
  srt/a/b.py::Cls.method      file defines class Cls and def method
  srt/a/b.py / srt/a/dir/     file or directory exists
  kernels/x/y.py::name        python/sglang/kernels/x/y.py
  test/..., docs/docs/..., .github/...  repo-root relative
Bare method names after a cited class in the same cell (e.g. ``Scheduler.get_next_batch_to_run``)
are checked against python/sglang/srt/managers/scheduler.py when the class is Scheduler, etc.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

CITE = re.compile(r"`((?:srt|kernels)/[\w/.\-]+|(?:test|docs/docs|\.github)/[\w/.\-]+)(?:::([\w.]+))?`")
# `Class.method` mentions resolved against known files
KNOWN = {
    "Scheduler": "python/sglang/srt/managers/scheduler.py",
    "ScheduleBatch": "python/sglang/srt/managers/schedule_batch.py",
    "PrefillAdder": "python/sglang/srt/managers/schedule_policy.py",
    "SchedulePolicy": "python/sglang/srt/managers/schedule_policy.py",
    "TokenizerManager": "python/sglang/srt/managers/tokenizer_manager.py",
    "DetokenizerManager": "python/sglang/srt/managers/detokenizer_manager.py",
    "ModelRunner": "python/sglang/srt/model_executor/model_runner.py",
    "RadixCache": "python/sglang/srt/mem_cache/radix_cache.py",
    "UnifiedRadixCache": "python/sglang/srt/mem_cache/unified_radix_cache.py",
    "MHATokenToKVPool": "python/sglang/srt/mem_cache/memory_pool.py",
    "ReqToTokenPool": "python/sglang/srt/mem_cache/memory_pool.py",
    "PagedTokenToKVPoolAllocator": "python/sglang/srt/mem_cache/allocator/paged.py",
    "TritonAttnBackend": "python/sglang/srt/layers/attention/triton_backend.py",
    "TorchNativeAttnBackend": "python/sglang/srt/layers/attention/torch_native_backend.py",
    "QKVParallelLinear": "python/sglang/srt/layers/linear.py",
    "Qwen3ForCausalLM": "python/sglang/srt/models/qwen3.py",
    "EAGLEWorkerV2": "python/sglang/srt/speculative/eagle_worker_v2.py",
    "LogitsProcessor": "python/sglang/srt/layers/logits_processor.py",
    "Sampler": "python/sglang/srt/layers/sampler.py",
    "GroupCoordinator": "python/sglang/srt/distributed/parallel_state.py",
    "ForwardBatch": "python/sglang/srt/model_executor/forward_batch_info.py",
    "Engine": "python/sglang/srt/entrypoints/engine.py",
    "HiCacheController": "python/sglang/srt/managers/cache_controller.py",
}
METHOD = re.compile(r"`(" + "|".join(KNOWN) + r")\.(\w+)`")


def defines(path: str, name: str) -> bool:
    src = open(path, encoding="utf-8").read()
    return re.search(rf"^\s*(?:async\s+)?(?:class|def)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]", src, re.M) is not None


def resolve(root: str, p: str) -> str:
    if p.startswith("srt/") or p.startswith("kernels/"):
        return os.path.join(root, "python/sglang", p)
    return os.path.join(root, p)


def main() -> int:
    root = sys.argv[1]
    files = sys.argv[2:] or [os.path.join(os.path.dirname(__file__), "..", "sglang-source-map.md")]
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = bad = 0
    failures = []
    for f in files:
        text = open(f, encoding="utf-8").read()
        for m in CITE.finditer(text):
            path, sym = m.group(1), m.group(2)
            full = resolve(root, path.rstrip("/"))
            if not os.path.exists(full):
                bad += 1
                failures.append(f"missing path: {path}")
                continue
            if sym:
                parts = sym.split(".")
                missing = [s for s in parts if not os.path.isfile(full) or not defines(full, s)]
                if missing:
                    bad += 1
                    failures.append(f"missing symbol: {path}::{sym} ({missing})")
                    continue
            ok += 1
        for m in METHOD.finditer(text):
            cls, meth = m.groups()
            full = os.path.join(root, KNOWN[cls])
            if defines(full, cls) and defines(full, meth):
                ok += 1
            else:
                bad += 1
                failures.append(f"missing method: {cls}.{meth} in {KNOWN[cls]}")
    print(f"sglang commit {commit}: {ok} citations OK, {bad} missing")
    for x in failures:
        print("  " + x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
