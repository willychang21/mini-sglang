"""Interleaved A/B import timing for two SGLang checkouts.

Each sample runs a fresh interpreter. The timer starts after `import torch`, so
the number is the part of the import attributable to SGLang. Base and patched
samples are interleaved so machine drift hits both sides equally.

    python ab_import_time.py --base /path/sglang-base/python \
        --patched /path/sglang/python -n 9 \
        sglang.srt.entrypoints.systemone.serving \
        sglang.srt.managers.tokenizer_manager \
        sglang.srt.entrypoints.http_server
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

PROBES = ("sglang.srt.layers.quantization", "torch._inductor.compile_fx")

CODE = (
    "import json, sys, time, warnings\n"
    "warnings.simplefilter('ignore')\n"
    "import torch\n"
    "t = time.perf_counter()\n"
    "import {module}\n"
    "d = time.perf_counter() - t\n"
    "print(json.dumps([d, [p in sys.modules for p in {probes!r}]]))\n"
)


def sample(pythonpath: str, module: str):
    env = dict(os.environ, PYTHONPATH=pythonpath)
    out = subprocess.run(
        [sys.executable, "-c", CODE.format(module=module, probes=PROBES)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("-n", type=int, default=9)
    ap.add_argument("modules", nargs="+")
    args = ap.parse_args()
    print(f"| module | base | patched | delta | {', '.join(PROBES)} loaded (base -> patched) |")
    print("|---|---|---|---|---|")
    for module in args.modules:
        times = {"base": [], "patched": []}
        loaded = {}
        for _ in range(args.n):
            for side in ("base", "patched"):
                d, flags = sample(getattr(args, side), module)
                times[side].append(d)
                loaded[side] = flags
        b = statistics.median(times["base"])
        a = statistics.median(times["patched"])
        print(
            f"| `{module}` | {b * 1e3:.0f} ms | {a * 1e3:.0f} ms | "
            f"{(a - b) * 1e3:+.0f} ms ({(a - b) / b * 100:+.0f}%) | "
            f"{loaded['base']} -> {loaded['patched']} |"
        )


if __name__ == "__main__":
    main()
