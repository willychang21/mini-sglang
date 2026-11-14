"""Engine flags shared by every benchmark script (GPU path, Part III Ch1).

    --device cuda --dtype bfloat16 --attention-backend triton [--disable-cuda-graph]
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from minisgl.attention import SUPPORTED_ATTENTION_BACKENDS


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="default: float32 on CPU, bfloat16 on CUDA")
    ap.add_argument("--attention-backend", choices=SUPPORTED_ATTENTION_BACKENDS.names(), default=None)
    ap.add_argument("--disable-cuda-graph", action="store_true")


def engine_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        device=args.device,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    if args.dtype:
        kw["dtype"] = args.dtype
    return kw


def describe(args: argparse.Namespace) -> str:
    return (
        f"device={args.device} dtype={args.dtype or 'default'} "
        f"attention_backend={args.attention_backend or 'default'} "
        f"cuda_graph={'off' if args.disable_cuda_graph else 'on (CUDA only)'}"
    )
