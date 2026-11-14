"""Command-line arguments for ``python -m minisgl``."""

from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from dataclasses import dataclass, field

from ..config import EngineConfig
from ..distributed import DistributedInfo
from ..scheduler.config import SchedulerConfig


@dataclass
class ServerArgs:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 1919
    tp_size: int = 1
    num_tokenizers: int = 1
    device: str | None = None
    dtype: str | None = None
    page_size: int = 1
    max_total_tokens: int = 32768
    max_running_req: int = 64
    max_seq_len: int = 4096
    prefill_budget: int = 2048
    cache_type: str = "radix"
    admission: str = "conservative"
    disable_overlap: bool = True
    max_buffered_bytes: int = 1 << 20
    client_stall_timeout: float = 60.0
    seed: int = 0
    dist_port: int = 29517
    ipc_dir: str = field(default_factory=lambda: tempfile.gettempdir())
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def _ipc(self, name: str) -> str:
        return f"ipc://{os.path.join(self.ipc_dir, f'minisgl-{self.run_id}-{name}')}"

    # ZMQ endpoints: frontend -> tokenizers -> scheduler -> detokenizer -> frontend
    def tokenizer_addr(self, i: int) -> str:
        """One endpoint per tokenizer worker: the frontend routes uid -> uid % N."""
        return self._ipc(f"tokenizer-{i}")

    @property
    def tokenizer_addrs(self) -> list:
        return [self.tokenizer_addr(i) for i in range(self.num_tokenizers)]

    @property
    def scheduler_addr(self) -> str:
        return self._ipc("scheduler")

    @property
    def detokenizer_addr(self) -> str:
        return self._ipc("detokenizer")

    @property
    def frontend_addr(self) -> str:
        return self._ipc("frontend")

    def engine_config(self, rank: int = 0) -> EngineConfig:
        kwargs = dict(
            model_path=self.model_path,
            tp_info=DistributedInfo(rank, self.tp_size),
            page_size=self.page_size,
            max_total_tokens=self.max_total_tokens,
            max_running_req=self.max_running_req,
            max_seq_len=self.max_seq_len,
            prefill_budget=self.prefill_budget,
            cache_type=self.cache_type,
            seed=self.seed,
            dist_init_addr=f"tcp://127.0.0.1:{self.dist_port}",
        )
        if self.device:
            kwargs["device"] = self.device
        if self.dtype:
            kwargs["dtype"] = self.dtype
        return EngineConfig(**kwargs)


    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(admission=self.admission, disable_overlap=self.disable_overlap)  # type: ignore[arg-type]


def parse_args(argv=None) -> ServerArgs:
    p = argparse.ArgumentParser(prog="python -m minisgl", description="mini-sglang server")
    p.add_argument("--model", "--model-path", dest="model_path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--tp-size", "--tensor-parallel-size", dest="tp_size", type=int, default=1)
    p.add_argument("--num-tokenizers", "--num-tokenizer", dest="num_tokenizers", type=int, default=1)
    p.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    p.add_argument("--dtype", default=None, help="float32 / bfloat16 / float16")
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--max-total-tokens", type=int, default=32768, help="KV cache capacity in tokens")
    p.add_argument("--max-running-req", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--prefill-budget", type=int, default=2048, help="max prefill tokens per step")
    p.add_argument("--cache-type", choices=["radix", "naive"], default="radix")
    p.add_argument(
        "--admission",
        choices=["conservative", "optimistic"],
        default="conservative",
        help="conservative: reserve each request's full max_tokens of KV (never preempts); "
        "optimistic: reserve max_tokens * new_token_ratio and retract decode requests on KV exhaustion",
    )
    overlap = p.add_mutually_exclusive_group()
    overlap.add_argument(
        "--enable-overlap",
        dest="disable_overlap",
        action="store_false",
        help="overlap scheduling: process batch N and schedule N+2 while batch N+1 runs "
        "(off by default: on CPU it has no spare cores to overlap onto)",
    )
    overlap.add_argument(
        "--disable-overlap", dest="disable_overlap", action="store_true", help="use normal_loop (default)"
    )
    p.set_defaults(disable_overlap=True)
    p.add_argument(
        "--max-buffered-bytes",
        type=int,
        default=1 << 20,
        help="backpressure: abort a request whose client has more than this much undelivered text (0 = off)",
    )
    p.add_argument(
        "--client-stall-timeout",
        type=float,
        default=60.0,
        help="backpressure: abort a request whose replies sat unread for this many seconds (0 = off)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dist-port", type=int, default=29517)
    args = p.parse_args(argv)
    return ServerArgs(**vars(args))
