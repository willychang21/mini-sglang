"""Data structures shared by every module (Chapter 1, Section 3.3).

``SamplingParams``  how to pick tokens for one request
``Req``             one running request: its tokens, lengths, cache handle
``Batch``           the requests run together in one forward pass
``Context``         per-process runtime (KV pool, page table, attention backend,
                    the batch currently being executed)

Length bookkeeping for a request (Chapter 4, Section 5)::

    [0, cached_len)            K/V already in the KV pool (reusable)
    [cached_len, device_len)   tokens whose K/V is computed in this step
    extend_len = device_len - cached_len
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

import torch

if TYPE_CHECKING:
    from .attention.base import BaseAttnBackend, BaseAttnMetadata
    from .kvcache.base import BaseCacheHandle, BaseKVCachePool


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 means greedy
    top_k: int = -1  # -1 disables top-k
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 128

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0 or self.top_k == 1


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # host (CPU) copy of prompt + generated tokens
    table_idx: int  # row in token_pool / page_table
    cached_len: int
    output_len: int  # remaining output budget when the request was admitted
    uid: int
    sampling_params: SamplingParams
    cache_handle: "BaseCacheHandle"
    output_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def complete_one(self) -> None:
        """Called after a forward pass: everything computed is now cached and the
        sampled token occupies the next position."""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, token: int) -> None:
        self.input_ids = torch.cat([self.input_ids, self.input_ids.new_tensor([token])])
        self.output_ids.append(token)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(uid={self.uid}, table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


class ChunkedReq(Req):
    """A request whose prompt is only partially prefilled (chunked prefill).

    It must never sample: the logits of an intermediate chunk do not predict a
    real output token (Chapter 6, Section 4.2).
    """

    @property
    def can_decode(self) -> bool:
        return False


# "verify": speculative verification (SGLang ``ForwardMode.TARGET_VERIFY``). It
# is an extend batch - each request runs ``[last_token, d1..dk]`` on top of its
# cached prefix - that needs logits for every position (``return_all_logits``).
Phase = Literal["prefill", "decode", "verify"]


@dataclass(eq=False)
class Batch:
    reqs: List[Req]
    phase: Phase
    # When True the LM head returns logits for every position instead of only
    # the last position of each request (needed by speculative verification).
    return_all_logits: bool = False
    # Filled by Scheduler._prepare_batch / the attention backend.
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    attn_metadata: "BaseAttnMetadata" = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_verify(self) -> bool:
        return self.phase == "verify"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def num_tokens(self) -> int:
        return sum(r.extend_len for r in self.reqs)


@dataclass
class Context:
    page_table: torch.Tensor
    kv_cache: "BaseKVCachePool"
    attn_backend: "BaseAttnBackend"
    _batch: Batch | None = None

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "no batch is being executed"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "nested forward_batch"
        self._batch = batch
        try:
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context | None) -> None:
    global _GLOBAL_CTX
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "global context is not initialized"
    return _GLOBAL_CTX


def any_to_list(x: Any) -> List[int]:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)
