"""CUDA Graphs for decode (Part II Ch4 "+1 dummy page", mirrors reference ``engine/graph.py``).

Why: a decode step of a small model is hundreds of tiny kernel launches; the
CPU cannot launch them as fast as the GPU runs them. A CUDA Graph records the
whole forward pass once and replays it with a single launch.

The catch: a graph replays *fixed* shapes and *fixed* addresses. So

1. Graphs are captured for a few batch sizes (``determine_graph_bs``). A batch
   of size 5 runs in the graph for 8 (``pad_size``): the 3 extra rows are a
   **dummy request** that owns the reserved page-table row ``max_running_req``
   and writes its K/V into the reserved **dummy page** (the ``num_pages + 1``-th
   page of the pool). Real requests' K/V are never touched by padding rows.
2. Every input lives in a static buffer (``GraphInputBuffers``) and the
   attention backend keeps its decode metadata in static buffers too
   (``init_capture_graph`` / ``capture_metadata`` / ``prepare_for_replay``).
   Before a replay we only *copy* the new step's values into those buffers.

Only decode is captured: prefill shapes vary too much to be worth it.

On CPU there are no CUDA Graphs; ``GraphRunner(..., use_cuda_graph=False)``
"replays" by running the model eagerly on the same static, padded buffers.
Tests use that to check everything except the capture itself.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List

import torch

from ..core import Batch, Context
from ..utils import init_logger

logger = init_logger(__name__)

DUMMY_TOKEN_ID = 0


def determine_graph_bs(
    cuda_graph_bs: List[int] | None, cuda_graph_max_bs: int | None, max_running_req: int
) -> List[int]:
    """Batch sizes to capture: explicit list, or 1, 2, 4, 8, 16, 24, ... up to
    ``cuda_graph_max_bs`` (default 160, like the reference), never more than
    the scheduler can run at once."""
    if cuda_graph_bs is None:
        max_bs = 160 if cuda_graph_max_bs is None else cuda_graph_max_bs
        cuda_graph_bs = [1, 2, 4] + list(range(8, max_bs + 1, 8))
        cuda_graph_bs = [bs for bs in cuda_graph_bs if bs <= max_bs]
    return sorted({bs for bs in cuda_graph_bs if 1 <= bs <= max_running_req})


def pad_size(bs: int, graph_bs: List[int]) -> int | None:
    """Smallest captured size that fits ``bs`` (None: too big, run eagerly)."""
    return next((g for g in graph_bs if g >= bs), None)


@dataclass
class GraphInputBuffers:
    """Static model inputs shared by every captured graph (sliced per size)."""

    input_ids: torch.Tensor  # [max_bs]
    positions: torch.Tensor  # [max_bs]
    out_loc: torch.Tensor  # [max_bs]

    @classmethod
    def create(cls, max_bs: int, device: torch.device, dummy_slot: int) -> "GraphInputBuffers":
        return cls(
            input_ids=torch.full((max_bs,), DUMMY_TOKEN_ID, dtype=torch.int64, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int64, device=device),
            out_loc=torch.full((max_bs,), dummy_slot, dtype=torch.int64, device=device),
        )

    def bind(self, batch: Batch, padded_bs: int) -> None:
        """Point ``batch`` at the buffers (used once, when capturing)."""
        batch.input_ids = self.input_ids[:padded_bs]
        batch.positions = self.positions[:padded_bs]
        batch.out_loc = self.out_loc[:padded_bs]

    def copy_from(self, batch: Batch, padded_bs: int, dummy_slot: int) -> None:
        """Real rows ``[0, bs)`` from the batch, dummy rows ``[bs, padded_bs)``:
        token 0 at position 0, K/V written to the dummy page."""
        bs = batch.size
        self.input_ids[:bs].copy_(batch.input_ids)
        self.positions[:bs].copy_(batch.positions)
        self.out_loc[:bs].copy_(batch.out_loc)
        self.input_ids[bs:padded_bs].fill_(DUMMY_TOKEN_ID)
        self.positions[bs:padded_bs].fill_(0)
        self.out_loc[bs:padded_bs].fill_(dummy_slot)


class GraphRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        ctx: Context,
        graph_bs: List[int],
        dummy_row: int,
        dummy_slot: int,
        device: torch.device,
        use_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.ctx = ctx
        self.attn_backend = ctx.attn_backend
        self.graph_bs = sorted(graph_bs)
        self.max_bs = self.graph_bs[-1] if self.graph_bs else 0
        self.dummy_row = dummy_row
        self.dummy_slot = dummy_slot
        self.use_cuda_graph = use_cuda_graph
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_batches: Dict[int, Batch] = {}
        self.static_logits: Dict[int, torch.Tensor] = {}
        if self.max_bs == 0:
            return
        assert getattr(self.attn_backend, "supports_cuda_graph", False), (
            f"{type(self.attn_backend).__name__} cannot run inside a CUDA Graph"
        )
        self.attn_backend.init_capture_graph(self.max_bs, dummy_row)
        self.buffers = GraphInputBuffers.create(self.max_bs, device, dummy_slot)
        for bs in self.graph_bs:
            batch = Batch(reqs=[], phase="decode")  # reqs are never read by the model
            self.buffers.bind(batch, bs)
            batch.attn_metadata = self.attn_backend.capture_metadata(bs)
            self.static_batches[bs] = batch
        if use_cuda_graph:
            self._capture()

    @torch.inference_mode()
    def _capture(self) -> None:
        # UNTESTED in CI (needs a GPU). Largest first so smaller graphs fit in
        # the memory pool the first one allocated.
        logger.info("Capturing CUDA graphs for decode batch sizes %s", self.graph_bs)
        torch.cuda.synchronize()
        pool = None
        for bs in reversed(self.graph_bs):
            batch = self.static_batches[bs]
            with self.ctx.forward_batch(batch):
                self.model.forward()  # warm-up: Triton JIT/autotune must not happen in capture
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self.static_logits[bs] = self.model.forward()
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph
        torch.cuda.synchronize()

    def can_use(self, batch: Batch) -> bool:
        return batch.is_decode and not batch.return_all_logits and 0 < batch.size <= self.max_bs

    @torch.inference_mode()
    def replay(self, batch: Batch) -> torch.Tensor:
        """Run decode for ``batch`` in the graph of size ``pad_size(batch.size)``.
        ``batch.attn_metadata`` must already be prepared (by the scheduler)."""
        assert self.can_use(batch)
        bs = batch.size
        padded_bs = pad_size(bs, self.graph_bs)
        assert padded_bs is not None
        self.buffers.copy_from(batch, padded_bs, self.dummy_slot)
        self.attn_backend.prepare_for_replay(batch.attn_metadata, bs, padded_bs, self.dummy_row)
        if self.use_cuda_graph:
            self.graphs[padded_bs].replay()
            logits = self.static_logits[padded_bs]
        else:  # CPU stand-in: same buffers, same padding, eager execution
            with self.ctx.forward_batch(self.static_batches[padded_bs]):
                logits = self.model.forward()
        return logits[:bs]

    def destroy(self) -> None:
        # Must run before NCCL is torn down, or freeing the graphs can hang.
        self.graphs.clear()
        self.static_logits.clear()
        gc.collect()
