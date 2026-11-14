"""The Engine owns everything on one TP worker: model, KV pool, page table,
attention backend and sampler (Chapters 2, 4, 9).

It does *not* decide which requests run; the Scheduler hands it a prepared
``Batch`` and the Engine runs the forward pass and samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..attention import create_attention_backend
from ..config import EngineConfig
from ..core import Batch, Context, set_global_ctx
from ..distributed import set_tp_info
from ..kvcache import create_kvcache_pool
from ..models import create_model, load_model_weights
from ..utils import div_ceil, init_logger
from .graph import GraphRunner, determine_graph_bs
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    next_tokens_gpu: torch.Tensor | None
    next_tokens_cpu: torch.Tensor | None


def resolve_device(device: str, tp_rank: int, tp_size: int) -> torch.device:
    """``"cuda"`` -> ``cuda:<index>``: ``torch.cuda.set_device`` rejects a CUDA
    device without an index, and with TP each rank owns GPU ``rank``."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return dev
    if tp_size > 1:
        return torch.device("cuda", tp_rank)
    return torch.device("cuda", dev.index if dev.index is not None else torch.cuda.current_device())


class Engine:
    def __init__(self, config: EngineConfig, page_table: torch.Tensor | None = None) -> None:
        """``page_table``: share another engine's table (a speculative *draft*
        engine). Both KV pools then use the same slot numbering, so one
        allocator / radix cache manages both (SGLang: the draft worker shares
        ``req_to_token_pool`` and ``token_to_kv_pool_allocator``)."""
        # TP info must be set before any layer is built (layers size their
        # weights with get_tp_info()).
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.config = config
        self.tp_info = config.tp_info
        self.device = resolve_device(config.device, config.tp_info.rank, config.tp_info.size)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)  # before NCCL init and any allocation
        self.dtype = config.dtype
        self.tp_cpu_group = self._init_communication(config)

        self.model_config = config.model_config
        torch.manual_seed(config.seed)
        self.model = create_model(self.model_config, self.device, self.dtype)
        load_model_weights(self.model, config.model_path, self.model_config)

        # KV pool + page table (Chapter 4, Section 4.3)
        self.page_size = config.page_size
        self.num_pages = config.num_pages or div_ceil(config.max_total_tokens, config.page_size)
        # +1: a reserved *dummy page* (Ch4). CUDA-graph padding rows write their
        # K/V there, so they can never corrupt a real request's cache.
        self.kv_cache = create_kvcache_pool(
            model_config=self.model_config,
            num_pages=self.num_pages + 1,
            page_size=config.page_size,
            dtype=self.dtype,
            device=self.device,
        )
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(
            config.max_seq_len, num_tokens, self.model_config.max_position_embeddings
        )
        # One row per running request; entries are raw *token slots*. The extra
        # last row belongs to the dummy request and points at the dummy page.
        if page_table is not None:
            assert page_table.shape[0] == config.max_running_req + 1
            self.max_seq_len = min(self.max_seq_len, page_table.shape[1])
            self.page_table = page_table
        else:
            self.page_table = torch.zeros(
                (config.max_running_req + 1, self.max_seq_len), dtype=torch.int64, device=self.device
            )
        self.dummy_row = config.max_running_req
        self.dummy_slot = self.num_pages * config.page_size  # first slot of the dummy page
        self.page_table[self.dummy_row].fill_(self.dummy_slot)
        self.attn_backend = create_attention_backend(
            config.attention_backend, self.kv_cache, self.page_table, self.model_config
        )
        self.ctx = Context(
            page_table=self.page_table, kv_cache=self.kv_cache, attn_backend=self.attn_backend
        )
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.device, config.seed)
        self.graph_runner = self._init_graph_runner(config)
        logger.info(
            "Engine ready: %s layers=%d tp=%d/%d device=%s dtype=%s attn=%s kv_tokens=%d (%.1f MiB)",
            self.model_config.architecture,
            self.model_config.num_layers,
            self.tp_info.rank,
            self.tp_info.size,
            self.device,
            self.dtype,
            config.attention_backend,
            num_tokens,
            self.kv_cache.num_bytes / 2**20,
        )

    def _init_graph_runner(self, config: EngineConfig) -> GraphRunner | None:
        if self.device.type != "cuda" or config.disable_cuda_graph:
            return None
        if not self.attn_backend.supports_cuda_graph:
            logger.warning("attention backend %s has no CUDA graph support", config.attention_backend)
            return None
        graph_bs = determine_graph_bs(
            config.cuda_graph_bs, config.cuda_graph_max_bs, config.max_running_req
        )
        if not graph_bs:
            return None
        return GraphRunner(
            self.model, self.ctx, graph_bs, self.dummy_row, self.dummy_slot, self.device
        )

    def _init_communication(self, config: EngineConfig):
        if config.tp_info.size == 1:
            return None
        backend = "nccl" if self.device.type == "cuda" else "gloo"
        kwargs = {"device_id": self.device} if backend == "nccl" else {}
        dist.init_process_group(
            backend=backend,
            init_method=config.dist_init_addr,
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            **kwargs,  # NCCL: bind this rank's communicator to its GPU eagerly
        )
        # Control messages (request broadcast) always travel over a CPU group.
        return dist.group.WORLD if backend == "gloo" else dist.new_group(backend="gloo")

    @torch.inference_mode()
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs | None) -> ForwardOutput:
        set_global_ctx(self.ctx)
        if self.graph_runner is not None and self.graph_runner.can_use(batch):
            logits = self.graph_runner.replay(batch)
        else:
            with self.ctx.forward_batch(batch):
                logits = self.model.forward()
        for req in batch.reqs:
            req.complete_one()
        if args is None:
            return ForwardOutput(logits=logits, next_tokens_gpu=None, next_tokens_cpu=None)
        next_tokens = self.sampler.sample(logits, args)
        return ForwardOutput(
            logits=logits, next_tokens_gpu=next_tokens, next_tokens_cpu=next_tokens.to("cpu")
        )

    def shutdown(self) -> None:
        if self.graph_runner is not None:
            self.graph_runner.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
