"""The scheduling loop: receive -> schedule -> execute -> process (Chapter 6).

``normal_loop`` is one iteration of continuous batching. Each step runs either a
prefill batch (preferred) or a decode batch; requests join and leave the active
set between steps, so a finished request's slot is reused immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from ..config import EngineConfig
from ..core import Batch, ChunkedReq, Req
from ..engine import BatchSamplingArgs, Engine, ForwardOutput
from ..message import AbortBackendMsg, BaseBackendMsg, DetokenizeMsg, ExitMsg, UserMsg
from ..utils import init_logger
from .cache import CacheManager
from .decode import DecodeManager
from .io import SchedulerIO
from .prefill import PendingReq, PrefillManager
from .table import TableManager

logger = init_logger(__name__)


@dataclass
class ForwardInput:
    batch: Batch
    sample_args: BatchSamplingArgs
    input_mapping: Tuple[torch.Tensor, torch.Tensor]  # (row, position) of each input token
    output_mapping: Tuple[torch.Tensor, torch.Tensor]  # where sampled tokens are stored
    output_rows: torch.Tensor  # which batch rows produce a real token


@dataclass
class SchedulerStats:
    prefill_steps: int = 0
    decode_steps: int = 0
    prefill_tokens: int = 0  # tokens actually computed during prefill
    decode_tokens: int = 0
    finished_reqs: int = 0
    aborted_reqs: int = 0
    batch_sizes: List[int] = field(default_factory=list)


class Scheduler:
    def __init__(self, config: EngineConfig, io: SchedulerIO, engine: Engine | None = None) -> None:
        self.config = config
        self.engine = engine or Engine(config)
        self.io = io
        self.device = self.engine.device
        self.eos_token_ids = set(self.engine.model_config.eos_token_ids)
        self.prefill_budget = config.prefill_budget
        self.page_size = self.engine.page_size

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.device,
            self.engine.num_pages,
            self.page_size,
            self.engine.page_table,
            config.cache_type,
        )
        self.decode_manager = DecodeManager(self.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.stats = SchedulerStats()
        self._exit = False

    # ------------------------------------------------------------------ loop
    @property
    def has_work(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def normal_loop(self) -> None:
        blocking = not self.has_work  # only wait for messages when idle
        for msg in self.io.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    def run_forever(self) -> None:
        while not self._exit:
            self.normal_loop()
        logger.info("scheduler exiting")

    def run_until_idle(self) -> None:
        """Offline mode: step until every request has finished."""
        while True:
            self.normal_loop()
            if not self.has_work and not self.io.has_pending():
                break

    # -------------------------------------------------------------- messages
    def _reply_error(self, uid: int, error: str) -> None:
        logger.warning("request %d rejected: %s", uid, error)
        self.io.send_result(
            [DetokenizeMsg(uid=uid, next_token=-1, finished=True, finish_reason="error", error=error)]
        )

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, UserMsg):
            input_len = len(msg.input_ids)
            max_seq_len = self.engine.max_seq_len
            capacity = self.engine.num_pages * self.page_size - 2 * self.page_size
            if input_len == 0:
                return self._reply_error(msg.uid, "empty prompt")
            if input_len >= max_seq_len or input_len >= capacity:
                return self._reply_error(
                    msg.uid, f"prompt has {input_len} tokens, limit is {min(max_seq_len, capacity) - 1}"
                )
            output_len = min(msg.sampling_params.max_tokens, max_seq_len - input_len, capacity - input_len)
            if output_len <= 0:
                return self._reply_error(msg.uid, "max_tokens must be positive")
            self.prefill_manager.add_one_req(
                PendingReq(
                    uid=msg.uid,
                    input_ids=torch.tensor(msg.input_ids, dtype=torch.int64),
                    sampling_params=msg.sampling_params,
                    output_len=output_len,
                )
            )
        elif isinstance(msg, AbortBackendMsg):
            found, req_to_free = self.prefill_manager.abort_req(msg.uid)
            if not found:
                req_to_free = self.decode_manager.abort_req(msg.uid)
                found = req_to_free is not None
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
            if found:
                self.stats.aborted_reqs += 1
                self.io.send_result(
                    [DetokenizeMsg(uid=msg.uid, next_token=-1, finished=True, finish_reason="abort")]
                )
        elif isinstance(msg, ExitMsg):
            self._exit = True
        else:
            raise TypeError(f"unknown message {msg!r}")

    # ------------------------------------------------------------- schedule
    def _schedule_next_batch(self) -> ForwardInput | None:
        # Prefill has priority; fall back to decode when no prefill batch fits.
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.cache_manager.allocate_paged(batch.reqs)
        rows, positions = [], []
        for req in batch.reqs:
            positions.append(torch.arange(req.cached_len, req.device_len))
            rows.append(torch.full((req.extend_len,), req.table_idx, dtype=torch.int64))
        input_mapping = (torch.cat(rows).to(self.device), torch.cat(positions).to(self.device))
        batch.positions = input_mapping[1]
        # physical slots where this step's K/V are written
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)

        out_rows = [i for i, r in enumerate(batch.reqs) if not isinstance(r, ChunkedReq)]
        output_mapping = (
            torch.tensor([batch.reqs[i].table_idx for i in out_rows], dtype=torch.int64, device=self.device),
            torch.tensor([batch.reqs[i].device_len for i in out_rows], dtype=torch.int64, device=self.device),
        )
        if batch.is_prefill:
            self.stats.prefill_steps += 1
            self.stats.prefill_tokens += batch.num_tokens
        else:
            self.stats.decode_steps += 1
            self.stats.decode_tokens += batch.num_tokens
        self.stats.batch_sizes.append(batch.size)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            output_rows=torch.tensor(out_rows, dtype=torch.int64, device=self.device),
        )

    # -------------------------------------------------------------- execute
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        batch.input_ids = self.table_manager.token_pool[forward_input.input_mapping]
        output = self.engine.forward_batch(batch, forward_input.sample_args)
        assert output.next_tokens_gpu is not None
        if len(forward_input.output_rows) > 0:
            self.table_manager.token_pool[forward_input.output_mapping] = output.next_tokens_gpu[
                forward_input.output_rows
            ]
        self.decode_manager.filter_reqs(batch.reqs)
        return output

    # -------------------------------------------------------------- process
    def _process_last_data(self, ongoing_data: Tuple[ForwardInput, ForwardOutput] | None) -> None:
        if ongoing_data is None:
            return
        forward_input, output = ongoing_data
        batch = forward_input.batch
        assert output.next_tokens_cpu is not None
        next_tokens = output.next_tokens_cpu.tolist()
        replies: List[DetokenizeMsg] = []
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue  # an intermediate chunk never commits a token
                token = int(next_tokens[i])
                req.append_host(token)
                finished, reason = not req.can_decode, "length"
                if not req.sampling_params.ignore_eos and token in self.eos_token_ids:
                    finished, reason = True, "stop"
                replies.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=token,
                        finished=finished,
                        finish_reason=reason if finished else None,
                        num_prompt_tokens=req.max_device_len - req.output_len,
                    )
                )
                if finished:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    self.stats.finished_reqs += 1
                elif batch.is_prefill:
                    # prefill done: publish the prompt to the prefix cache
                    self.cache_manager.cache_req(req, finished=False)
        self.io.send_result(replies)

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def shutdown(self) -> None:
        self.io.close()
        self.engine.shutdown()
