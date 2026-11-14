"""The scheduling loop: receive -> schedule -> execute -> process (Chapter 6).

``normal_loop`` is one iteration of continuous batching. Each step runs either a
prefill batch (preferred) or a decode batch; requests join and leave the active
set between steps, so a finished request's slot is reused immediately.

``overlap_loop`` (SGLang ``event_loop_overlap``) runs the same steps, but the
CPU work for the *next* batch (process N, receive, schedule N+2) runs while
batch N+1 is in the forward pass::

    normal :  [sched N][forward N][process N][sched N+1][forward N+1][process N+1]
    overlap:  forward thread:  [forward N  ][forward N+1      ][forward N+2 ]
              CPU thread:      [proc N-1, sched N+1][proc N, sched N+2]...

On CUDA, the scheduler thread would just enqueue the forward on a stream and
keep going. On CPU there is no asynchronous device, so the two halves need two
threads. The forward stays on the *main* thread and the scheduling work moves
to a helper thread pinned to one OpenMP thread: the main thread already owns
the OpenMP team that ran model loading, and a second thread issuing parallel
torch ops would create a second team and make every parallel region slower.
See ``_mark_launched`` / ``_run_forward`` / ``_process_results`` for the
invariants that make this safe.
"""

from __future__ import annotations

import copy
import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import torch

from ..config import EngineConfig
from ..core import Batch, ChunkedReq, Req
from ..engine import BatchSamplingArgs, Engine, ForwardOutput
from ..message import AbortBackendMsg, BaseBackendMsg, DetokenizeMsg, ExitMsg, UserMsg
from ..utils import div_ceil, init_logger
from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIO
from .prefill import PendingReq, PrefillManager
from .retract import NewTokenRatioTracker, retraction_order
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
class LaunchedBatch:
    """Overlap mode: a batch whose bookkeeping is done (``_mark_launched``).

    ``output`` is None while it waits for / runs in the forward pass, and set
    once the forward has finished; its results are processed after that.
    """

    forward_input: ForwardInput
    seq: int  # launch order
    output: ForwardOutput | None = None


def _limit_openmp_to_one_thread() -> None:
    """Thread initializer for the scheduler's CPU thread (best effort).

    ``torch.set_num_threads`` is process-wide; libgomp's ``omp_set_num_threads``
    only affects the calling thread. Touch ATen first so its per-thread lazy
    init (which resets the count) has already run."""
    torch.ones(16).sum()  # small: below the parallel grain size, so no team is created
    try:
        with open("/proc/self/maps") as f:  # find the libgomp torch loaded
            libs = [line.split()[-1] for line in f if "libgomp" in line]
    except OSError:  # not Linux
        return
    if libs:
        ctypes.CDLL(libs[0]).omp_set_num_threads(1)


class _AlreadyAdvanced:
    """Overlap mode: a request seen by ``Engine.forward_batch`` whose
    ``complete_one`` was already done at scheduling time. Everything else is
    delegated read-only to the real request."""

    __slots__ = ("_req",)

    def __init__(self, req: Req) -> None:
        self._req = req

    def complete_one(self) -> None:
        pass

    def __getattr__(self, name: str):
        return getattr(self._req, name)


@dataclass
class SchedulerStats:
    prefill_steps: int = 0
    decode_steps: int = 0
    prefill_tokens: int = 0  # tokens actually computed during prefill
    decode_tokens: int = 0
    finished_reqs: int = 0
    aborted_reqs: int = 0
    retracted_reqs: int = 0  # retraction events (a request can be retracted twice)
    retract_steps: int = 0  # decode steps that had to retract
    # overlap mode: tokens computed for requests that had already finished or
    # been aborted / retracted by the time the batch was processed
    discarded_tokens: int = 0
    batch_sizes: List[int] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        config: EngineConfig,
        io: SchedulerIO,
        engine: Engine | None = None,
        sched_config: SchedulerConfig | None = None,
    ) -> None:
        self.config = config
        self.sched_config = sched_config or SchedulerConfig()
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
        self.new_token_ratio = NewTokenRatioTracker.from_config(self.sched_config)
        # uid -> prompt length, for usage reporting and the retraction policy
        # (a resumed request's "prompt" also contains what it generated so far)
        self.prompt_lens: Dict[int, int] = {}
        self._exit = False

        # ---- overlap scheduling state
        self.overlap = not self.sched_config.disable_overlap
        if self.overlap and config.tp_info.size > 1:
            # On CPU, TP collectives in the forward and the request broadcast
            # share one Gloo group; issuing them from two threads could
            # interleave them differently on each rank.
            logger.warning("overlap scheduling is not supported with tp_size > 1; using normal_loop")
            self.overlap = False
        self._cpu_thread: ThreadPoolExecutor | None = None
        self._ready: LaunchedBatch | None = None  # scheduled, waiting for the forward
        self._done: LaunchedBatch | None = None  # forward finished, not processed
        self._launched = 0  # seq of the last launched batch
        # Requests freed (finished / aborted / retracted) while a batch that
        # contains them may still be in flight: (seq of that batch, req).
        # Their results in that batch are ignored and their table row is only
        # returned once the batch has been processed.
        self._graveyard: List[Tuple[int, Req]] = []
        self._processed = 0  # seq of the last processed batch
        self.finished_reqs: Set[Req] = set()

    # ------------------------------------------------------------------ loop
    @property
    def has_work(self) -> bool:
        return (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._ready is not None
            or self._done is not None
        )

    def step(self) -> None:
        """One iteration of whichever loop is configured."""
        if self.overlap:
            self.overlap_loop()
        else:
            self.normal_loop()

    def normal_loop(self) -> None:
        blocking = not self.has_work  # only wait for messages when idle
        for msg in self.io.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    def overlap_loop(self) -> None:
        """Run batch N+1 here while the CPU thread processes N and schedules N+2."""
        ready, done = self._ready, self._done
        if ready is None:  # nothing to overlap with (start-up / idle)
            self._done = None
            self._ready = self._cpu_step(done, forward_running=False)
            return
        if self._cpu_thread is None:
            self._cpu_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="minisgl-sched-cpu", initializer=_limit_openmp_to_one_thread
            )
        future = self._cpu_thread.submit(self._cpu_step, done, True)
        try:
            ready.output = self._run_forward(ready.forward_input)
        finally:
            next_ready = future.result()  # always join before touching state again
        self._done, self._ready = ready, next_ready

    def _cpu_step(self, done: LaunchedBatch | None, forward_running: bool) -> LaunchedBatch | None:
        """Process the last finished batch, take new messages, schedule the next.

        Scheduling sees every request's lengths as if the running batch had
        completed (``_mark_launched`` advanced them), but not its sampled
        tokens: a request that hits EOS in batch N+1 is still scheduled into
        N+2, and that extra token is dropped later (``finished_reqs``).
        """
        if done is not None:
            assert done.output is not None
            self._process_results(done.forward_input.batch, done.output, done.seq)
        idle = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.io.receive_msg(blocking=idle and not forward_running):
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()
        return self._mark_launched(forward_input) if forward_input is not None else None

    def drain(self) -> None:
        """Overlap mode: finish and process every batch that was scheduled."""
        if self._done is not None:
            done, self._done = self._done, None
            assert done.output is not None
            self._process_results(done.forward_input.batch, done.output, done.seq)
        if self._ready is not None:
            ready, self._ready = self._ready, None
            ready.output = self._run_forward(ready.forward_input)
            self._process_results(ready.forward_input.batch, ready.output, ready.seq)

    def run_forever(self) -> None:
        while not self._exit:
            self.step()
        self.drain()
        logger.info("scheduler exiting")

    def run_until_idle(self) -> None:
        """Offline mode: step until every request has finished."""
        while True:
            self.step()
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
            if msg.error is not None:
                return self._reply_error(msg.uid, msg.error)
            if input_len == 0:
                return self._reply_error(msg.uid, "empty prompt")
            # Out-of-range ids would raise IndexError in the embedding inside
            # the scheduler loop and take the whole server down.
            vocab_size = self.engine.model_config.vocab_size
            if min(msg.input_ids) < 0 or max(msg.input_ids) >= vocab_size:
                return self._reply_error(msg.uid, f"token ids must be in [0, {vocab_size})")
            if input_len >= max_seq_len or input_len >= capacity:
                return self._reply_error(
                    msg.uid, f"prompt has {input_len} tokens, limit is {min(max_seq_len, capacity) - 1}"
                )
            output_len = min(msg.sampling_params.max_tokens, max_seq_len - input_len, capacity - input_len)
            if output_len <= 0:
                return self._reply_error(msg.uid, "max_tokens must be positive")
            self.prompt_lens[msg.uid] = input_len
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
                self.prompt_lens.pop(msg.uid, None)
                self.stats.aborted_reqs += 1
                self.io.send_result(
                    [DetokenizeMsg(uid=msg.uid, next_token=-1, finished=True, finish_reason="abort")]
                )
        elif isinstance(msg, ExitMsg):
            self._exit = True
        else:
            raise TypeError(f"unknown message {msg!r}")

    # ------------------------------------------------------------- schedule
    @property
    def optimistic(self) -> bool:
        return self.sched_config.admission == "optimistic"

    def _schedule_next_batch(self) -> ForwardInput | None:
        # Prefill has priority; fall back to decode when no prefill batch fits.
        self.prefill_manager.new_token_ratio = self.new_token_ratio.current
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is None and self.decode_manager.runnable:
            if self.optimistic:
                self._retract_decode_if_needed()
            batch = self.decode_manager.schedule_next_batch()
        return self._prepare_batch(batch) if batch else None

    # -------------------------------------------------------------- retract
    def _decode_fits(self, reqs: List[Req]) -> bool:
        return self.cache_manager.pages_needed(reqs) <= self.cache_manager.available_pages

    def _retract_decode_if_needed(self) -> None:
        """SGLang ``update_running_batch`` + ``ScheduleBatch.retract_decode``.

        If the next decode step's pages do not fit, retract the least valuable
        running requests until they do. Unlike SGLang we may retract the last
        one too: the only things that can still hold memory then are chunked
        prefills, which were admitted with room for themselves, so retracting
        everyone always lets them finish and the queue makes progress.
        """
        running = list(self.decode_manager.running_reqs)
        if self._decode_fits(running):
            self.new_token_ratio.decay_step()
            return
        keep = retraction_order(running, self.prompt_lens)
        victims: List[Req] = []
        retracted: List[PendingReq] = []
        # At least one; then until the rest fits. (No lazy_free_region here:
        # the fit check must see the pages each retraction frees.)
        while keep and (not retracted or not self._decode_fits(keep)):
            victims.append(keep.pop())
            retracted.append(self._retract_one(victims[-1], self._headroom_pages(keep)))
        old = self.new_token_ratio.current
        self.new_token_ratio.current = self.new_token_ratio.estimate_after_retract(keep or victims)
        self.prefill_manager.add_retracted(retracted)
        self.stats.retract_steps += 1
        self.stats.retracted_reqs += len(retracted)
        logger.info(
            "KV cache full: retracted %d request(s), %d left running, new_token_ratio %.3f -> %.3f",
            len(retracted), len(keep), old, self.new_token_ratio.current,
        )

    def _headroom_pages(self, survivors: List[Req]) -> int:
        """Pages the survivors still lack for their next ``retract_decode_steps``
        decode steps (SGLang's ``release_req`` evicts the same headroom). Freeing
        this much right away keeps the next steps from evicting the rest of the
        retracted request's KV out of the radix cache."""
        steps = max(1, self.sched_config.retract_decode_steps)
        want = div_ceil(len(survivors) * steps, self.page_size) + self.cache_manager.pages_needed(survivors)
        return want - self.cache_manager.available_pages

    def _retract_one(self, req: Req, short_pages: int) -> PendingReq:
        """Free a decoding request but keep everything needed to resume it."""
        self.decode_manager.remove_req(req)
        pending = PendingReq(
            uid=req.uid,
            input_ids=req.input_ids,  # prompt + every token generated so far
            sampling_params=req.sampling_params,
            # Same max_device_len as before. Host tokens == device_len in
            # normal_loop; in overlap_loop the token of the in-flight batch is
            # not on the host yet and is dropped (it is re-sampled on resume).
            output_len=req.max_device_len - len(req.input_ids),
            retracted=True,
        )
        # Frees the table row, returns ``short_pages`` tail pages to the free
        # list and hands the rest of [0, cached_len) to the radix cache
        # (unlocked -> evictable). Re-admission matches it back for free unless
        # memory pressure evicts it first.
        self._release_row(req)
        self.cache_manager.release_for_retract(req, short_pages)
        return pending

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
    def _mark_launched(self, forward_input: ForwardInput) -> LaunchedBatch:
        """Overlap mode: the bookkeeping ``Engine.forward_batch`` would do.

        ``forward_batch`` advances each request (``complete_one``) after the
        forward. Here that must happen at scheduling time, on the CPU thread:
        the next ``_schedule_next_batch`` reads those lengths while this
        forward is still running. So the forward later gets a view of the
        batch whose requests ignore ``complete_one`` (``_AlreadyAdvanced``);
        ``Engine.forward_batch`` stays the single entry point (CUDA graphs).
        """
        batch = forward_input.batch
        for req in batch.reqs:
            req.complete_one()
        self.decode_manager.filter_reqs(batch.reqs)
        self._launched += 1
        return LaunchedBatch(forward_input, self._launched)

    def _run_forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """Overlap mode: the forward pass, on the main ("device") thread.

        Forwards run one at a time in launch order, like kernels on one CUDA
        stream. token_pool write ordering: a decode request's input token is
        the one sampled by the *previous* batch and may not have reached the
        host yet. Both the gather of this batch's input ids and the scatter of
        its sampled tokens happen here, so batch N+1 always reads what batch N
        wrote (the role of stream order + SGLang's FutureMap on GPU).
        """
        view = copy.copy(forward_input.batch)
        # Already advanced by _mark_launched. Keep one entry per request so
        # batch.size (CUDA-graph selection / padding) is unchanged.
        view.reqs = [_AlreadyAdvanced(r) for r in forward_input.batch.reqs]
        view.input_ids = self.table_manager.token_pool[forward_input.input_mapping]
        output = self.engine.forward_batch(view, forward_input.sample_args)
        assert output.next_tokens_gpu is not None
        if len(forward_input.output_rows) > 0:
            self.table_manager.token_pool[forward_input.output_mapping] = output.next_tokens_gpu[
                forward_input.output_rows
            ]
        return output

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
        self._process_results(forward_input.batch, output, seq=None)

    def _process_results(self, batch: Batch, output: ForwardOutput, seq: int | None) -> None:
        """Commit sampled tokens, finish requests, send replies.

        In overlap mode the requests' lengths have already moved past this
        batch (the next one is launched), but host tokens have not: for a
        request still in flight, ``len(input_ids) == cached_len`` after the
        append below, which is what ``cache_req`` needs.
        """
        assert output.next_tokens_cpu is not None
        next_tokens = output.next_tokens_cpu.tolist()
        replies: List[DetokenizeMsg] = []
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue  # an intermediate chunk never commits a token
                if req in self.finished_reqs:
                    # Finished in the previous batch (EOS was not known when
                    # this one was scheduled), aborted or retracted while this
                    # batch was in flight: its token here is not real.
                    self.stats.discarded_tokens += 1
                    continue
                token = int(next_tokens[i])
                req.append_host(token)
                # Judge the length limit by *host* tokens: in overlap mode the
                # request's device_len already counts the next, in-flight batch.
                finished, reason = len(req.input_ids) >= req.max_device_len, "length"
                if not req.sampling_params.ignore_eos and token in self.eos_token_ids:
                    finished, reason = True, "stop"
                replies.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=token,
                        finished=finished,
                        finish_reason=reason if finished else None,
                        num_prompt_tokens=self.prompt_lens[req.uid],
                    )
                )
                if finished:
                    self.prompt_lens.pop(req.uid, None)
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    self.stats.finished_reqs += 1
                elif batch.is_prefill:
                    # prefill done: publish the prompt to the prefix cache
                    self.cache_manager.cache_req(req, finished=False)
        self.io.send_result(replies)
        if seq is not None:
            self._bury_until(seq)

    def _release_row(self, req: Req) -> None:
        """Return a request's table row now, or once the batch that may still
        use it has been processed. Freeing the row early would let the next
        admission write a new prompt into ``token_pool[row]`` while the
        in-flight forward still scatters this request's token into it."""
        if self.overlap and self._launched > 0 and self._has_unprocessed_batch():
            self._graveyard.append((self._launched, req))
            self.finished_reqs.add(req)
        else:
            self.table_manager.free(req.table_idx)

    def _has_unprocessed_batch(self) -> bool:
        # ``_launched`` is the newest batch whose bookkeeping is done; it may
        # still be waiting for or running its forward.
        return self._processed < self._launched

    def _bury_until(self, seq: int) -> None:
        """Batch ``seq`` is processed: rows freed while it was in flight are safe."""
        self._processed = seq
        keep: List[Tuple[int, Req]] = []
        for tag, req in self._graveyard:
            if tag <= seq:
                self.table_manager.free(req.table_idx)
                self.finished_reqs.discard(req)
            else:
                keep.append((tag, req))
        self._graveyard = keep

    def _free_req_resources(self, req: Req) -> None:
        # KV pages can be freed (or cached) right away even if a forward that
        # reads them is in flight: the worker runs forwards in launch order,
        # so a page handed to a new request is only written by a later batch.
        self._release_row(req)
        self.cache_manager.cache_req(req, finished=True)

    def shutdown(self) -> None:
        self.drain()
        if self._cpu_thread is not None:
            self._cpu_thread.shutdown(wait=True)
        self.io.close()
        self.engine.shutdown()
