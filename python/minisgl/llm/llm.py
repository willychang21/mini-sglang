"""Offline Python API: the whole pipeline in one process, no HTTP.

    llm = LLM("Qwen/Qwen3-0.6B")
    out = llm.generate(["What is KV cache?"], SamplingParams(max_tokens=32))

Requests are tokenized, pushed into the scheduler's inbox, and the scheduler
loop runs until every request has finished. It is the same continuous-batching
code path the server uses, only the IO differs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Sequence

from ..config import EngineConfig
from ..core import SamplingParams
from ..message import AbortBackendMsg, UserMsg
from ..scheduler import OfflineIO, Scheduler, SchedulerConfig
from ..tokenizer import DetokenizeManager, TokenizeManager, load_tokenizer


@dataclass
class GenerateOutput:
    text: str
    token_ids: List[int]
    prompt_tokens: int
    finish_reason: str | None
    error: str | None = None
    # timing (seconds since submission)
    ttft: float | None = None
    token_times: List[float] = field(default_factory=list)


class LLM:
    def __init__(self, model_path: str, load_tokenizer_: bool = True, **kwargs: Any) -> None:
        # scheduler knobs (admission=..., disable_overlap=...) vs engine knobs
        self.sched_config, engine_kwargs = SchedulerConfig.split_kwargs(kwargs)
        self.config = EngineConfig(model_path=model_path, **engine_kwargs)
        self.io = OfflineIO()
        self.scheduler = Scheduler(self.config, self.io, sched_config=self.sched_config)
        self.tokenizer = load_tokenizer(model_path) if load_tokenizer_ else None
        self._uid = 0

    @property
    def engine(self):
        return self.scheduler.engine

    def _new_uid(self) -> int:
        self._uid += 1
        return self._uid

    def tokenize(self, prompt: str | List[Dict[str, Any]] | List[int]) -> List[int]:
        if isinstance(prompt, list) and (not prompt or isinstance(prompt[0], int)):
            return list(prompt)  # token ids
        assert self.tokenizer is not None, "pass token ids or load a tokenizer"
        return TokenizeManager(self.tokenizer).tokenize_one(prompt)

    def generate(
        self,
        prompts: Sequence[str | List[Dict[str, Any]] | List[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        abort_after: Dict[int, int] | None = None,
    ) -> List[GenerateOutput]:
        """Generate for all prompts with continuous batching.

        ``abort_after`` (testing aid) maps a prompt index to a number of output
        tokens after which that request is cancelled.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)

        uids: List[int] = []
        start = time.perf_counter()
        for prompt, sp in zip(prompts, sampling_params):
            uid = self._new_uid()
            uids.append(uid)
            self.io.put(UserMsg(uid=uid, input_ids=self.tokenize(prompt), sampling_params=replace(sp)))

        results: Dict[int, GenerateOutput] = {
            uid: GenerateOutput("", [], 0, None) for uid in uids
        }
        abort_at = {uids[i]: n for i, n in (abort_after or {}).items()}
        detok = DetokenizeManager(self.tokenizer) if self.tokenizer is not None else None
        done: set[int] = set()
        while len(done) < len(uids):
            self.scheduler.step()
            replies, self.io.outbox = self.io.outbox, []
            now = time.perf_counter() - start
            for msg in replies:
                if msg.uid not in results:
                    continue
                out = results[msg.uid]
                if msg.next_token >= 0:
                    out.token_ids.append(msg.next_token)
                    out.token_times.append(now)
                    if out.ttft is None:
                        out.ttft = now
                out.prompt_tokens = msg.num_prompt_tokens or out.prompt_tokens
                if msg.finished:
                    out.finish_reason = msg.finish_reason
                    out.error = msg.error
                    done.add(msg.uid)
                elif msg.uid in abort_at and len(out.token_ids) >= abort_at[msg.uid]:
                    self.io.put(AbortBackendMsg(uid=msg.uid))
                    abort_at.pop(msg.uid)
            if detok is not None:
                for reply in detok.detokenize(replies):
                    if reply.uid in results:
                        results[reply.uid].text += reply.incremental_output
            if not self.scheduler.has_work and not self.io.has_pending() and len(done) < len(uids):
                raise RuntimeError("scheduler went idle with unfinished requests")
        # overlap mode: the last launched batch may only hold requests that
        # already finished; process it so their table rows are returned
        self.scheduler.drain()
        return [results[uid] for uid in uids]

    def shutdown(self) -> None:
        self.scheduler.shutdown()
