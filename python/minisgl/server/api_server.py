"""OpenAI-style HTTP front end (Chapter 5).

The HTTP coroutine never runs the model. It converts the request into a
``TokenizeMsg`` tagged with a fresh ``uid``, sends it to the tokenizer over ZMQ,
and then awaits replies routed back by uid:

* ``ack_map[uid]``   replies that arrived but were not consumed yet (data)
* ``event_map[uid]`` wakes the coroutine waiting for that uid (notification)

One shared ``listen`` coroutine reads every ``UserReply`` from the detokenizer
and dispatches it by uid. Compared with the reference implementation this
version closes the cleanup gaps discussed in Chapter 5 Section 5: per-uid state
is removed in a ``finally`` block, non-streaming requests also detect client
disconnects, and errors are returned instead of leaving the client hanging.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..core import SamplingParams
from ..message import AbortMsg, BaseMsg, TokenizeMsg, UserReply, unwrap_msg
from ..utils import init_logger
from ..utils.zmq_utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue

logger = init_logger(__name__)


# ----------------------------------------------------------- request models
class Message(BaseModel):
    role: str
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | List[int] | None = None
    messages: List[Message] | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    stream: bool = False
    chat_template_kwargs: Dict[str, Any] | None = None

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
        )


# ---------------------------------------------------------- frontend state
class FrontendManager:
    """Per-uid state of the API server.

    Routing to tokenizers (multi-tokenizer abort race): with one PUSH socket
    shared by N tokenizers, ZMQ round-robins *messages*, so a request's
    ``TokenizeMsg`` and its later ``AbortMsg`` may go to different tokenizers.
    The abort then overtakes the (slower) tokenization and reaches the scheduler
    first, where it is ignored, and the request runs to completion. Instead we
    keep one PUSH socket per tokenizer and send every message of ``uid`` to
    tokenizer ``uid % N``. A tokenizer handles its inbox in order and ZMQ keeps
    per-socket FIFO order, so per-uid order is preserved end to end
    (frontend -> tokenizer[uid % N] -> scheduler).

    Backpressure (slow or stuck clients, Chapter 5 Section 5.4): the shared
    ``listen`` coroutine appends replies to ``ack_map[uid]`` as fast as the
    backend produces them, whether or not the client reads. We cannot slow
    the scheduler down for one request (it is batched with everyone else),
    so the policy is "bound, then cancel":

    1. *Coalesce*: once ``coalesce_after`` replies are waiting, a new reply is
       merged into the last undelivered one (text is concatenated, the latest
       status wins), so ``ack_map[uid]`` never holds more than that many
       objects and its cost is the undelivered text, not one object per
       token. A slow streaming client simply gets bigger chunks; a client
       that keeps up still sees one reply per token.
    2. *Size cap*: if undelivered text exceeds ``max_buffered_bytes``.
    3. *Stall timeout*: if replies have been waiting for longer than
       ``stall_timeout`` seconds without the consumer taking them.

    On (2) or (3) the request is aborted upstream (``AbortMsg``, which frees
    its KV and slot in the scheduler), further replies are dropped, and the
    buffer is replaced by one terminal error reply. If the consumer never comes
    back, the reaper drops that last entry after another ``stall_timeout``.
    Either limit can be disabled with 0.
    """

    def __init__(
        self,
        model_name: str,
        send_addrs: str | List[str] | None = None,
        recv_addr: str = "",
        max_buffered_bytes: int = 1 << 20,
        stall_timeout: float = 60.0,
        coalesce_after: int = 64,
        send_addr: str | None = None,  # pre-routing spelling: one tokenizer address
    ) -> None:
        self.model_name = model_name
        if send_addrs is None:
            assert send_addr is not None, "send_addrs is required"
            send_addrs = send_addr
        self.send_addrs = [send_addrs] if isinstance(send_addrs, str) else list(send_addrs)
        assert len(self.send_addrs) > 0
        self.recv_addr = recv_addr
        self.uid_counter = 0
        self.ack_map: Dict[int, List[UserReply]] = {}
        self.event_map: Dict[int, asyncio.Event] = {}
        self._send: List[ZmqAsyncPushQueue] = []
        self._recv: ZmqAsyncPullQueue | None = None
        self._listener: asyncio.Task | None = None
        # backpressure bookkeeping
        self.max_buffered_bytes = max_buffered_bytes
        self.stall_timeout = stall_timeout
        self.coalesce_after = max(1, coalesce_after)
        self._buffered_bytes: Dict[int, int] = {}  # undelivered text per uid
        self._last_drain: Dict[int, float] = {}  # when the consumer last took replies
        self._closed: Dict[int, float] = {}  # uid -> time it was cut off
        self._reaper: asyncio.Task | None = None
        self.num_backpressure_aborts = 0

    def bind(self) -> None:
        """Create sockets before the backend starts so workers can connect."""
        self._send = [ZmqAsyncPushQueue(addr, bind=True) for addr in self.send_addrs]
        self._recv = ZmqAsyncPullQueue(self.recv_addr, bind=True)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self._buffered_bytes[uid] = 0
        self._last_drain[uid] = time.monotonic()
        return uid

    def _forget(self, uid: int) -> None:
        for d in (self.ack_map, self.event_map, self._buffered_bytes, self._last_drain, self._closed):
            d.pop(uid, None)  # type: ignore[attr-defined]

    def _create_listener_once(self) -> None:
        loop = asyncio.get_running_loop()
        if self._listener is None or self._listener.done():
            self._listener = loop.create_task(self.listen())
        if self.stall_timeout > 0 and (self._reaper is None or self._reaper.done()):
            self._reaper = loop.create_task(self.reap_stalled())

    def route(self, uid: int) -> int:
        """Index of the tokenizer that owns every message of ``uid``."""
        return uid % len(self.send_addrs)

    async def send_one(self, msg: BaseMsg) -> None:
        self._create_listener_once()
        assert self._send, "call bind() first"
        uid = getattr(msg, "uid")  # TokenizeMsg / AbortMsg are always uid-tagged
        send = self._send[self.route(uid)] if isinstance(self._send, list) else self._send  # a test double
        await send.put(msg)

    async def listen(self) -> None:
        assert self._recv is not None
        while True:
            msg = await self._recv.get()
            for reply in unwrap_msg(msg):
                if isinstance(reply, UserReply):
                    await self.dispatch(reply)

    async def dispatch(self, reply: UserReply) -> None:
        uid = reply.uid
        if uid not in self.ack_map or uid in self._closed:
            return  # e.g. late tokens of a cancelled request
        pending = self.ack_map[uid]
        if len(pending) >= self.coalesce_after:  # consumer far behind: coalesce
            pending[-1] = _merge(pending[-1], reply)
        else:
            pending.append(reply)
        self._buffered_bytes[uid] += len(reply.incremental_output.encode())
        self.event_map[uid].set()
        if 0 < self.max_buffered_bytes < self._buffered_bytes[uid] and not reply.finished:
            await self._cut_off(uid, f"client too slow: more than {self.max_buffered_bytes} bytes undelivered")

    async def _cut_off(self, uid: int, reason: str) -> None:
        """Backpressure abort: free backend resources, end the request with an error."""
        logger.warning("aborting request %d: %s", uid, reason)
        self.num_backpressure_aborts += 1
        self._closed[uid] = time.monotonic()
        self.ack_map[uid] = [UserReply(uid=uid, incremental_output="", finished=True, finish_reason="abort", error=reason)]
        self._buffered_bytes[uid] = 0
        self.event_map[uid].set()
        await self.send_one(AbortMsg(uid=uid))

    async def reap_stalled(self) -> None:
        """Periodically cut off consumers that stopped taking their replies."""
        interval = max(0.01, min(1.0, self.stall_timeout / 4))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for uid in list(self.ack_map):
                if uid in self._closed:
                    if now - self._closed[uid] > self.stall_timeout:
                        self._forget(uid)  # the consumer never came back
                elif self.ack_map[uid] and now - self._last_drain[uid] > self.stall_timeout:
                    await self._cut_off(uid, f"client stalled for more than {self.stall_timeout:g}s")

    async def wait_for_ack(self, uid: int) -> AsyncIterator[UserReply]:
        event = self.event_map[uid]
        try:
            while True:
                await event.wait()  # 1. wait for a notification
                event.clear()  # 2. reset the flag (replies stay in the list)
                if uid not in self.ack_map:  # reaped while we were away
                    yield UserReply(uid, "", True, "abort", error="request was cut off (client stalled)")
                    return
                pending = self.ack_map[uid]  # 3. take the list, leave a fresh one
                self.ack_map[uid] = []
                self._buffered_bytes[uid] = 0
                self._last_drain[uid] = time.monotonic()
                for ack in pending:  # 4. hand out every reply
                    yield ack
                    if ack.finished:
                        return
                    self._last_drain[uid] = time.monotonic()
        finally:  # runs on normal end, early close, and cancellation
            self._forget(uid)

    async def abort_user(self, uid: int) -> None:
        already_cut_off = uid in self._closed
        self._forget(uid)
        if already_cut_off:
            return  # the backend was told when we cut it off
        logger.info("aborting request %d", uid)
        await self.send_one(AbortMsg(uid=uid))

    async def collect(self, uid: int) -> Dict[str, Any]:
        text, last = "", None
        async with aclosing(self.wait_for_ack(uid)) as replies:
            async for reply in replies:
                text += reply.incremental_output
                last = reply
        assert last is not None
        return {"text": text, "reply": last}


def _merge(old: UserReply, new: UserReply) -> UserReply:
    """Two consecutive replies of one request -> one (incremental text appends)."""
    return UserReply(
        uid=new.uid,
        incremental_output=old.incremental_output + new.incremental_output,
        finished=new.finished,
        finish_reason=new.finish_reason,
        num_prompt_tokens=new.num_prompt_tokens or old.num_prompt_tokens,
        num_output_tokens=new.num_output_tokens,
        error=new.error or old.error,
    )


_STATE: FrontendManager | None = None


def get_global_state() -> FrontendManager:
    assert _STATE is not None, "frontend not initialized"
    return _STATE


def set_global_state(state: FrontendManager) -> None:
    global _STATE
    _STATE = state


app = FastAPI(title="mini-sglang")


def _usage(reply: UserReply) -> Dict[str, int]:
    return {
        "prompt_tokens": reply.num_prompt_tokens,
        "completion_tokens": reply.num_output_tokens,
        "total_tokens": reply.num_prompt_tokens + reply.num_output_tokens,
    }


async def _submit(req: OpenAICompletionRequest, chat: bool) -> int:
    state = get_global_state()
    if chat and req.messages:
        prompt: Any = [m.model_dump() for m in req.messages]
    elif req.prompt is not None:
        prompt = req.prompt
    else:
        raise ValueError("either 'messages' or 'prompt' must be provided")
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=req.sampling_params(),
            chat_template_kwargs=req.chat_template_kwargs or {},
        )
    )
    return uid


async def _collect_or_abort(uid: int, request: Request) -> Dict[str, Any] | None:
    """Non-streaming wait that still notices a client disconnect."""
    state = get_global_state()
    task = asyncio.create_task(state.collect(uid))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if done:
                return task.result()
            if await request.is_disconnected():
                logger.info("client disconnected for user %d", uid)
                task.cancel()
                await state.abort_user(uid)
                return None
    except asyncio.CancelledError:
        task.cancel()
        await state.abort_user(uid)
        raise


async def _stream_with_cancellation(gen: AsyncIterator[bytes], request: Request, uid: int):
    state = get_global_state()
    try:
        async with aclosing(gen) as chunks:
            async for chunk in chunks:
                if await request.is_disconnected():
                    logger.info("client disconnected for user %d", uid)
                    raise asyncio.CancelledError
                yield chunk
    except asyncio.CancelledError:
        asyncio.get_running_loop().create_task(state.abort_user(uid))
        raise


def _sse(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


async def _stream(uid: int, chat: bool) -> AsyncIterator[bytes]:
    state = get_global_state()
    created = int(time.time())
    kind = "chat.completion.chunk" if chat else "text_completion"
    base = {"id": f"cmpl-{uid}", "object": kind, "created": created, "model": state.model_name}
    first = True
    finish_reason, last = None, None
    async with aclosing(state.wait_for_ack(uid)) as replies:
        async for reply in replies:
            last = reply
            if reply.error:
                yield _sse({"error": {"message": reply.error, "type": "invalid_request_error"}})
                yield b"data: [DONE]\n\n"
                return
            if chat:
                delta: Dict[str, Any] = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if reply.incremental_output:
                    delta["content"] = reply.incremental_output
                choice = {"index": 0, "delta": delta, "finish_reason": None}
            else:
                choice = {"index": 0, "text": reply.incremental_output, "finish_reason": None}
            if reply.incremental_output or chat and "role" in choice.get("delta", {}):
                yield _sse({**base, "choices": [choice]})
            finish_reason = reply.finish_reason
    end = {"index": 0, "finish_reason": finish_reason or "stop"}
    end.update({"delta": {}} if chat else {"text": ""})
    payload = {**base, "choices": [end]}
    if last is not None:
        payload["usage"] = _usage(last)
    yield _sse(payload)
    yield b"data: [DONE]\n\n"


async def _handle(req: OpenAICompletionRequest, request: Request, chat: bool):
    state = get_global_state()
    try:
        uid = await _submit(req, chat)
    except ValueError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=400)
    if req.stream:
        return StreamingResponse(
            _stream_with_cancellation(_stream(uid, chat), request, uid),
            media_type="text/event-stream",
        )
    result = await _collect_or_abort(uid, request)
    if result is None:
        return JSONResponse({"error": {"message": "client disconnected"}}, status_code=499)
    reply: UserReply = result["reply"]
    if reply.error:
        return JSONResponse({"error": {"message": reply.error}}, status_code=400)
    if chat:
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": reply.finish_reason,
        }
        obj = "chat.completion"
    else:
        choice = {"index": 0, "text": result["text"], "finish_reason": reply.finish_reason}
        obj = "text_completion"
    return {
        "id": f"cmpl-{uid}",
        "object": obj,
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [choice],
        "usage": _usage(reply),
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OpenAICompletionRequest, request: Request):
    return await _handle(req, request, chat=True)


@app.post("/v1/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    return await _handle(req, request, chat=False)


@app.get("/v1/models")
async def v1_models():
    state = get_global_state()
    return {"object": "list", "data": [{"id": state.model_name, "object": "model", "owned_by": "minisgl"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/state")
async def debug_state():
    """Number of per-request entries held by the frontend (should drop to 0)."""
    state = get_global_state()
    return {"ack_map": len(state.ack_map), "event_map": len(state.event_map)}
