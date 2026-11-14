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
    """

    def __init__(self, model_name: str, send_addrs: str | List[str], recv_addr: str) -> None:
        self.model_name = model_name
        self.send_addrs = [send_addrs] if isinstance(send_addrs, str) else list(send_addrs)
        assert len(self.send_addrs) > 0
        self.recv_addr = recv_addr
        self.uid_counter = 0
        self.ack_map: Dict[int, List[UserReply]] = {}
        self.event_map: Dict[int, asyncio.Event] = {}
        self._send: List[ZmqAsyncPushQueue] = []
        self._recv: ZmqAsyncPullQueue | None = None
        self._listener: asyncio.Task | None = None

    def bind(self) -> None:
        """Create sockets before the backend starts so workers can connect."""
        self._send = [ZmqAsyncPushQueue(addr, bind=True) for addr in self.send_addrs]
        self._recv = ZmqAsyncPullQueue(self.recv_addr, bind=True)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    def _create_listener_once(self) -> None:
        if self._listener is None or self._listener.done():
            self._listener = asyncio.get_running_loop().create_task(self.listen())

    def route(self, uid: int) -> int:
        """Index of the tokenizer that owns every message of ``uid``."""
        return uid % len(self.send_addrs)

    async def send_one(self, msg: BaseMsg) -> None:
        self._create_listener_once()
        assert self._send, "call bind() first"
        uid = getattr(msg, "uid")  # TokenizeMsg / AbortMsg are always uid-tagged
        await self._send[self.route(uid)].put(msg)

    async def listen(self) -> None:
        assert self._recv is not None
        while True:
            msg = await self._recv.get()
            for reply in unwrap_msg(msg):
                if not isinstance(reply, UserReply) or reply.uid not in self.ack_map:
                    continue  # e.g. late tokens of a cancelled request
                self.ack_map[reply.uid].append(reply)
                self.event_map[reply.uid].set()

    async def wait_for_ack(self, uid: int) -> AsyncIterator[UserReply]:
        event = self.event_map[uid]
        try:
            while True:
                await event.wait()  # 1. wait for a notification
                event.clear()  # 2. reset the flag (replies stay in the list)
                pending = self.ack_map[uid]  # 3. take the list, leave a fresh one
                self.ack_map[uid] = []
                for ack in pending:  # 4. hand out every reply
                    yield ack
                    if ack.finished:
                        return
        finally:  # runs on normal end, early close, and cancellation
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def abort_user(self, uid: int) -> None:
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
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
