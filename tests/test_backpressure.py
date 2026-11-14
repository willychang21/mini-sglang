"""Chapter 5 Section 5.4: slow / stuck clients must not grow frontend memory.

These drive ``FrontendManager`` directly: replies are injected with
``dispatch`` (what ``listen`` does for every message from the detokenizer) and
messages sent upstream are recorded instead of going to a ZMQ socket.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import List

from minisgl.message import AbortMsg, BaseMsg, UserReply
from minisgl.server.api_server import FrontendManager


def _frontend(**kw) -> tuple[FrontendManager, List[BaseMsg]]:
    front = FrontendManager("test", ["unused"], "unused", **kw)
    sent: List[BaseMsg] = []

    async def record(msg: BaseMsg) -> None:
        sent.append(msg)

    front.send_one = record  # type: ignore[method-assign]
    return front, sent


def _reply(uid: int, text: str, finished: bool = False, n: int = 0) -> UserReply:
    return UserReply(uid, text, finished, "length" if finished else None, 3, n)


def test_slow_consumer_replies_are_coalesced():
    async def run():
        front, sent = _frontend()
        uid = front.new_user()
        for i in range(1000):  # nobody is reading
            await front.dispatch(_reply(uid, "a", n=i + 1))
        await front.dispatch(_reply(uid, "b", finished=True, n=1001))
        assert len(front.ack_map[uid]) == front.coalesce_after == 64  # bounded, not 1001
        assert front._buffered_bytes[uid] == 1001
        out = [r async for r in front.wait_for_ack(uid)]
        assert [r.incremental_output for r in out[:63]] == ["a"] * 63  # first replies kept as-is
        assert "".join(r.incremental_output for r in out) == "a" * 1000 + "b"  # nothing lost
        assert out[-1].finished and out[-1].num_output_tokens == 1001 and out[-1].num_prompt_tokens == 3
        assert not sent and front.ack_map == {} and front.event_map == {}

    asyncio.run(run())


def test_buffer_cap_aborts_request():
    async def run():
        front, sent = _frontend(max_buffered_bytes=100, stall_timeout=0)
        uid = front.new_user()
        for _ in range(300):
            await front.dispatch(_reply(uid, "x"))
        assert [type(m) for m in sent] == [AbortMsg] and sent[0].uid == uid
        assert front.num_backpressure_aborts == 1
        assert front._buffered_bytes[uid] == 0  # buffer dropped, later replies ignored
        out = [r async for r in front.wait_for_ack(uid)]
        assert len(out) == 1 and out[0].finished and out[0].finish_reason == "abort"
        assert "client too slow" in out[0].error
        assert front.ack_map == {} and front.event_map == {}

    asyncio.run(run())


def test_fast_consumer_is_never_cut_off():
    async def run():
        front, sent = _frontend(max_buffered_bytes=10, stall_timeout=0.2)
        reaper = asyncio.create_task(front.reap_stalled())
        uid = front.new_user()

        async def produce():
            for i in range(200):
                await front.dispatch(_reply(uid, "xyz", finished=i == 199))
                await asyncio.sleep(0.001)

        async def consume():
            text = ""
            async with aclosing(front.wait_for_ack(uid)) as replies:
                async for r in replies:
                    text += r.incremental_output
            return text

        _, text = await asyncio.gather(produce(), consume())
        reaper.cancel()
        assert text == "xyz" * 200 and not sent and front.num_backpressure_aborts == 0

    asyncio.run(run())


def test_stalled_consumer_times_out_and_state_is_reaped():
    async def run():
        front, sent = _frontend(stall_timeout=0.2)
        reaper = asyncio.create_task(front.reap_stalled())
        uid = front.new_user()
        stream = front.wait_for_ack(uid)
        await front.dispatch(_reply(uid, "first"))
        assert (await stream.__anext__()).incremental_output == "first"
        # The client stops reading (e.g. its TCP window is full) while the
        # backend keeps producing.
        for _ in range(5):
            await front.dispatch(_reply(uid, "more"))
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        assert [type(m) for m in sent] == [AbortMsg] and sent[0].uid == uid
        await asyncio.sleep(0.5)  # the consumer never came back: entry dropped
        assert uid not in front.ack_map and uid not in front.event_map
        # If it does come back, it gets a terminal error instead of hanging.
        last = await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert last.finished and last.error
        await stream.aclose()
        reaper.cancel()
        assert front.ack_map == {} and front.num_backpressure_aborts == 1

    asyncio.run(run())


def test_idle_request_waiting_for_tokens_is_not_stalled():
    """No pending replies means the *backend* is slow, not the client."""

    async def run():
        front, sent = _frontend(stall_timeout=0.1)
        reaper = asyncio.create_task(front.reap_stalled())
        uid = front.new_user()
        await asyncio.sleep(0.4)  # e.g. long queueing / prefill
        await front.dispatch(_reply(uid, "done", finished=True))
        out = [r async for r in front.wait_for_ack(uid)]
        reaper.cancel()
        assert out[0].incremental_output == "done" and not sent

    asyncio.run(run())
