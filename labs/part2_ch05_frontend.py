"""Part II Ch5 lab: the API server's per-uid routing, without GPU, ZMQ or HTTP.

What it shows (all on the real ``FrontendManager`` from ``server/api_server.py``)
  1. Routing: the real ``listen()`` coroutine reads from an in-memory stand-in for the
     detokenizer socket. Replies for uids A and B arrive interleaved and in batches
     (several ``event.set()`` before one wake-up); each consumer gets only its own
     replies, in order, and ``ack_map`` / ``event_map`` are empty afterwards.
     A late reply for a finished uid is dropped.
  2. SSE encoding of ``_stream``: role in the first chunk, ``finish_reason``, ``usage``, ``[DONE]``.
  3. Cleanup on early exit: ``aclosing`` runs the ``finally`` of ``wait_for_ack``;
     the reference version (trailing ``del`` statements, no ``finally``) leaks.
  4. Cleanup on cancellation and client disconnect (non-streaming ``_collect_or_abort``
     and streaming ``_stream_with_cancellation``): state removed + ``AbortMsg`` sent.
  5. Why ``async def`` is not enough: ``time.sleep`` inside a coroutine stalls every
     other coroutine; ``await asyncio.sleep`` / ``asyncio.to_thread`` do not.

For the full multi-process server (real ZMQ, tokenizer/detokenizer workers,
HTTP disconnects) see ``tests/test_server.py``.

    python labs/part2_ch05_frontend.py
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import aclosing

from _common import lab_parser, quiet_logs, section, table


class FakeRecv:
    """Stands in for ZmqAsyncPullQueue (detokenizer -> API server)."""

    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()

    async def get(self):
        return await self.q.get()


class FakeSend:
    """Stands in for ZmqAsyncPushQueue (API server -> tokenizer); records what was sent."""

    def __init__(self) -> None:
        self.sent: list = []

    async def put(self, msg) -> None:
        self.sent.append(msg)


class CountingEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.sets = 0
        self.wakeups = 0

    def set(self) -> None:
        self.sets += 1
        super().set()

    async def wait(self):
        r = await super().wait()
        self.wakeups += 1
        return r


class FakeRequest:
    """Minimal starlette Request: reports a disconnect after ``after`` checks."""

    def __init__(self, after: int) -> None:
        self.after, self.calls = after, 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > self.after


def make_frontend():
    from minisgl.server.api_server import FrontendManager, set_global_state

    fm = FrontendManager("tiny", send_addr="unused", recv_addr="unused")  # bind() is never called
    fm._recv, fm._send = FakeRecv(), FakeSend()
    fm._listener = asyncio.get_running_loop().create_task(fm.listen())  # the real listener
    set_global_state(fm)
    return fm


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def reference_wait_for_ack(fm, uid):
    """The reference (commit 9a91cfa) wait_for_ack: cleanup is *after* the loop, not in finally."""
    event = fm.event_map[uid]
    while True:
        await event.wait()
        event.clear()
        pending = fm.ack_map[uid]
        fm.ack_map[uid] = []
        ack = None
        for ack in pending:
            yield ack
        if ack and ack.finished:
            break
    del fm.ack_map[uid]
    del fm.event_map[uid]


async def part1_routing():
    from minisgl.message import BatchMsg, UserReply

    section("1. one listener, many consumers: interleaved replies for A and B")
    fm = make_frontend()
    A, B = fm.new_user(), fm.new_user()
    for uid in (A, B):
        fm.event_map[uid] = CountingEvent()  # same class, plus counters
    events = {A: fm.event_map[A], B: fm.event_map[B]}
    got = {A: [], B: []}

    async def consumer(uid):
        async with aclosing(fm.wait_for_ack(uid)) as replies:
            async for r in replies:
                got[uid].append(r.incremental_output)

    tasks = [asyncio.create_task(consumer(A)), asyncio.create_task(consumer(B))]
    await settle()

    def R(uid, text, fin=False):
        return UserReply(uid=uid, incremental_output=text, finished=fin, finish_reason="stop" if fin else None)

    arrivals = [BatchMsg([R(A, "A1"), R(B, "B1"), R(B, "B2")]),  # one ZMQ message, three replies
                R(A, "A2"),
                BatchMsg([R(B, "B3", True), R(A, "A3", True)])]
    order = []
    for msg in arrivals:
        order.append(" ".join(r.incremental_output for r in getattr(msg, "data", [msg])))
        await fm._recv.q.put(msg)
        await settle()
    await asyncio.gather(*tasks)
    print("arrival order (one row per socket message): " + " | ".join(order))
    print(table([[n, got[u], events[u].sets, events[u].wakeups] for n, u in (("A", A), ("B", B))],
                ["consumer", "received (in order)", "event.set() calls", "wake-ups"]))
    assert got[A] == ["A1", "A2", "A3"] and got[B] == ["B1", "B2", "B3"]
    assert events[B].sets == 3 and events[B].wakeups == 2  # B1+B2 shared one wake-up
    assert fm.ack_map == {} and fm.event_map == {}
    await fm._recv.q.put(R(A, "late token"))  # e.g. a token already in flight
    await settle()
    assert fm.ack_map == {} and not fm._listener.done()
    print("-> B1 and B2 were set() twice but woke B once: the Event is a flag, the list holds the data.")
    print("-> after finish: ack_map == {} and event_map == {}; a late reply for a finished uid is ignored.")
    fm._listener.cancel()


async def part2_sse():
    from minisgl.message import UserReply
    from minisgl.server.api_server import _stream

    section("2. SSE chunks produced by _stream (chat)")
    fm = make_frontend()
    uid = fm.new_user()
    agen = _stream(uid, chat=True)
    chunks = []

    async def drain():
        async with aclosing(agen) as it:
            async for c in it:
                chunks.append(c.decode())

    t = asyncio.create_task(drain())
    for text, fin in (("KV", False), (" Cache", False), ("", True)):
        await fm._recv.q.put(UserReply(uid=uid, incremental_output=text, finished=fin,
                                       finish_reason="length" if fin else None, num_prompt_tokens=7,
                                       num_output_tokens=3 if fin else 0))
        await settle()
    await t
    for c in chunks:  # print without id/object/created/model to keep the output short and stable
        if c.startswith("data: {"):
            obj = json.loads(c[6:])
            c = "data: " + json.dumps({k: v for k, v in obj.items() if k in ("choices", "usage")})
        print("  " + c.strip())
    events = [json.loads(c[6:]) for c in chunks if c.startswith("data: {")]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": "KV"}
    assert events[-1]["choices"][0]["finish_reason"] == "length" and events[-1]["usage"]["completion_tokens"] == 3
    assert chunks[-1] == "data: [DONE]\n\n" and fm.ack_map == {}
    print("-> the first chunk carries role and text together; the end chunk carries the real finish_reason.")
    fm._listener.cancel()


async def part3_early_exit():
    from minisgl.message import UserReply

    section("3. early exit: finally + aclosing vs. the reference's trailing del")
    fm = make_frontend()
    rows = []

    async def run(name, make_gen, close):
        uid = fm.new_user()
        gen = make_gen(uid)
        await fm._recv.q.put(UserReply(uid=uid, incremental_output="x", finished=True, finish_reason="stop"))
        if close:
            async with aclosing(gen) as it:
                async for r in it:
                    if r.finished:
                        break  # what every caller does
        else:
            async for r in gen:
                if r.finished:
                    break
        leaked_while_alive = uid in fm.ack_map
        await gen.aclose()  # what happens eventually (aclosing, or GC finaliser)
        rows.append([name, "yes" if close else "no", leaked_while_alive, uid in fm.ack_map])
        fm.ack_map.pop(uid, None)
        fm.event_map.pop(uid, None)

    await run("this repo: try/finally", fm.wait_for_ack, close=True)
    await run("this repo: try/finally", fm.wait_for_ack, close=False)
    await run("reference: trailing del", lambda u: reference_wait_for_ack(fm, u), close=True)
    print(table(rows, ["wait_for_ack", "caller uses aclosing", "state left after break", "state left after aclose()"]))
    assert rows[0][2:] == [False, False]
    assert rows[1][2:] == [True, False]  # suspended generator still holds the uid until closed
    assert rows[2][2:] == [True, True]  # GeneratorExit skips the trailing del: permanent leak
    print("-> breaking out leaves the generator suspended at `yield`; only a finally block runs on aclose(),")
    print("   and aclosing makes that close happen immediately instead of whenever the generator is GC'd.")
    fm._listener.cancel()


async def part4_cancel():
    from minisgl.message import AbortMsg, UserReply
    from minisgl.server.api_server import _collect_or_abort, _stream, _stream_with_cancellation

    section("4. cancellation and client disconnects")
    fm = make_frontend()
    rows = []

    # (a) the handler task is cancelled while waiting (e.g. server shutdown / uvicorn cancels it)
    uid = fm.new_user()
    task = asyncio.create_task(fm.collect(uid))
    await settle()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    rows.append(["collect() task cancelled", uid in fm.ack_map, "-", "finally in wait_for_ack"])

    # (b) non-streaming request, client goes away (polled every 0.5 s)
    uid = fm.new_user()
    res = await _collect_or_abort(uid, FakeRequest(after=0))
    sent = [m.uid for m in fm._send.sent if isinstance(m, AbortMsg)]
    rows.append(["non-streaming disconnect", uid in fm.ack_map, uid in sent, f"returns {res}"])
    assert res is None and uid in sent

    # (c) streaming request, client goes away after the first chunk
    uid = fm.new_user()
    stream = _stream_with_cancellation(_stream(uid, chat=False), FakeRequest(after=1), uid)
    got = []

    async def consume():
        async for c in stream:
            got.append(c)

    t = asyncio.create_task(consume())
    for i in range(3):
        await fm._recv.q.put(UserReply(uid=uid, incremental_output=f"t{i}", finished=False))
        await settle()
    try:
        await t
        cancelled = False
    except asyncio.CancelledError:
        cancelled = True
    await settle()  # let the abort_user task run
    sent = [m.uid for m in fm._send.sent if isinstance(m, AbortMsg)]
    rows.append(["streaming disconnect", uid in fm.ack_map, uid in sent, f"{len(got)} chunk sent, CancelledError={cancelled}"])
    assert cancelled and uid in sent and len(got) == 1
    print(table(rows, ["scenario", "state left in ack_map", "AbortMsg sent", "note"]))
    assert not any(r[1] for r in rows) and fm.ack_map == {} and fm.event_map == {}
    print("-> AbortMsg then travels: tokenizer (AbortBackendMsg) -> Scheduler._process_one_msg -> abort_req +")
    print("   _free_req_resources -> DetokenizeMsg(finish_reason='abort') -> detokenizer drops the uid.")
    fm._listener.cancel()


async def part5_blocking():
    section("5. a blocking call inside `async def` stalls the event loop")
    rows = {}

    async def measure(name, work):
        ticks = []

        async def ticker():
            for _ in range(8):
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.05)

        tk = asyncio.create_task(ticker())
        await asyncio.sleep(0.06)
        t0 = time.perf_counter()
        await work()
        took = time.perf_counter() - t0
        await tk
        rows[name] = (took, max(b - a for a, b in zip(ticks, ticks[1:])))

    async def blocking():
        time.sleep(0.3)  # no await: nothing else can run

    async def cooperative():
        await asyncio.sleep(0.3)

    async def threaded():
        await asyncio.to_thread(time.sleep, 0.3)

    for name, fn in (("time.sleep(0.3) in async def", blocking), ("await asyncio.sleep(0.3)", cooperative),
                     ("await asyncio.to_thread(time.sleep, 0.3)", threaded)):
        await measure(name, fn)
    print(table([[n, f"{a * 1000:.0f}", f"{g * 1000:.0f}"] for n, (a, g) in rows.items()],
                ["handler body", "handler ms", "max gap between 50 ms ticks of another coroutine (ms)"]))
    gaps = [g for _, g in rows.values()]
    assert gaps[0] > 0.25 and gaps[1] < 0.2 and gaps[2] < 0.2
    print("-> mini-sglang's HTTP coroutine only awaits ZMQ / Events; the model runs in the scheduler process.")


async def amain() -> None:
    await part1_routing()
    await part2_sse()
    await part3_early_exit()
    await part4_cancel()
    await part5_blocking()


def main(argv=None) -> None:
    p = lab_parser(__doc__.splitlines()[0])
    p.parse_args(argv)  # --model is accepted for uniformity but unused (no model needed)
    quiet_logs()
    import logging

    logging.getLogger("minisgl.server.api_server").setLevel(logging.ERROR)
    asyncio.run(amain())


if __name__ == "__main__":
    main()
