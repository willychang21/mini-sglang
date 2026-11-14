"""Chapter 5 follow-up: the multi-tokenizer abort race.

A request's ``TokenizeMsg`` and its ``AbortMsg`` must reach the scheduler in
that order. With several tokenizers behind one round-robin PUSH socket the
abort can be handed to an idle tokenizer and overtake the slow tokenization of
its own request; the scheduler then sees an abort for an unknown uid, ignores
it, and runs the request to completion.

These tests run real ``tokenize_worker`` processes between a real
``FrontendManager`` and a fake scheduler socket, abort every request right
after submitting it, and check the order in which the scheduler receives them.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import tempfile
import uuid
from typing import Dict, List

import pytest

from minisgl.core import SamplingParams
from minisgl.message import AbortBackendMsg, AbortMsg, TokenizeMsg, UserMsg, unwrap_msg
from minisgl.server.api_server import FrontendManager
from minisgl.tokenizer.server import tokenize_worker
from minisgl.utils.zmq_utils import ZmqPullQueue

NUM_TOKENIZERS = 4
NUM_REQS = 32


def _addr(run: str, name: str) -> str:
    return f"ipc://{os.path.join(tempfile.gettempdir(), f'minisgl-test-{run}-{name}')}"


def _run(model: str, shared_socket: bool) -> Dict[int, List[str]]:
    """Returns, per uid, the kinds of messages the scheduler saw in arrival order."""
    run = uuid.uuid4().hex[:8]
    sched_addr = _addr(run, "scheduler")
    if shared_socket:  # the old topology: every tokenizer pulls from one socket
        tok_addrs = [_addr(run, "tokenizer")] * NUM_TOKENIZERS
        front_addrs = [tok_addrs[0]]
    else:
        tok_addrs = [_addr(run, f"tokenizer-{i}") for i in range(NUM_TOKENIZERS)]
        front_addrs = tok_addrs

    sched = ZmqPullQueue(sched_addr, bind=True)
    front = FrontendManager("test", front_addrs, _addr(run, "frontend"))
    front.bind()
    ctx = mp.get_context("spawn")
    acks = ctx.Queue()
    procs = [
        ctx.Process(target=tokenize_worker, args=(model, tok_addrs[i], sched_addr, acks, f"tok-{i}"), daemon=True)
        for i in range(NUM_TOKENIZERS)
    ]
    for p in procs:
        p.start()
    try:
        for _ in procs:
            acks.get(timeout=120)

        # A long prompt makes tokenization slow; the abort is trivial to forward.
        long_prompt = "the quick brown fox jumps over the lazy dog " * 400

        async def submit():
            await asyncio.sleep(0.5)  # let every tokenizer finish connecting
            for uid in range(NUM_REQS):
                await front.send_one(TokenizeMsg(uid=uid, text=long_prompt, sampling_params=SamplingParams()))
                await front.send_one(AbortMsg(uid=uid))
            front._listener.cancel()

        asyncio.run(submit())

        seen: Dict[int, List[str]] = {uid: [] for uid in range(NUM_REQS)}
        remaining = 2 * NUM_REQS
        while remaining > 0:
            msg = sched.get(timeout_ms=60_000)
            assert msg is not None, "tokenizers stopped forwarding"
            for m in unwrap_msg(msg):
                kind = "user" if isinstance(m, UserMsg) else "abort"
                assert isinstance(m, (UserMsg, AbortBackendMsg))
                seen[m.uid].append(kind)
                remaining -= 1
        return seen
    finally:
        for p in procs:
            p.terminate()
        sched.close()


def test_shared_socket_reproduces_abort_race(tiny_model_with_tokenizer):
    """The pre-fix topology: aborts overtake their own requests."""
    seen = _run(tiny_model_with_tokenizer, shared_socket=True)
    overtaken = [uid for uid, kinds in seen.items() if kinds == ["abort", "user"]]
    assert overtaken, "expected at least one abort to overtake its request"


def test_uid_routing_preserves_per_uid_order(tiny_model_with_tokenizer):
    seen = _run(tiny_model_with_tokenizer, shared_socket=False)
    assert all(kinds == ["user", "abort"] for kinds in seen.values()), seen


def test_route_is_stable_per_uid():
    front = FrontendManager("m", ["a", "b", "c"], "r")
    assert [front.route(uid) for uid in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert FrontendManager("m", "only", "r").route(12345) == 0
