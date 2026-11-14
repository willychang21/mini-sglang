"""Thin ZMQ PUSH/PULL wrappers carrying pickled ``BaseMsg`` objects."""

from __future__ import annotations

import zmq
import zmq.asyncio

from ..message import BaseMsg


class ZmqPushQueue:
    def __init__(self, addr: str, bind: bool) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(addr)

    def put(self, msg: BaseMsg) -> None:
        self.socket.send(msg.encode())

    def close(self) -> None:
        self.socket.close()


class ZmqPullQueue:
    def __init__(self, addr: str, bind: bool) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(addr)

    def get(self, timeout_ms: int | None = None) -> BaseMsg | None:
        if timeout_ms is not None and not self.socket.poll(timeout_ms):
            return None
        return BaseMsg.decode(self.socket.recv())

    def get_nowait(self) -> BaseMsg | None:
        try:
            return BaseMsg.decode(self.socket.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            return None

    def close(self) -> None:
        self.socket.close()


class ZmqAsyncPushQueue:
    def __init__(self, addr: str, bind: bool) -> None:
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(addr)

    async def put(self, msg: BaseMsg) -> None:
        await self.socket.send(msg.encode())


class ZmqAsyncPullQueue:
    def __init__(self, addr: str, bind: bool) -> None:
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(addr)

    async def get(self) -> BaseMsg:
        return BaseMsg.decode(await self.socket.recv())
