"""How the scheduler receives requests and returns tokens (Chapters 5 & 9).

* ``OfflineIO``: in-process queues, used by the Python ``LLM`` API and tests.
* ``ZmqSchedulerIO``: the serving path. Only rank 0 talks to the tokenizer and
  detokenizer over ZMQ; it broadcasts every received message to the other TP
  ranks over a Gloo CPU group, so every rank sees the same requests in the same
  order and therefore builds identical batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List

import torch.distributed as dist

from ..message import BaseBackendMsg, BatchMsg, DetokenizeMsg, unwrap_msg
from ..utils.zmq_utils import ZmqPullQueue, ZmqPushQueue


class SchedulerIO(ABC):
    @abstractmethod
    def receive_msg(self, blocking: bool) -> List[BaseBackendMsg]: ...

    @abstractmethod
    def send_result(self, replies: List[DetokenizeMsg]) -> None: ...

    def has_pending(self) -> bool:
        return False

    def close(self) -> None:
        pass


class OfflineIO(SchedulerIO):
    def __init__(self) -> None:
        self.inbox: Deque[BaseBackendMsg] = deque()
        self.outbox: List[DetokenizeMsg] = []

    def put(self, msg: BaseBackendMsg) -> None:
        self.inbox.append(msg)

    def has_pending(self) -> bool:
        return len(self.inbox) > 0

    def receive_msg(self, blocking: bool) -> List[BaseBackendMsg]:
        msgs = list(self.inbox)
        self.inbox.clear()
        return msgs

    def send_result(self, replies: List[DetokenizeMsg]) -> None:
        self.outbox.extend(replies)


class ZmqSchedulerIO(SchedulerIO):
    def __init__(self, recv_addr: str, send_addr: str, tp_rank: int, tp_size: int, tp_cpu_group=None):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = tp_cpu_group
        if tp_rank == 0:
            self._recv = ZmqPullQueue(recv_addr, bind=True)
            self._send = ZmqPushQueue(send_addr, bind=False)

    def _recv_local(self, blocking: bool) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        if blocking:
            msgs.extend(unwrap_msg(self._recv.get()))
        while (msg := self._recv.get_nowait()) is not None:
            msgs.extend(unwrap_msg(msg))
        return msgs  # type: ignore[return-value]

    def receive_msg(self, blocking: bool) -> List[BaseBackendMsg]:
        if self.tp_size == 1:
            return self._recv_local(blocking)
        payload = [self._recv_local(blocking) if self.tp_rank == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=self.group)
        return payload[0]  # type: ignore[return-value]

    def send_result(self, replies: List[DetokenizeMsg]) -> None:
        if self.tp_rank == 0 and replies:
            self._send.put(BatchMsg(data=list(replies)))

    def close(self) -> None:
        if self.tp_rank == 0:
            self._recv.close()
            self._send.close()
