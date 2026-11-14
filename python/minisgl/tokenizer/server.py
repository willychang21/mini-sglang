"""Tokenizer and detokenizer worker processes (Chapters 2 & 5)."""

from __future__ import annotations

from typing import List

import torch

from ..message import (
    AbortBackendMsg,
    AbortMsg,
    BaseMsg,
    BatchMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    unwrap_msg,
)
from ..utils import exit_when_parent_dies, init_logger
from ..utils.zmq_utils import ZmqPullQueue, ZmqPushQueue
from .detokenize import DetokenizeManager
from .tokenize import TokenizeManager, load_tokenizer

logger = init_logger(__name__)


def _drain(queue: ZmqPullQueue) -> List[BaseMsg]:
    msgs = unwrap_msg(queue.get())  # block for the first message
    while (more := queue.get_nowait()) is not None:
        msgs.extend(unwrap_msg(more))
    return msgs


def tokenize_worker(model_path: str, recv_addr: str, send_addr: str, ack_queue, name: str) -> None:
    exit_when_parent_dies()
    torch.set_num_threads(1)
    manager = TokenizeManager(load_tokenizer(model_path))
    recv = ZmqPullQueue(recv_addr, bind=False)  # from the API server
    send = ZmqPushQueue(send_addr, bind=False)  # to scheduler rank 0
    ack_queue.put(f"{name} ready")
    while True:
        out: List[BaseMsg] = []
        for msg in _drain(recv):
            if isinstance(msg, TokenizeMsg):
                try:
                    ids = manager.tokenize_one(msg.text, msg.chat_template_kwargs)
                    out.append(UserMsg(uid=msg.uid, input_ids=ids, sampling_params=msg.sampling_params))
                except Exception as e:  # report instead of leaving the client hanging
                    logger.exception("tokenization failed for uid %d", msg.uid)
                    out.append(UserMsg(msg.uid, [], msg.sampling_params, error=f"tokenization failed: {e}"))
            elif isinstance(msg, AbortMsg):
                out.append(AbortBackendMsg(uid=msg.uid))
            else:
                logger.warning("tokenizer ignoring %r", msg)
        if out:
            send.put(BatchMsg(data=out))


def detokenize_worker(model_path: str, recv_addr: str, send_addr: str, ack_queue, name: str) -> None:
    exit_when_parent_dies()
    torch.set_num_threads(1)
    manager = DetokenizeManager(load_tokenizer(model_path))
    recv = ZmqPullQueue(recv_addr, bind=True)  # from scheduler rank 0
    send = ZmqPushQueue(send_addr, bind=False)  # to the API server
    ack_queue.put(f"{name} ready")
    while True:
        msgs = [m for m in _drain(recv) if isinstance(m, DetokenizeMsg)]
        replies = manager.detokenize(msgs)
        if replies:
            send.put(BatchMsg(data=list(replies)))
