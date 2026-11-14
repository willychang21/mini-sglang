"""Serializable messages exchanged between workers (Chapters 2 & 5).

Request path::

    API server --TokenizeMsg--> tokenizer --UserMsg--> scheduler
    scheduler --DetokenizeMsg--> detokenizer --UserReply--> API server

Cancellation path::

    API server --AbortMsg--> tokenizer --AbortBackendMsg--> scheduler

The ``uid`` assigned by the API server travels unchanged through every hop; it
is how the API server routes each reply back to the right HTTP connection.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core import SamplingParams


@dataclass
class BaseMsg:
    def encode(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def decode(data: bytes) -> "BaseMsg":
        msg = pickle.loads(data)
        assert isinstance(msg, BaseMsg)
        return msg


# ----------------------------------------------------------- frontend -> tokenizer
@dataclass
class TokenizeMsg(BaseMsg):
    uid: int
    text: str | List[Dict[str, Any]] | List[int]  # prompt, chat messages, or token ids
    sampling_params: SamplingParams
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AbortMsg(BaseMsg):
    uid: int


# ---------------------------------------------------------- tokenizer -> scheduler
@dataclass
class BaseBackendMsg(BaseMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: List[int]
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


# -------------------------------------------------------- scheduler -> detokenizer
@dataclass
class DetokenizeMsg(BaseMsg):
    uid: int
    next_token: int  # -1 when no token is attached (error / abort)
    finished: bool
    finish_reason: str | None = None  # "stop" | "length" | "abort" | "error"
    num_prompt_tokens: int = 0
    error: str | None = None


# --------------------------------------------------------- detokenizer -> frontend
@dataclass
class UserReply(BaseMsg):
    uid: int
    incremental_output: str
    finished: bool
    finish_reason: str | None = None
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    error: str | None = None


@dataclass
class BatchMsg(BaseMsg):
    data: List[BaseMsg]


def unwrap_msg(msg: BaseMsg) -> List[BaseMsg]:
    return list(msg.data) if isinstance(msg, BatchMsg) else [msg]


__all__ = [
    "BaseMsg",
    "TokenizeMsg",
    "AbortMsg",
    "BaseBackendMsg",
    "UserMsg",
    "AbortBackendMsg",
    "ExitMsg",
    "DetokenizeMsg",
    "UserReply",
    "BatchMsg",
    "unwrap_msg",
]
