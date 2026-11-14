"""Token ids -> incremental text (Chapter 2, Section 4.6).

A single token may hold only part of a UTF-8 character, so decoding token by
token would emit garbage ("�"). For every request we keep two offsets into
its token list:

* ``surr_offset``: start of a short "surrounding" window used as context
* ``read_offset``: end of the text that has already been emitted

Each step decodes ``ids[surr:read]`` and ``ids[surr:]``; the difference is the
new text. If it ends with an incomplete character we hold it back until a later
token completes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from transformers import PreTrainedTokenizerBase

from ..message import DetokenizeMsg, UserReply

_REPLACEMENT = "�"


@dataclass
class DecodeStatus:
    ids: List[int] = field(default_factory=list)
    surr_offset: int = 0
    read_offset: int = 0
    num_prompt_tokens: int = 0


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.status: Dict[int, DecodeStatus] = {}
        self.eos_ids = set()
        for attr in ("eos_token_id", "pad_token_id"):
            if (tid := getattr(tokenizer, attr, None)) is not None:
                self.eos_ids.add(tid)

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _step(self, s: DecodeStatus, flush: bool) -> str:
        surr_str = self._decode(s.ids[s.surr_offset : s.read_offset])
        read_str = self._decode(s.ids[s.surr_offset :])
        new_text = read_str[len(surr_str) :]
        if new_text and (flush or not new_text.endswith(_REPLACEMENT)):
            s.surr_offset = s.read_offset
            s.read_offset = len(s.ids)
            return new_text
        return ""  # wait for the bytes that complete the character

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[UserReply]:
        replies: List[UserReply] = []
        for msg in msgs:
            s = self.status.setdefault(msg.uid, DecodeStatus())
            if msg.num_prompt_tokens:
                s.num_prompt_tokens = msg.num_prompt_tokens
            text = ""
            if msg.next_token >= 0:
                s.ids.append(msg.next_token)
                text = self._step(s, flush=msg.finished)
            replies.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output=text,
                    finished=msg.finished,
                    finish_reason=msg.finish_reason,
                    num_prompt_tokens=s.num_prompt_tokens,
                    num_output_tokens=len(s.ids),
                    error=msg.error,
                )
            )
            if msg.finished:  # completion, abort and errors all clean up here
                del self.status[msg.uid]
        return replies
