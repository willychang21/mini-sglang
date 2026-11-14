"""Incremental detokenization must never emit broken UTF-8 and must reproduce
the full decode exactly."""

from __future__ import annotations

import os

import pytest

from conftest import REAL_MODEL
from minisgl.message import DetokenizeMsg
from minisgl.tokenizer import DetokenizeManager, load_tokenizer

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(REAL_MODEL, "tokenizer.json")), reason="no tokenizer"
)


@pytest.mark.parametrize(
    "text",
    ["Hello, world! KV cache 🚀 is great.", "用一句话介绍北京。北京是中国的首都🏯", "émoji 👨‍👩‍👧 mixed 中文 text"],
)
def test_incremental_equals_full_decode(text):
    tok = load_tokenizer(REAL_MODEL)
    ids = tok.encode(text, add_special_tokens=False)
    manager = DetokenizeManager(tok)
    pieces = []
    for i, t in enumerate(ids):
        (reply,) = manager.detokenize([DetokenizeMsg(uid=7, next_token=t, finished=i == len(ids) - 1)])
        assert "�" not in reply.incremental_output
        pieces.append(reply.incremental_output)
    assert "".join(pieces) == tok.decode(ids)
    assert manager.status == {}  # state is dropped when the request finishes


def test_abort_message_cleans_state():
    tok = load_tokenizer(REAL_MODEL)
    manager = DetokenizeManager(tok)
    manager.detokenize([DetokenizeMsg(uid=1, next_token=tok.encode("hi")[0], finished=False)])
    assert 1 in manager.status
    (reply,) = manager.detokenize([DetokenizeMsg(uid=1, next_token=-1, finished=True, finish_reason="abort")])
    assert reply.finished and reply.finish_reason == "abort" and manager.status == {}
