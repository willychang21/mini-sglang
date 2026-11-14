"""Chapter 9: tensor parallelism over Gloo on CPU must reproduce TP=1."""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(model: str, size: int, out: str, prompt, max_tokens: int = 10) -> dict:
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.join(HERE, "tp_worker.py"), str(r), str(size), model, str(port),
             out, str(max_tokens), ",".join(map(str, prompt))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for r in range(size)
    ]
    for p in procs:
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err.decode()[-2000:]
    return torch.load(out)


@pytest.mark.parametrize("fixture", ["tiny_model", "tiny_model_untied"])
def test_tp2_matches_tp1(request, fixture, tmp_path):
    # tiny_model_untied has a single KV head -> replicated across the 2 ranks
    model = request.getfixturevalue(fixture)
    prompt = [3, 17, 99, 4, 250, 8, 77, 5]
    a = _run(model, 1, str(tmp_path / "tp1.pt"), prompt)
    b = _run(model, 2, str(tmp_path / "tp2.pt"), prompt)
    torch.testing.assert_close(a["logits"], b["logits"], atol=1e-4, rtol=1e-4)
    assert a["tokens"] == b["tokens"] and len(a["tokens"]) == 10
