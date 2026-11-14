"""Chapter 5: the HTTP server, multi-process pipeline, streaming and cancellation."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from minisgl import SamplingParams
from minisgl.llm import LLM


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(model: str, *extra: str):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "minisgl", "--model", model, "--device", "cpu", "--port", str(port),
         "--max-total-tokens", "4096", "--max-seq-len", "1024", "--dist-port", str(_free_port()), *extra],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read().decode()[-3000:])
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return proc, url
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise TimeoutError("server did not start")


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server(tiny_model_with_tokenizer):
    proc, url = _start(tiny_model_with_tokenizer, "--num-tokenizers", "2")
    yield url
    _stop(proc)


@pytest.fixture(scope="module")
def reference(tiny_model_with_tokenizer):
    llm = LLM(tiny_model_with_tokenizer, device="cpu", max_total_tokens=4096, max_seq_len=1024)

    def gen(prompt, n):
        return llm.generate([prompt], SamplingParams(max_tokens=n, ignore_eos=True))[0].text

    return gen


def _chat(text):
    return [{"role": "user", "content": text}]


def test_non_streaming_chat_matches_offline(server, reference):
    body = {"messages": _chat("hello there"), "max_tokens": 12, "temperature": 0, "ignore_eos": True}
    r = httpx.post(server + "/v1/chat/completions", json=body, timeout=60).json()
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"]["content"] == reference(_chat("hello there"), 12)
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 12


def test_streaming_matches_non_streaming(server):
    body = {"prompt": "Once upon a time", "max_tokens": 16, "temperature": 0, "ignore_eos": True}
    full = httpx.post(server + "/v1/completions", json=body, timeout=60).json()["choices"][0]["text"]
    chunks, done = [], False
    with httpx.stream("POST", server + "/v1/completions", json={**body, "stream": True}, timeout=60) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)["choices"][0]
            chunks.append(chunk["text"])
    assert done and "".join(chunks) == full


def test_chat_stream_has_role_and_finish_reason(server):
    body = {"messages": _chat("hi"), "max_tokens": 5, "temperature": 0, "stream": True, "ignore_eos": True}
    events = []
    with httpx.stream("POST", server + "/v1/chat/completions", json=body, timeout=60) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] == "length"


def test_concurrent_requests_are_isolated(server, reference):
    prompts = [f"Request number {i} says" for i in range(8)]

    async def run():
        async with httpx.AsyncClient(timeout=120) as client:
            tasks = [
                client.post(
                    server + "/v1/completions",
                    json={"prompt": p, "max_tokens": 10 + i, "temperature": 0, "ignore_eos": True},
                )
                for i, p in enumerate(prompts)
            ]
            return [r.json()["choices"][0]["text"] for r in await asyncio.gather(*tasks)]

    texts = asyncio.run(run())
    for i, (p, t) in enumerate(zip(prompts, texts)):
        assert t == reference(p, 10 + i)


def test_client_disconnect_aborts_and_cleans_up(server):
    body = {"prompt": "count forever", "max_tokens": 900, "temperature": 0, "stream": True, "ignore_eos": True}
    with httpx.stream("POST", server + "/v1/completions", json=body, timeout=60) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 4:
                break  # hang up mid-generation
    deadline = time.time() + 30
    while time.time() < deadline:
        state = httpx.get(server + "/debug/state").json()
        if state == {"ack_map": 0, "event_map": 0}:
            break
        time.sleep(0.2)
    assert state == {"ack_map": 0, "event_map": 0}
    # the server keeps working afterwards
    ok = httpx.post(server + "/v1/completions", json={"prompt": "x", "max_tokens": 2}, timeout=60)
    assert ok.status_code == 200


def test_errors_are_reported(server):
    too_long = {"prompt": "word " * 2000, "max_tokens": 4}
    r = httpx.post(server + "/v1/completions", json=too_long, timeout=60)
    assert r.status_code == 400 and "limit" in r.json()["error"]["message"]
    r = httpx.post(server + "/v1/chat/completions", json={"max_tokens": 4}, timeout=60)
    assert r.status_code == 400
    assert httpx.get(server + "/debug/state").json() == {"ack_map": 0, "event_map": 0}


def test_tp2_server_matches(tiny_model_with_tokenizer, server):
    body = {"prompt": "Tensor parallel test", "max_tokens": 12, "temperature": 0, "ignore_eos": True}
    expected = httpx.post(server + "/v1/completions", json=body, timeout=60).json()["choices"][0]["text"]
    proc, url = _start(tiny_model_with_tokenizer, "--tp-size", "2")
    try:
        got = httpx.post(url + "/v1/completions", json=body, timeout=120).json()["choices"][0]["text"]
    finally:
        _stop(proc)
    assert got == expected


def test_children_exit_with_server(tiny_model_with_tokenizer):
    proc, _ = _start(tiny_model_with_tokenizer)
    out = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
    children = [int(x) for x in out]
    assert children
    proc.kill()  # no chance to clean up
    proc.wait()
    deadline = time.time() + 15
    alive = children
    while alive and time.time() < deadline:
        alive = [c for c in alive if os.path.exists(f"/proc/{c}") and "Z" not in open(f"/proc/{c}/stat").read().split()[2]]
        time.sleep(0.5)
    assert not alive


def test_backpressure_flags_reach_frontend(tiny_model_with_tokenizer):
    """A 1-byte buffer cap cuts off every request that produces text (the
    policy itself is unit-tested in test_backpressure.py)."""
    proc, url = _start(tiny_model_with_tokenizer, "--max-buffered-bytes", "1", "--client-stall-timeout", "5")
    try:
        body = {"prompt": "hello hello hello", "max_tokens": 32, "temperature": 0, "ignore_eos": True}
        r = httpx.post(url + "/v1/completions", json=body, timeout=60)
        assert r.status_code == 400 and "client too slow" in r.json()["error"]["message"]
        deadline = time.time() + 10
        while httpx.get(url + "/debug/state").json() != {"ack_map": 0, "event_map": 0}:
            assert time.time() < deadline
            time.sleep(0.2)
    finally:
        _stop(proc)


def test_overlap_optimistic_server_matches(tiny_model_with_tokenizer, reference):
    """The served pipeline with overlap scheduling + optimistic admission."""
    proc, url = _start(tiny_model_with_tokenizer, "--enable-overlap", "--admission", "optimistic",
                       "--num-tokenizers", "3")
    prompts = [f"Overlap request {i} says" for i in range(6)]

    async def run():
        async with httpx.AsyncClient(timeout=120) as client:
            tasks = [
                client.post(url + "/v1/completions",
                            json={"prompt": p, "max_tokens": 8 + i, "temperature": 0, "ignore_eos": True})
                for i, p in enumerate(prompts)
            ]
            return [r.json()["choices"][0]["text"] for r in await asyncio.gather(*tasks)]

    try:
        texts = asyncio.run(run())
    finally:
        _stop(proc)
    for i, (p, t) in enumerate(zip(prompts, texts)):
        assert t == reference(p, 8 + i)


@pytest.mark.parametrize("algo", [["ngram"], ["draft_model", "--speculative-draft-num-layers", "1"]])
def test_speculative_server_matches(tiny_model_with_tokenizer, reference, algo):
    """Speculative decoding through the served pipeline: several tokens per
    request per step must stream out in order and match target-only output."""
    proc, url = _start(tiny_model_with_tokenizer, "--speculative-algorithm", *algo,
                       "--speculative-num-draft-tokens", "3", "--page-size", "4")
    prompts = [f"Speculative request {i} repeats repeats repeats" for i in range(4)]

    async def run():
        async with httpx.AsyncClient(timeout=120) as client:
            tasks = [
                client.post(url + "/v1/completions",
                            json={"prompt": p, "max_tokens": 10 + 3 * i, "temperature": 0, "ignore_eos": True})
                for i, p in enumerate(prompts)
            ]
            texts = [r.json()["choices"][0]["text"] for r in await asyncio.gather(*tasks)]
            # streaming: pieces concatenate to the same text
            pieces = []
            async with client.stream("POST", url + "/v1/completions",
                                     json={"prompt": prompts[0], "max_tokens": 10, "temperature": 0,
                                           "ignore_eos": True, "stream": True}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        pieces.append(json.loads(line[6:])["choices"][0]["text"])
            return texts, "".join(pieces)

    try:
        texts, streamed = asyncio.run(run())
    finally:
        _stop(proc)
    for i, (p, t) in enumerate(zip(prompts, texts)):
        assert t == reference(p, 10 + 3 * i)
    assert streamed == texts[0]
