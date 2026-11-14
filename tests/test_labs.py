"""Smoke tests for the study-guide labs (labs/*.py).

Each lab asserts its own claims, so "the lab ran to the end" means its printed
observations were checked. Labs run in a subprocess: they set process-global
state (TP info, global context) and must not leak it into other tests.

Fast labs (< ~20 s on the tiny model) always run. Slower ones run only with
``MINISGL_RUN_SLOW_LABS=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABS = os.path.join(ROOT, "labs")
REAL_MODEL = os.environ.get("MINISGL_TEST_MODEL", "/tmp/models/Qwen3-0.6B")
HAS_TOKENIZER = os.path.exists(os.path.join(REAL_MODEL, "tokenizer.json"))

# (script, extra args, needs the real tokenizer)
FAST_LABS = [
    ("part1_ch02_roofline.py", [], False),
    ("part1_ch04_kv_memory.py", [], False),
    ("part1_ch05_metrics.py", [], False),
    ("part2_ch02_request_path.py", [], True),
    ("part2_ch03_forward_shapes.py", [], False),
    ("part2_ch04_kv_cache.py", [], False),
    ("part2_ch05_frontend.py", [], False),
    ("part2_ch06_batching_timeline.py", [], False),
    ("part2_ch07_paged_memory.py", [], False),
    ("part2_ch08_radix_tree.py", [], False),
    ("part2_ch09_tp_shapes.py", [], False),
    ("part2_ch10_speculative.py", [], False),
]
SLOW_LABS: list = []


def _run(script: str, args: list, timeout: int) -> None:
    env = dict(os.environ, PYTHONUNBUFFERED="1", MINISGL_LOG_LEVEL="WARNING")
    proc = subprocess.run(
        [sys.executable, os.path.join(LABS, script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"


@pytest.mark.parametrize("script,args,needs_tok", FAST_LABS, ids=[s for s, _, _ in FAST_LABS])
def test_fast_lab(script, args, needs_tok):
    if needs_tok and not HAS_TOKENIZER:
        pytest.skip("real Qwen3 tokenizer not available")
    _run(script, args, timeout=180)


@pytest.mark.skipif(not os.environ.get("MINISGL_RUN_SLOW_LABS"), reason="set MINISGL_RUN_SLOW_LABS=1")
@pytest.mark.parametrize("script,args,needs_tok", SLOW_LABS, ids=[s for s, _, _ in SLOW_LABS])
def test_slow_lab(script, args, needs_tok):
    if needs_tok and not HAS_TOKENIZER:
        pytest.skip("real Qwen3 tokenizer not available")
    _run(script, args, timeout=600)


def test_every_lab_is_listed():
    listed = {s for s, _, _ in FAST_LABS + SLOW_LABS}
    on_disk = {f for f in os.listdir(LABS) if f.endswith(".py") and not f.startswith("_")}
    assert on_disk == listed, f"unlisted: {on_disk - listed}, missing: {listed - on_disk}"
