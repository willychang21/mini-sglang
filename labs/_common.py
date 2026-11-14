"""Shared helpers for the study-guide labs.

Every lab is a standalone script with a ``main(argv=None) -> None`` entry point:

    python labs/part2_ch04_kv_cache.py                # tiny random Qwen3 (fast, default)
    python labs/part2_ch04_kv_cache.py --model real   # Qwen/Qwen3-0.6B (if downloaded)

Labs *check* what they print with ``assert`` so that ``tests/test_labs.py`` can run
them as smoke tests: a lab that runs to the end has verified its own claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

REAL_MODEL = os.environ.get("MINISGL_TEST_MODEL", "/tmp/models/Qwen3-0.6B")
CACHE_DIR = os.environ.get("MINISGL_LABS_CACHE", os.path.join(tempfile.gettempdir(), "minisgl-labs"))

# Same tiny architecture as tests/conftest.py: 3 layers, hidden 64, 4 Q heads,
# 2 KV heads (GQA group of 2), head_dim 16, vocab 512, tied embeddings.
TINY_CONFIG = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def make_tiny_model(with_tokenizer: bool = False, seed: int = 0, **overrides: Any) -> str:
    """Create (once, cached on disk) a tiny random Qwen3 checkpoint and return its path.

    ``with_tokenizer=True`` uses the real Qwen3 vocabulary size and copies the real
    tokenizer next to the tiny weights, so text in / text out works (the text is
    gibberish: the weights are random).
    """
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = dict(TINY_CONFIG)
    if with_tokenizer:
        if not os.path.exists(os.path.join(REAL_MODEL, "tokenizer.json")):
            raise SystemExit(f"--with-tokenizer needs the real Qwen3 tokenizer in {REAL_MODEL}")
        cfg.update(vocab_size=151936, eos_token_id=151645, bos_token_id=151643)
    cfg.setdefault("eos_token_id", cfg["vocab_size"] - 1)
    cfg.setdefault("bos_token_id", cfg["vocab_size"] - 2)
    cfg.update(overrides)
    key = json.dumps({"cfg": cfg, "seed": seed, "tok": with_tokenizer}, sort_keys=True)
    path = os.path.join(CACHE_DIR, "tiny-" + hashlib.sha1(key.encode()).hexdigest()[:10])
    if os.path.exists(os.path.join(path, "model.safetensors")):
        return path

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=CACHE_DIR)
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
    with torch.no_grad():  # random init is too tame; make logits peaky and diverse
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.2)
    model.save_pretrained(tmp, safe_serialization=True)
    gen = os.path.join(tmp, "generation_config.json")
    if os.path.exists(gen):
        os.remove(gen)
    if with_tokenizer:
        for f in _TOKENIZER_FILES:
            shutil.copy(os.path.join(REAL_MODEL, f), tmp)
    try:
        os.rename(tmp, path)  # atomic: concurrent labs never see a half-written model
    except OSError:  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def resolve_model(name: str) -> str:
    if name == "tiny":
        return make_tiny_model()
    if name == "real":
        if not os.path.exists(os.path.join(REAL_MODEL, "config.json")):
            raise SystemExit(f"real model not found at {REAL_MODEL} (set MINISGL_TEST_MODEL)")
        return REAL_MODEL
    return name


def lab_parser(description: str, default_model: str = "tiny") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--model",
        default=default_model,
        help="'tiny' (random 3-layer Qwen3, default), 'real' (Qwen3-0.6B) or a checkpoint path",
    )
    return p


def make_llm(model_path: str, **engine_kwargs: Any):
    """An offline ``LLM`` on CPU. Without a tokenizer, prompts must be token-id lists."""
    from minisgl.llm import LLM

    engine_kwargs.setdefault("device", "cpu")
    engine_kwargs.setdefault("max_total_tokens", 2048)
    engine_kwargs.setdefault("max_seq_len", 512)
    load_tok = engine_kwargs.pop(
        "load_tokenizer_", os.path.exists(os.path.join(model_path, "tokenizer.json"))
    )
    return LLM(model_path, load_tokenizer_=load_tok, **engine_kwargs)


def make_engine(model_path: str, **engine_kwargs: Any):
    from minisgl.config import EngineConfig
    from minisgl.engine import Engine

    engine_kwargs.setdefault("device", "cpu")
    engine_kwargs.setdefault("max_total_tokens", 2048)
    engine_kwargs.setdefault("max_seq_len", 512)
    return Engine(EngineConfig(model_path=model_path, **engine_kwargs))


def quiet_logs() -> None:
    """Silence the engine's INFO logs so lab output stays readable."""
    import logging

    for name in list(logging.root.manager.loggerDict):
        if name.startswith("minisgl"):
            logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("MINISGL_LOG_LEVEL", "WARNING")


def section(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 74 - len(title)))


def table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    """Render a plain-text table (also valid Markdown)."""
    rows = [[_fmt(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    line = lambda cells: "| " + " | ".join(c.rjust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    out = [line(list(headers)), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TiB"


# Make `python labs/xxx.py` work from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MINISGL_LOG_LEVEL", "WARNING")
