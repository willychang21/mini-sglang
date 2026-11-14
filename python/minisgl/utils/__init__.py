"""Small shared helpers (arithmetic on lengths/pages, logging, a tiny registry)."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Generic, TypeVar


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_down(a: int, b: int) -> int:
    return (a // b) * b


def align_up(a: int, b: int) -> int:
    return div_ceil(a, b) * b


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Split ``a`` evenly into ``b`` parts.

    With ``allow_replicate`` a value smaller than ``b`` (e.g. 2 KV heads on 4 TP
    ranks) is replicated: every rank keeps exactly one item.
    """
    if allow_replicate and a < b:
        assert b % a == 0, f"cannot replicate {a} items over {b} ranks"
        return 1
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(processName)s %(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.environ.get("MINISGL_LOG_LEVEL", "INFO"))
    return logger


T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string name to a factory, e.g. ``"radix" -> RadixPrefixCache``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            self._items[name] = fn
            return fn

        return decorator

    def __getitem__(self, name: str) -> Callable[..., T]:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}', choose from {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)
