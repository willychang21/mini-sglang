"""Tensor-parallel identity of the current process (Chapter 9).

Layers call :func:`get_tp_info` while they are constructed to decide the *local*
shape of their weights, so :func:`set_tp_info` must run before the model is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    rank: int = 0
    size: int = 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO = DistributedInfo()


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    assert 0 <= rank < size
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    return _TP_INFO
