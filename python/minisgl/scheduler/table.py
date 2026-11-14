"""Request slots (Chapter 6).

Each running request owns one row of ``token_pool`` (its token ids) and the
matching row of ``page_table`` (the KV slot of each of those tokens).
"""

from __future__ import annotations

from typing import List

import torch


class TableManager:
    def __init__(self, max_running_req: int, page_table: torch.Tensor) -> None:
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table)  # token ids, same shape
        self._free_rows: List[int] = list(range(max_running_req))

    @property
    def available_size(self) -> int:
        return len(self._free_rows)

    def allocate(self) -> int:
        return self._free_rows.pop(0)

    def free(self, table_idx: int) -> None:
        assert table_idx not in self._free_rows, f"row {table_idx} freed twice"
        self._free_rows.append(table_idx)
