"""RadixAttention's prefix cache: a path-compressed radix tree (Chapter 8).

"The Radix Tree records mappings, while the KV Cache Pool stores the data."

Every node stores a contiguous run of token ids (``key``) and the physical KV
slots holding their K/V (``value``). Node lengths are always multiples of
``page_size`` so that only whole pages are ever shared.

Lifecycle: match -> lock -> allocate -> insert -> unlock -> evict/free.
``ref_count > 0`` means some running request uses the node, so it must not be
evicted; only unreferenced *leaves* are evicted, oldest timestamp first (LRU).
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Callable, Dict, List, Tuple

import torch

from ..utils import align_down
from .base import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo

KeyFn = Callable[[torch.Tensor], Any]


def _get_key_fn(page_size: int) -> KeyFn:
    # Children are indexed by their first token (page_size == 1) or by their
    # first full page of tokens (page_size > 1).
    if page_size == 1:
        return lambda x: int(x[0].item())
    return lambda x: tuple(x[:page_size].tolist())


def _fast_compare_key(key: torch.Tensor, input_ids: torch.Tensor) -> int:
    """Length of the common prefix of ``key`` and ``input_ids``."""
    n = min(len(key), len(input_ids))
    if n == 0:
        return 0
    mismatch = torch.nonzero(key[:n] != input_ids[:n])
    return n if len(mismatch) == 0 else int(mismatch[0].item())


class RadixTreeNode:
    def __init__(self, key_fn: KeyFn, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count = 0
        self.timestamp = tic or time.monotonic_ns()
        self._key = torch.empty(0, dtype=torch.int64)
        self._value = torch.empty(0, dtype=torch.int64)
        self._length = 0

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: "RadixTreeNode") -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def parent(self) -> "RadixTreeNode":
        assert self._parent is not None
        return self._parent

    @property
    def key(self) -> torch.Tensor:
        return self._key

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def length(self) -> int:
        return self._length

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        return _fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> "RadixTreeNode":
        """Split ``[0, pos)`` into a new parent node; ``self`` keeps the suffix.

        The physical KV does not move, only the index tensors are sliced. The new
        node inherits ``ref_count``: everyone protecting ``self`` protects its
        prefix too.
        """
        assert 0 < pos < self._length
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)  # replaces `self` in parent's children
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: "RadixTreeNode") -> bool:  # for the eviction heap
        return self.timestamp < other.timestamp


class RadixCacheHandle(BaseCacheHandle):
    def __init__(self, cached_len: int, node: RadixTreeNode) -> None:
        self.cached_len = cached_len
        self.node = node

    def get_matched_indices(self) -> torch.Tensor:
        # Each node only stores its own segment: walk to the root, then reverse.
        values: List[torch.Tensor] = []
        node = self.node
        while not node.is_root():
            values.append(node.value)
            node = node.parent
        if not values:
            return torch.empty(0, dtype=torch.int64, device=self.node.value.device)
        values.reverse()
        return torch.cat(values)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, page_size: int = 1) -> None:
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # the root is never evicted
        self.evictable_size = 0
        self.protected_size = 0

    # ------------------------------------------------------------------ match
    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < len(input_ids):
            remaining = input_ids[prefix_len:]
            if len(remaining) < self.page_size:
                break
            child = node.children.get(self.key_fn(remaining))
            if child is None:
                break
            node = child

            match_len = align_down(node.get_match_len(remaining), self.page_size)
            prefix_len += match_len

            if match_len != node.length:  # stopped inside the node -> split it
                node = node.split_at(match_len)
                node.timestamp = tic
                break

            node.timestamp = tic  # full-node hit: refresh the LRU timestamp

        return node, prefix_len

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids.cpu())
        return MatchResult(RadixCacheHandle(prefix_len, node))

    # ------------------------------------------------------------------- lock
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        while not node.is_root():
            if unlock:
                assert node.ref_count > 0, "unlocking a node that is not locked"
                node.ref_count -= 1
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
            else:
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
            node = node.parent

    # ----------------------------------------------------------------- insert
    def insert_prefix(
        self, input_ids: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[int, BaseCacheHandle]:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids = input_ids[:insert_len].cpu()
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len < insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(
                input_ids[prefix_len:].clone(), indices[prefix_len:insert_len].clone()
            )
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return prefix_len, RadixCacheHandle(insert_len, node)

    # ------------------------------------------------------------------ evict
    def _collect_evictable_leaves(self) -> List[RadixTreeNode]:
        leaves: List[RadixTreeNode] = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if not node.is_root() and node.ref_count == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def evict(self, size: int) -> torch.Tensor:
        if size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if size > self.evictable_size:
            raise RuntimeError(f"cannot evict {size} tokens, only {self.evictable_size} evictable")
        heap = self._collect_evictable_leaves()
        heapq.heapify(heap)
        evicted: List[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            assert heap, "eviction ran out of leaves"
            leaf = heapq.heappop(heap)
            evicted.append(leaf.value)
            evicted_size += leaf.length
            self.evictable_size -= leaf.length
            parent = leaf.parent
            del parent.children[self.key_fn(leaf.key)]
            if parent.is_leaf() and not parent.is_root() and parent.ref_count == 0:
                heapq.heappush(heap, parent)
        return torch.cat(evicted)

    # ------------------------------------------------------------------- info
    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        evictable = protected = 0
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            assert node.length > 0 and node.length % self.page_size == 0
            assert node.ref_count >= 0
            for child in node.children.values():
                assert child.ref_count <= node.ref_count, "child locked more than parent"
                assert child.parent is node
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length
            stack.extend(node.children.values())
        assert evictable == self.evictable_size, (evictable, self.evictable_size)
        assert protected == self.protected_size, (protected, self.protected_size)

    def num_nodes(self) -> int:
        count, stack = 0, [self.root_node]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count
