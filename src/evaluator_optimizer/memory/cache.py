"""
L1 in-process memory cache.

MemoryCache provides a bounded LRU cache using OrderedDict.

The cache intentionally remains small because L1 is intended for
frequently reused memories during the current application process.
"""

from __future__ import annotations

from collections import OrderedDict

from evaluator_optimizer.config import DEFAULT_MEMORY_CACHE_SIZE
from evaluator_optimizer.logging_config import get_logger
from evaluator_optimizer.memory.models import MemoryItem


logger = get_logger("memory.cache")


class MemoryCache:
    """
    Bounded LRU cache for MemoryItem objects.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MEMORY_CACHE_SIZE,
    ) -> None:
        """Initialize the bounded cache."""
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self._max_size = max_size
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()

    @property
    def max_size(self) -> int:
        """Return maximum cache capacity."""
        return self._max_size

    @property
    def size(self) -> int:
        """Return current number of cached memories."""
        return len(self._items)

    def get(self, memory_id: str) -> MemoryItem | None:
        """
        Retrieve a memory and promote it to the most recently used
        position.
        """
        memory = self._items.get(memory_id)

        if memory is None:
            return None

        self._items.move_to_end(memory_id)
        memory.mark_accessed()

        return memory

    def put(self, memory: MemoryItem) -> None:
        """Insert or replace a memory."""
        self._items[memory.memory_id] = memory
        self._items.move_to_end(memory.memory_id)

        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

    def remove(self, memory_id: str) -> None:
        """Remove a memory if it exists."""
        self._items.pop(memory_id, None)

    def clear(self) -> None:
        """Remove all cached memories."""
        self._items.clear()

    def values(self) -> list[MemoryItem]:
        """Return cached memories."""
        return list(self._items.values())