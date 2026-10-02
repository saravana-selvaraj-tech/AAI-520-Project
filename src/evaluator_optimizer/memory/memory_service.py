"""
Memory service.

Coordinates L1 in-process caching and L2 persistent memory.

Retrieval order:

    L1 cache
       ↓
    L2 repository
       ↓
    no memory

A persistent hit is promoted into L1.
"""

from __future__ import annotations

from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.models import MemoryItem
from evaluator_optimizer.logging_config import get_logger
from evaluator_optimizer.memory.repository import (
    MemoryQuery,
    MemoryRepository,
)


logger = get_logger("memory.service")


class MemoryService:
    """
    Application service for memory storage and retrieval.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        cache: MemoryCache,
    ) -> None:
        self._repository = repository
        self._cache = cache

    def save(self, memory: MemoryItem) -> None:
        """
        Save memory to both L1 and L2.
        """
        logger.info("Memory persistence started | memory_id=%s", memory.memory_id)
        self._cache.put(memory)
        self._repository.save(memory)
        logger.info("Memory persistence completed | memory_id=%s", memory.memory_id)

    def get(
        self,
        memory_id: str,
    ) -> MemoryItem | None:
        """
        Retrieve memory from L1 first, then L2.
        """
        memory = self._cache.get(memory_id)

        if memory is not None:
            return memory

        memory = self._repository.get(memory_id)

        if memory is not None:
            memory.mark_accessed()
            self._cache.put(memory)

        return memory

    def search(
        self,
        query: MemoryQuery,
    ) -> list[MemoryItem]:
        """
        Search L1 first and then L2.

        Persistent results not already present in L1 are promoted
        into the cache.
        """
        logger.info("Memory retrieval started | max_results=%d", query.max_results)
        cache_matches = [
            memory
            for memory in self._cache.values()
            if self._matches(memory, query)
        ]

        if len(cache_matches) >= query.max_results:
            result = cache_matches[: query.max_results]
            logger.info(
                "Memory retrieval completed | source=L1 | matches=%d",
                len(result),
            )
            return result

        persistent_matches = self._repository.search(query)

        merged: dict[str, MemoryItem] = {
            memory.memory_id: memory
            for memory in cache_matches
        }

        for memory in persistent_matches:
            memory.mark_accessed()
            self._cache.put(memory)
            merged[memory.memory_id] = memory

        result = list(merged.values())[: query.max_results]
        logger.info(
            "Memory retrieval completed | source=L1+L2 | matches=%d",
            len(result),
        )
        return result

    @staticmethod
    def _matches(
        memory: MemoryItem,
        query: MemoryQuery,
    ) -> bool:
        """Apply MemoryQuery criteria to an in-memory item."""
        if (
            query.exchange_symbol
            and memory.exchange_symbol.upper()
            != query.exchange_symbol.upper()
        ):
            return False

        if (
            query.query_signature
            and memory.query_signature
            != query.query_signature
        ):
            return False

        if query.topics:
            memory_topics = {
                topic.lower()
                for topic in memory.topics
            }

            requested_topics = {
                topic.lower()
                for topic in query.topics
            }

            if not memory_topics.intersection(requested_topics):
                return False

        if query.query_text:
            return query.query_text.lower() in memory.query.lower()

        return True