"""
Persistent memory repository.

PickleMemoryRepository provides a simple local persistence mechanism
for the academic project.

The repository stores MemoryItem objects at:

    data/memory/memory_store.pkl

The implementation intentionally hides persistence details from
MemoryService.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from pydantic import BaseModel, Field

from evaluator_optimizer.config import (
    DEFAULT_MEMORY_MAX_RESULTS,
    DEFAULT_MEMORY_STORE_PATH,
)
from evaluator_optimizer.logging_config import get_logger
from evaluator_optimizer.memory.models import MemoryItem
from evaluator_optimizer.memory.serializer import (
    MemorySerializer,
    PickleMemorySerializer,
)


logger = get_logger("memory.repository")


class MemoryQuery(BaseModel):
    """
    Search criteria for persistent memory.
    """

    exchange_symbol: str | None = None

    topics: list[str] = Field(
        default_factory=list,
    )

    query_signature: str | None = None

    query_text: str | None = None

    max_results: int = DEFAULT_MEMORY_MAX_RESULTS


class MemoryRepository:
    """
    Repository interface for persistent memory.
    """

    def save(self, memory: MemoryItem) -> None:
        """Persist a memory item."""
        raise NotImplementedError

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory by identifier."""
        raise NotImplementedError

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """Search persistent memories."""
        raise NotImplementedError

    def load_all(self) -> list[MemoryItem]:
        """Load all persisted memories."""
        raise NotImplementedError


class PickleMemoryRepository(MemoryRepository):
    """
    Local pickle-backed MemoryRepository.

    This implementation is appropriate for the project's academic
    prototype. A production implementation could later replace this
    repository with a database or vector store without changing the
    MemoryService contract.
    """

    def __init__(
        self,
        file_path: str | Path = DEFAULT_MEMORY_STORE_PATH,
        serializer: MemorySerializer | None = None,
    ) -> None:
        """Initialize the repository."""
        self._file_path = Path(file_path)
        self._serializer = serializer or PickleMemorySerializer()

    @property
    def file_path(self) -> Path:
        """Return repository storage path."""
        return self._file_path

    def save(self, memory: MemoryItem) -> None:
        """Insert or update a memory."""
        logger.info(
            "Memory persistence started | memory_id=%s",
            memory.memory_id,
        )
        memories = self.load_all()

        existing_index = next(
            (
                index
                for index, item in enumerate(memories)
                if item.memory_id == memory.memory_id
            ),
            None,
        )

        if existing_index is None:
            memories.append(memory)
        else:
            memories[existing_index] = memory

        self._write(memories)
        logger.info(
            "Memory persistence completed | memory_id=%s",
            memory.memory_id,
        )

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a memory by ID."""
        return next(
            (
                memory
                for memory in self.load_all()
                if memory.memory_id == memory_id
            ),
            None,
        )

    def search(
        self,
        query: MemoryQuery,
    ) -> list[MemoryItem]:
        """Search memories using simple deterministic matching."""
        logger.info(
            "Memory retrieval started | max_results=%d",
            query.max_results,
        )
        memories = self.load_all()

        matches = [
            memory
            for memory in memories
            if self._matches(memory, query)
        ]

        result = matches[: query.max_results]
        logger.info(
            "Memory retrieval completed | matches=%d",
            len(result),
        )
        return result

    def load_all(self) -> list[MemoryItem]:
        """Load all memories from disk."""
        if not self._file_path.exists():
            return []

        with self._file_path.open("rb") as file:
            return self._serializer.deserialize(file.read())

    def _write(self, memories: list[MemoryItem]) -> None:
        """Persist the complete memory collection."""
        self._file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self._file_path.with_suffix(".tmp")

        with temporary_path.open("wb") as file:
            file.write(self._serializer.serialize(memories))

        temporary_path.replace(self._file_path)

    @staticmethod
    def _matches(
        memory: MemoryItem,
        query: MemoryQuery,
    ) -> bool:
        """Determine whether a memory satisfies a search query."""
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
            search_text = query.query_text.lower()

            if search_text not in memory.query.lower():
                return False

        return True