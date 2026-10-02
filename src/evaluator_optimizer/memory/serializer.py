"""Serialization abstractions for persistent memory."""

from abc import ABC, abstractmethod
import pickle

from evaluator_optimizer.memory.models import MemoryItem


class MemorySerializer(ABC):
    """Interface for converting MemoryItem objects to and from bytes."""

    @abstractmethod
    def serialize(self, memories: list[MemoryItem]) -> bytes:
        """Serialize memory items."""

    @abstractmethod
    def deserialize(self, data: bytes) -> list[MemoryItem]:
        """Deserialize memory items."""


class PickleMemorySerializer(MemorySerializer):
    """Trusted-local pickle serializer.

    Pickle is appropriate for the local PoC store but must never deserialize
    untrusted files because pickle can execute arbitrary Python code.
    """

    def serialize(self, memories: list[MemoryItem]) -> bytes:
        """Serialize memory items into pickle bytes."""
        return pickle.dumps(memories, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> list[MemoryItem]:
        """Deserialize pickle bytes into memory items."""
        result = pickle.loads(data)

        if not isinstance(result, list):
            raise ValueError("Memory store must contain a list.")

        if not all(isinstance(item, MemoryItem) for item in result):
            raise ValueError("Memory store contains an invalid object.")

        return result
