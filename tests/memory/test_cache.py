import pytest

from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.models import MemoryItem


def make_memory(memory_id: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        exchange_symbol="NASDAQ:AAPL",
        query=f"Question {memory_id}",
        query_signature=memory_id,
    )


def test_cache_stores_and_retrieves_memory() -> None:
    cache = MemoryCache(max_size=2)
    memory = make_memory("one")

    cache.put(memory)

    assert cache.get("one") is memory
    assert cache.size == 1


def test_cache_evicts_least_recently_used_memory() -> None:
    cache = MemoryCache(max_size=2)

    first = make_memory("one")
    second = make_memory("two")
    third = make_memory("three")

    cache.put(first)
    cache.put(second)

    # Promote "one".
    assert cache.get("one") is first

    cache.put(third)

    assert cache.get("one") is first
    assert cache.get("two") is None
    assert cache.get("three") is third


def test_cache_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        MemoryCache(max_size=0)