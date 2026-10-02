from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.memory_service import MemoryService
from evaluator_optimizer.memory.models import MemoryItem
from evaluator_optimizer.memory.repository import (
    MemoryQuery,
    PickleMemoryRepository,
)


def make_memory(memory_id: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        exchange_symbol="NASDAQ:AAPL",
        query="What was revenue?",
        query_signature="revenue-signature",
        topics=["revenue"],
    )


def test_service_saves_to_l1_and_l2(tmp_path) -> None:
    repository = PickleMemoryRepository(
        tmp_path / "memory.pkl"
    )
    cache = MemoryCache(max_size=5)

    service = MemoryService(
        repository=repository,
        cache=cache,
    )

    memory = make_memory("memory-1")

    service.save(memory)

    assert cache.get("memory-1") is not None
    assert repository.get("memory-1") is not None


def test_service_retrieves_from_l1(tmp_path) -> None:
    repository = PickleMemoryRepository(
        tmp_path / "memory.pkl"
    )
    cache = MemoryCache(max_size=5)

    service = MemoryService(
        repository=repository,
        cache=cache,
    )

    memory = make_memory("memory-1")

    service.save(memory)

    retrieved = service.get("memory-1")

    assert retrieved is memory


def test_service_promotes_l2_hit_into_l1(tmp_path) -> None:
    repository = PickleMemoryRepository(
        tmp_path / "memory.pkl"
    )

    original = make_memory("memory-1")
    repository.save(original)

    cache = MemoryCache(max_size=5)

    service = MemoryService(
        repository=repository,
        cache=cache,
    )

    retrieved = service.get("memory-1")

    assert retrieved is not None
    assert retrieved.memory_id == "memory-1"
    assert cache.get("memory-1") is not None


def test_service_searches_relevant_memory(tmp_path) -> None:
    repository = PickleMemoryRepository(
        tmp_path / "memory.pkl"
    )

    cache = MemoryCache(max_size=5)

    service = MemoryService(
        repository=repository,
        cache=cache,
    )

    service.save(make_memory("memory-1"))

    results = service.search(
        MemoryQuery(
            exchange_symbol="NASDAQ:AAPL",
            query_signature="revenue-signature",
        )
    )

    assert len(results) == 1
    assert results[0].memory_id == "memory-1"