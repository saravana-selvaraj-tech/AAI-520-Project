from evaluator_optimizer.memory.models import MemoryItem
from evaluator_optimizer.memory.memory_service import MemoryService
from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.repository import (
    MemoryQuery,
    PickleMemoryRepository,
)


def make_memory(
    memory_id: str,
    symbol: str = "NASDAQ:AAPL",
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        exchange_symbol=symbol,
        query="What was revenue?",
        query_signature="revenue-signature",
        topics=["revenue"],
        research_summary="Revenue increased.",
    )


def test_repository_persists_memory(tmp_path) -> None:
    path = tmp_path / "memory_store.pkl"

    repository = PickleMemoryRepository(path)

    memory = make_memory("memory-1")

    repository.save(memory)

    assert path.exists()

    loaded = repository.get("memory-1")

    assert loaded is not None
    assert loaded.memory_id == "memory-1"
    assert loaded.exchange_symbol == "NASDAQ:AAPL"


def test_repository_survives_new_repository_instance(
    tmp_path,
) -> None:
    path = tmp_path / "memory_store.pkl"

    repository_one = PickleMemoryRepository(path)

    memory = make_memory("memory-1")

    repository_one.save(memory)

    # Simulates a process restart.
    repository_two = PickleMemoryRepository(path)

    loaded = repository_two.get("memory-1")

    assert loaded is not None
    assert loaded.memory_id == "memory-1"


def test_repository_searches_by_symbol(tmp_path) -> None:
    path = tmp_path / "memory_store.pkl"

    repository = PickleMemoryRepository(path)

    repository.save(
        make_memory(
            "aapl-memory",
            "NASDAQ:AAPL",
        )
    )

    repository.save(
        make_memory(
            "msft-memory",
            "NASDAQ:MSFT",
        )
    )

    results = repository.search(
        MemoryQuery(
            exchange_symbol="NASDAQ:AAPL",
        )
    )

    assert len(results) == 1
    assert results[0].memory_id == "aapl-memory"


def test_repository_searches_by_query_signature(
    tmp_path,
) -> None:
    path = tmp_path / "memory_store.pkl"

    repository = PickleMemoryRepository(path)

    repository.save(make_memory("memory-1"))

    results = repository.search(
        MemoryQuery(
            query_signature="revenue-signature",
        )
    )

    assert len(results) == 1
    assert results[0].memory_id == "memory-1"


def test_repository_returns_empty_for_missing_store(
    tmp_path,
) -> None:
    path = tmp_path / "missing.pkl"

    repository = PickleMemoryRepository(path)

    assert repository.load_all() == []


def test_memory_survives_restart_and_is_retrievable(
    tmp_path,
) -> None:
    path = tmp_path / "memory_store.pkl"

    # Process 1.
    repository_one = PickleMemoryRepository(path)
    cache_one = MemoryCache(max_size=5)

    service_one = MemoryService(
        repository=repository_one,
        cache=cache_one,
    )

    service_one.save(
        make_memory("persistent-memory")
    )

    # Process 2.
    repository_two = PickleMemoryRepository(path)
    cache_two = MemoryCache(max_size=5)

    service_two = MemoryService(
        repository=repository_two,
        cache=cache_two,
    )

    context = service_two.search(
        MemoryQuery(
            exchange_symbol="NASDAQ:AAPL",
        )
    )

    assert len(context) == 1
    assert context[0].memory_id == "persistent-memory"