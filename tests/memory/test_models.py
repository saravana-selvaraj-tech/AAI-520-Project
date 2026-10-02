from evaluator_optimizer.memory.models import MemoryItem


def test_memory_item_can_be_created() -> None:
    memory = MemoryItem(
        memory_id="memory-1",
        exchange_symbol="NASDAQ:AAPL",
        query="What was Apple's revenue?",
        query_signature="abc123",
        research_summary="Revenue increased.",
        key_findings=["Revenue increased."],
        claims=["Revenue was $416B."],
        evidence=["FY2025 annual report"],
        identified_gaps=[],
    )

    assert memory.memory_id == "memory-1"
    assert memory.exchange_symbol == "NASDAQ:AAPL"
    assert memory.access_count == 0
    assert memory.last_accessed_at is None


def test_memory_item_tracks_access() -> None:
    memory = MemoryItem(
        memory_id="memory-1",
        exchange_symbol="NASDAQ:AAPL",
        query="Revenue?",
        query_signature="abc123",
    )

    memory.mark_accessed()

    assert memory.access_count == 1
    assert memory.last_accessed_at is not None