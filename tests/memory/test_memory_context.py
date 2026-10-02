from evaluator_optimizer.memory.context import MemoryContext
from evaluator_optimizer.memory.models import MemoryItem


def make_memory(memory_id: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        exchange_symbol="NASDAQ:AAPL",
        query="Revenue?",
        query_signature="revenue",
    )


def test_memory_context_adds_memory_once() -> None:
    context = MemoryContext()

    memory = make_memory("memory-1")

    context.add_memory(memory)
    context.add_memory(memory)

    assert len(context.memories) == 1


def test_memory_context_tracks_reused_evidence() -> None:
    context = MemoryContext()

    context.add_reused_evidence("evidence-1")
    context.add_reused_evidence("evidence-1")

    assert context.reused_evidence_ids == [
        "evidence-1",
    ]


def test_memory_context_tracks_stale_evidence() -> None:
    context = MemoryContext()

    context.add_stale_evidence("evidence-2024")

    assert context.stale_evidence_ids == [
        "evidence-2024",
    ]


def test_memory_context_tracks_research_gaps() -> None:
    context = MemoryContext()

    context.add_research_gap(
        "Latest segment revenue unavailable."
    )

    context.add_research_gap(
        "Latest segment revenue unavailable."
    )

    assert context.research_gaps == [
        "Latest segment revenue unavailable.",
    ]