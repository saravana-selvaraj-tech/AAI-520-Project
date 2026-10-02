"""Tests for intelligent memory reuse."""

from datetime import datetime, timezone
from types import SimpleNamespace

from evaluator_optimizer.memory.memory_reuse import (
    MemoryReusePolicy,
)
from evaluator_optimizer.memory.temporal_validity import (
    TemporalValidityChecker,
)


REFERENCE_DATE = datetime(
    2026,
    9,
    29,
    tzinfo=timezone.utc,
)


def build_memory() -> SimpleNamespace:
    """Create a representative memory object."""
    return SimpleNamespace(
        memory_id="memory-001",
        evidence=[
            {
                "evidence_id": "recent-evidence",
                "publication_date": "2026-08-01T00:00:00+00:00",
            },
            {
                "evidence_id": "old-evidence",
                "publication_date": "2024-01-01T00:00:00+00:00",
            },
        ],
        identified_gaps=[
            "Latest quarterly guidance requires verification.",
        ],
    )


def test_memory_reuse_separates_stale_evidence() -> None:
    """Reusable and stale evidence should be separated."""
    policy = MemoryReusePolicy(
        temporal_checker=TemporalValidityChecker(
            max_age_days=365,
        )
    )

    decision = policy.evaluate_memory(
        memory=build_memory(),
        reference_date=REFERENCE_DATE,
    )

    assert decision.reusable is True
    assert decision.reusable_evidence_ids == ["recent-evidence"]
    assert decision.stale_evidence_ids == ["old-evidence"]


def test_memory_research_gaps_are_preserved() -> None:
    """Open research gaps should survive memory evaluation."""
    policy = MemoryReusePolicy(
        temporal_checker=TemporalValidityChecker(
            max_age_days=365,
        )
    )

    decision = policy.evaluate_memory(
        memory=build_memory(),
        reference_date=REFERENCE_DATE,
    )

    assert decision.research_gaps == [
        "Latest quarterly guidance requires verification."
    ]


def test_memory_without_reusable_evidence_is_not_reusable() -> None:
    """A memory containing only stale evidence should not be reused."""
    memory = SimpleNamespace(
        memory_id="memory-old",
        evidence=[
            {
                "evidence_id": "old-evidence",
                "publication_date": "2022-01-01T00:00:00+00:00",
            }
        ],
        identified_gaps=[],
    )

    policy = MemoryReusePolicy(
        temporal_checker=TemporalValidityChecker(
            max_age_days=365,
        )
    )

    decision = policy.evaluate_memory(
        memory=memory,
        reference_date=REFERENCE_DATE,
    )

    assert decision.reusable is False
    assert decision.stale_evidence_ids == ["old-evidence"]


def test_memory_without_id_is_rejected_for_reuse() -> None:
    """A memory without an identifier cannot be safely reused."""
    memory = SimpleNamespace(
        memory_id="",
        evidence=[],
        identified_gaps=[],
    )

    policy = MemoryReusePolicy()

    decision = policy.evaluate_memory(memory)

    assert decision.reusable is False
    assert decision.memory_id == ""