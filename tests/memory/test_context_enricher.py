"""Tests for MemoryContext enrichment."""

from datetime import datetime, timezone
from types import SimpleNamespace

from evaluator_optimizer.memory.context import MemoryContext
from evaluator_optimizer.memory.context_enricher import (
    MemoryContextEnricher,
)
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


def test_context_tracks_reused_and_stale_evidence() -> None:
    """Context should record both reusable and stale evidence."""
    memory = SimpleNamespace(
        memory_id="memory-001",
        evidence=[
            {
                "evidence_id": "recent",
                "publication_date": "2026-08-01T00:00:00+00:00",
            },
            {
                "evidence_id": "stale",
                "publication_date": "2024-01-01T00:00:00+00:00",
            },
        ],
        identified_gaps=["Verify latest earnings guidance."],
    )

    policy = MemoryReusePolicy(
        temporal_checker=TemporalValidityChecker(
            max_age_days=365,
        )
    )

    enricher = MemoryContextEnricher(
        reuse_policy=policy,
    )

    context = MemoryContext()

    result = enricher.enrich(
        context=context,
        memories=[memory],
        reference_date=REFERENCE_DATE,
    )

    assert result is context
    assert "recent" in context.reused_evidence_ids
    assert "stale" in context.stale_evidence_ids
    assert "Verify latest earnings guidance." in (
        context.research_gaps
    )