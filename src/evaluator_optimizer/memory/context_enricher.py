"""Enrich MemoryContext with W2.4 reuse decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .context import MemoryContext
from .memory_reuse import MemoryReusePolicy


class MemoryContextEnricher:
    """Apply intelligent memory reuse decisions to MemoryContext."""

    def __init__(
        self,
        reuse_policy: MemoryReusePolicy | None = None,
    ) -> None:
        """Initialize the context enricher."""
        self.reuse_policy = reuse_policy or MemoryReusePolicy()

    def enrich(
        self,
        context: MemoryContext,
        memories: Iterable[Any],
        reference_date: datetime | None = None,
    ) -> MemoryContext:
        """Populate context with reusable and stale evidence metadata."""
        decisions = self.reuse_policy.evaluate_memories(
            memories=memories,
            reference_date=reference_date,
        )

        for decision in decisions:
            for evidence_id in decision.reusable_evidence_ids:
                context.add_reused_evidence(evidence_id)

            for evidence_id in decision.stale_evidence_ids:
                context.add_stale_evidence(evidence_id)

            for research_gap in decision.research_gaps:
                context.add_research_gap(research_gap)

        return context