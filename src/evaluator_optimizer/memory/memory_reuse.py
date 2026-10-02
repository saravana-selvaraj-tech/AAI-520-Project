"""Intelligent memory reuse policy.

Determines which previously stored memories can safely contribute to a
current investment research task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from .temporal_validity import TemporalValidityChecker


@dataclass
class MemoryReuseDecision:
    """Decision describing how a memory should be used."""

    memory_id: str
    reusable: bool
    stale_evidence_ids: list[str] = field(default_factory=list)
    reusable_evidence_ids: list[str] = field(default_factory=list)
    research_gaps: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class MemoryReusePolicy:
    """Apply relevance and temporal-validity rules to stored memories."""

    def __init__(
        self,
        temporal_checker: TemporalValidityChecker | None = None,
    ) -> None:
        """Initialize the memory reuse policy."""
        self.temporal_checker = (
            temporal_checker or TemporalValidityChecker()
        )

    def evaluate_memory(
        self,
        memory: Any,
        reference_date: datetime | None = None,
    ) -> MemoryReuseDecision:
        """Determine whether a memory can be reused.

        A memory remains useful even when some of its evidence is stale.
        Stale evidence is excluded from reuse while non-stale evidence and
        research gaps remain available.
        """
        memory_id = str(getattr(memory, "memory_id", ""))

        if not memory_id:
            return MemoryReuseDecision(
                memory_id="",
                reusable=False,
                reasons=["Memory does not contain a memory_id."],
            )

        evidence = getattr(memory, "evidence", []) or []

        stale_ids: list[str] = []
        reusable_ids: list[str] = []

        for index, item in enumerate(evidence):
            evidence_id = self._evidence_id(item, index)

            result = self.temporal_checker.evaluate(
                evidence_id=evidence_id,
                evidence=item,
                reference_date=reference_date,
            )

            if result.is_stale:
                stale_ids.append(evidence_id)
            else:
                reusable_ids.append(evidence_id)

        research_gaps = list(
            getattr(memory, "identified_gaps", []) or []
        )

        reasons: list[str] = []

        if reusable_ids:
            reasons.append(
                f"{len(reusable_ids)} evidence item(s) are reusable."
            )

        if stale_ids:
            reasons.append(
                f"{len(stale_ids)} evidence item(s) are stale."
            )

        if research_gaps:
            reasons.append(
                f"{len(research_gaps)} research gap(s) remain open."
            )

        return MemoryReuseDecision(
            memory_id=memory_id,
            reusable=bool(reusable_ids or research_gaps),
            stale_evidence_ids=stale_ids,
            reusable_evidence_ids=reusable_ids,
            research_gaps=research_gaps,
            reasons=reasons,
        )

    def evaluate_memories(
        self,
        memories: Iterable[Any],
        reference_date: datetime | None = None,
    ) -> list[MemoryReuseDecision]:
        """Evaluate multiple memories."""
        return [
            self.evaluate_memory(
                memory=memory,
                reference_date=reference_date,
            )
            for memory in memories
        ]

    @staticmethod
    def _evidence_id(evidence: Any, index: int) -> str:
        """Return a stable evidence identifier."""
        if isinstance(evidence, dict):
            for field_name in ("evidence_id", "id", "source_id"):
                value = evidence.get(field_name)
                if value:
                    return str(value)

        for field_name in ("evidence_id", "id", "source_id"):
            value = getattr(evidence, field_name, None)
            if value:
                return str(value)

        return f"evidence-{index}"