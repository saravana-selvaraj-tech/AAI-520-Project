"""
Memory domain models.

MemoryItem represents reusable knowledge learned from a previous
investment-research execution.

The memory stores research findings, claims, evidence, identified
gaps, and evaluation information so that subsequent runs can reuse
prior work rather than merely replaying previous text.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class MemoryItem(BaseModel):
    """
    Persistent research memory.

    A MemoryItem represents a reusable snapshot of a prior research
    execution.
    """

    memory_id: str

    exchange_symbol: str

    query: str

    query_signature: str

    topics: list[str] = Field(
        default_factory=list,
    )

    research_summary: str = ""

    key_findings: list[str] = Field(
        default_factory=list,
    )

    claims: list[str] = Field(
        default_factory=list,
    )

    evidence: list[str] = Field(
        default_factory=list,
    )

    identified_gaps: list[str] = Field(
        default_factory=list,
    )

    evaluation_summary: dict[str, float] = Field(
        default_factory=dict,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    last_accessed_at: datetime | None = None

    access_count: int = 0

    def mark_accessed(self) -> None:
        """Update access metadata after memory retrieval."""
        self.last_accessed_at = utc_now()
        self.access_count += 1
        self.updated_at = utc_now()