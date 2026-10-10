"""Shared state contract for the Cohort 2 research workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """A source-backed fact passed from Cohort 1 to downstream cohorts."""

    source: str
    title: str = ""
    content: str
    published_at: str | None = None
    ticker: str | None = None
    url: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None

    def citation(self) -> str:
        """Return a compact, human-readable provenance label."""
        label = self.title or self.source
        return f"{label} ({self.source})"


class ResearchState(BaseModel):
    """Mutable data contract shared by planner, tools, chains, and graph."""

    user_query: str
    ticker: str | None = None
    research_plan: list[Any] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    retrieved_documents: list[EvidenceRecord] = Field(default_factory=list)
    financial_data: dict[str, Any] = Field(default_factory=dict)
    news_data: list[EvidenceRecord] = Field(default_factory=list)
    filings: list[EvidenceRecord] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    processed_evidence: list[dict[str, Any]] = Field(default_factory=list)
    draft_report: str = ""
    evaluation: Any = None
    revision_count: int = 0
    memory: list[Any] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)