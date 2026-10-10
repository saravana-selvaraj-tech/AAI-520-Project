"""Provider boundary for financial, news, and filing retrieval."""

from __future__ import annotations

from typing import Protocol

from agent.state import EvidenceRecord


class ResearchProvider(Protocol):
    """Cohort 1 adapter required by the Cohort 2 graph."""

    def financial(self, ticker: str | None, query: str) -> dict[str, object]: ...
    def news(self, ticker: str | None, query: str) -> list[EvidenceRecord]: ...
    def filings(self, ticker: str | None, query: str) -> list[EvidenceRecord]: ...


class InMemoryResearchProvider:
    """Offline provider useful for tests, notebooks, and local demos."""

    def __init__(self, financial_data=None, news=None, filings=None) -> None:
        self._financial_data = financial_data or {}
        self._news = list(news or [])
        self._filings = list(filings or [])

    def financial(self, ticker, query):
        return dict(self._financial_data)

    def news(self, ticker, query):
        return list(self._news)

    def filings(self, ticker, query):
        return list(self._filings)