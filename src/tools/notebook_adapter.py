"""Adapters for notebook and live research data outputs."""

from __future__ import annotations

import os
from typing import Any

from agent.state import EvidenceRecord


class NotebookResearchProvider:
    """Expose notebook DataFrames through the research provider protocol."""

    def __init__(
        self,
        ticker: str,
        stock_data: Any | None = None,
        news_data: Any | None = None,
        company_name: str | None = None,
        filings: list[EvidenceRecord] | None = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.stock_data = stock_data
        self.news_data = news_data
        self.company_name = company_name or self.ticker
        self._filings = list(filings or [])

    def financial(self, ticker: str | None, query: str) -> dict[str, object]:
        """Summarize the supplied market DataFrame."""
        data = self.stock_data
        if data is None or len(data) == 0:
            return {"ticker": self.ticker, "company": self.company_name}

        close = _column(data, "Close")
        result: dict[str, object] = {
            "ticker": self.ticker,
            "company": self.company_name,
            "observations": len(data),
        }
        if close is not None:
            numeric_close = close.astype(float)
            result["latest_close"] = round(float(numeric_close.iloc[-1]), 4)
            result["period_return"] = round(
                float((numeric_close.iloc[-1] / numeric_close.iloc[0]) - 1),
                6,
            )
        volume = _column(data, "Volume")
        if volume is not None:
            result["latest_volume"] = int(volume.iloc[-1])
        return result

    def news(self, ticker: str | None, query: str) -> list[EvidenceRecord]:
        """Convert NewsAPI-shaped rows into provenance-preserving records."""
        if self.news_data is None:
            return []
        records = []
        for index, row in self.news_data.iterrows():
            title = _value(row, "title", f"News item {index + 1}")
            content = " ".join(
                str(_value(row, field, ""))
                for field in ("description", "content")
                if _value(row, field, "")
            ).strip()
            if not content:
                content = title
            records.append(
                EvidenceRecord(
                    source=str(_value(row, "source", "NewsAPI")),
                    title=title,
                    content=content,
                    published_at=_value(row, "publishedAt"),
                    ticker=self.ticker,
                    url=_value(row, "url"),
                    document_id=_value(row, "url", f"news-{index}"),
                )
            )
        return records

    def filings(self, ticker: str | None, query: str) -> list[EvidenceRecord]:
        """Return filings supplied by an SEC adapter, if available."""
        return list(self._filings)

    @classmethod
    def from_live(cls, ticker: str, period: str = "1y") -> NotebookResearchProvider:
        """Collect live Yahoo Finance and NewsAPI data using environment secrets."""
        import requests
        import yfinance as yf

        stock = yf.Ticker(ticker)
        stock_data = stock.history(period=period).reset_index()
        company_name = stock.info.get("longName", ticker)
        api_key = os.getenv("NEWS_API_KEY")
        news_data = []
        if api_key:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "apiKey": api_key,
                    "language": "en",
                    "pageSize": 50,
                    "sortBy": "relevancy",
                },
                timeout=30,
            )
            response.raise_for_status()
            news_data = response.json().get("articles", [])
            import pandas as pd
            news_data = pd.DataFrame(news_data)
        return cls(ticker, stock_data, news_data, company_name)


def _column(data: Any, name: str) -> Any | None:
    if name not in data.columns:
        return None
    return data[name].dropna()


def _value(row: Any, name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    if value is None:
        return default
    return value.get("name", default) if isinstance(value, dict) else value