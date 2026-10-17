"""Adapters for notebook and live research data outputs."""

from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
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
        max_filings: int = 4,
        sec_user_agent: str | None = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.stock_data = stock_data
        self.news_data = news_data
        self.company_name = company_name or self.ticker
        self._filings = list(filings or [])
        self.max_filings = max(1, int(max_filings))
        self.sec_user_agent = (
            sec_user_agent
            or os.getenv("SEC_USER_AGENT")
            or "AAI520 Investment Research Agent contact: research@example.com"
        )

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
        max_news_records = max(1, int(os.getenv("RESEARCH_MAX_NEWS_RECORDS", "12")))
        max_content_chars = max(500, int(os.getenv("RESEARCH_MAX_EVIDENCE_CHARS", "8000")))
        for index, row in self.news_data.head(max_news_records).iterrows():
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
                    content=content[:max_content_chars],
                    published_at=_value(row, "publishedAt"),
                    ticker=self.ticker,
                    url=_value(row, "url"),
                    document_id=_value(row, "url", f"news-{index}"),
                )
            )
        return records

    def filings(self, ticker: str | None, query: str) -> list[EvidenceRecord]:
        """Fetch recent SEC EDGAR filings and convert them to evidence records.

        Explicitly supplied records are retained as an offline/test override.
        Otherwise the method resolves the ticker to a CIK through SEC's public
        company-ticker mapping, reads the submissions index, and fetches each
        selected primary filing document. SEC requires a descriptive User-Agent;
        set ``SEC_USER_AGENT`` to include your application's name and a contact.
        """
        if self._filings:
            return list(self._filings)

        import requests

        symbol = (ticker or self.ticker).strip().upper()
        if not symbol:
            return []

        headers = {"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        ticker_response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=30,
        )
        ticker_response.raise_for_status()
        ticker_rows = ticker_response.json().values()
        company = next(
            (row for row in ticker_rows if str(row.get("ticker", "")).upper() == symbol),
            None,
        )
        if company is None:
            # Permit a zero-padded or unpadded CIK as input as well as a ticker.
            if symbol.isdigit():
                cik = symbol.zfill(10)
                company_name = self.company_name
            else:
                return []
        else:
            cik = str(company["cik_str"]).zfill(10)
            company_name = str(company.get("title") or self.company_name)

        submissions_response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=headers,
            timeout=30,
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_documents = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        accepted_forms = {"10-K", "10-Q", "8-K", "20-F", "6-K", "40-F", "S-1"}
        requested_forms = _requested_filing_forms(query)
        records: list[EvidenceRecord] = []

        for index, form in enumerate(forms):
            if form not in accepted_forms:
                continue
            if requested_forms and form not in requested_forms:
                continue
            if index >= len(accession_numbers) or index >= len(primary_documents):
                continue

            accession = accession_numbers[index]
            primary_document = primary_documents[index]
            accession_path = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_path}/{primary_document}"
            )
            try:
                document_response = requests.get(
                    filing_url, headers=headers, timeout=45
                )
                document_response.raise_for_status()
            except requests.RequestException:
                # One unavailable exhibit/document should not discard other filings.
                continue

            content = _html_to_text(document_response.text)
            if not content:
                continue
            filing_date = filing_dates[index] if index < len(filing_dates) else None
            description = descriptions[index] if index < len(descriptions) else ""
            title = f"{company_name} {form} filing"
            if filing_date:
                title += f" filed {filing_date}"
            if description:
                title += f" — {description}"
            records.append(
                EvidenceRecord(
                    source="SEC EDGAR",
                    title=title,
                    content=content[:20000],
                    published_at=filing_date,
                    ticker=symbol,
                    url=filing_url,
                    document_id=accession,
                )
            )
            if len(records) >= self.max_filings:
                break

        return records

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



class _TextExtractor(HTMLParser):
    """Small HTML-to-text converter for SEC filing documents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._retained_chars = 0
        self._max_chars = 20_000

    def _append(self, value: str) -> None:
        remaining = self._max_chars - self._retained_chars
        if remaining <= 0:
            return
        fragment = value[:remaining]
        self.parts.append(fragment)
        self._retained_chars += len(fragment)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1
        elif tag.lower() in {"p", "div", "br", "tr", "td", "th", "li", "section"}:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "div", "tr", "li", "section"}:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._retained_chars >= self._max_chars:
            return
        self._append(data)


def _html_to_text(document: str) -> str:
    """Normalize filing HTML into readable, bounded plain text."""
    parser = _TextExtractor()
    parser.feed(document)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _requested_filing_forms(query: str) -> set[str]:
    """Return form filters only when a query explicitly names filing types."""
    normalized = (query or "").upper().replace(" ", "")
    forms = {"10-K", "10-Q", "8-K", "20-F", "6-K", "40-F", "S-1"}
    return {form for form in forms if form in normalized}

def _column(data: Any, name: str) -> Any | None:
    if name not in data.columns:
        return None
    return data[name].dropna()


def _value(row: Any, name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    if value is None:
        return default
    return value.get("name", default) if isinstance(value, dict) else value