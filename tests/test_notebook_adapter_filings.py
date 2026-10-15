"""Tests for SEC EDGAR filing collection in NotebookResearchProvider."""

from unittest.mock import Mock, patch

from tools.notebook_adapter import NotebookResearchProvider
from agent.state import EvidenceRecord


def _response(payload=None, text=""):
    response = Mock()
    response.json.return_value = payload or {}
    response.text = text
    response.raise_for_status.return_value = None
    return response


def test_filings_fetches_sec_submissions_and_builds_evidence_records():
    ticker_map = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    submissions = {"filings": {"recent": {
        "form": ["10-K", "4"],
        "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
        "filingDate": ["2024-11-01", "2024-11-02"],
        "primaryDocument": ["annual.htm", "form4.htm"],
        "primaryDocDescription": ["Annual report", "Insider filing"],
    }}}
    with patch("requests.get", side_effect=[
        _response(ticker_map), _response(submissions), _response(text="<html><body><h1>Revenue</h1><p>Revenue grew 8%.</p><script>hidden</script></body></html>"),
    ]) as get:
        provider = NotebookResearchProvider("AAPL", sec_user_agent="Test contact test@example.com")
        records = provider.filings("AAPL", "annual report")

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, EvidenceRecord)
    assert record.source == "SEC EDGAR"
    assert record.ticker == "AAPL"
    assert record.published_at == "2024-11-01"
    assert record.document_id == "0000320193-24-000001"
    assert record.url.endswith("/000032019324000001/annual.htm")
    assert "Revenue grew 8%." in record.content
    assert "hidden" not in record.content
    assert get.call_count == 3


def test_filings_explicit_records_override_live_sec_calls():
    supplied = EvidenceRecord(source="SEC EDGAR", title="Offline filing", content="Text", ticker="MSFT")
    provider = NotebookResearchProvider("MSFT", filings=[supplied])
    with patch("requests.get") as get:
        result = provider.filings("MSFT", "10-K")
    assert result == [supplied]
    get.assert_not_called()


def test_filings_returns_empty_for_unknown_ticker():
    with patch("requests.get", return_value=_response({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})):
        provider = NotebookResearchProvider("UNKNOWN")
        assert provider.filings("UNKNOWN", "recent filings") == []


def test_filings_query_can_filter_form_type():
    from tools.notebook_adapter import _requested_filing_forms
    assert _requested_filing_forms("show 10-K annual reports") == {"10-K"}
    assert _requested_filing_forms("latest financial filings") == set()
