"""Tests for rubric-observable planning, routing, and prompt-chain evidence."""

from agent.graph import AgentGraph
from agent.state import EvidenceRecord
from chains.preprocessing import process_evidence
from planning.planner import TaskType
from tools.providers import InMemoryResearchProvider


def test_prompt_chain_exposes_preprocess_classify_extract_and_summary_outputs():
    evidence = EvidenceRecord(
        source="SEC EDGAR",
        title="Annual report",
        content="Revenue increased 12%. Debt risk remains elevated.",
        ticker="AAPL",
        url="https://example.test/annual",
    )

    result = process_evidence([evidence])[0]

    assert result["preprocessed_text"] == (
        "Revenue increased 12%. Debt risk remains elevated."
    )
    assert result["classification"] == "positive"
    assert result["extracted_facts"]
    assert "Revenue increased 12%" in result["summary"]
    assert set(result["stage_outputs"]) == {
        "preprocess", "classify", "extract", "summarize"
    }


def test_router_trace_names_specialists_and_real_provider_calls():
    evidence = EvidenceRecord(
        source="SEC EDGAR",
        title="Annual report",
        content="Revenue increased 12%. Debt risk remains elevated.",
        ticker="AAPL",
        document_id="annual-1",
        url="https://example.test/annual",
    )
    provider = InMemoryResearchProvider(
        financial_data={"latest_close": 123.0},
        news=[],
        filings=[evidence],
    )

    state = AgentGraph(provider).run(
        "Analyze AAPL earnings, news, SEC filings, and risks",
        ticker="AAPL",
    )

    trace = state.metadata["routing_trace"]
    by_type = {row["task_type"]: row for row in trace}

    assert by_type["financial"]["specialist"] == "Financial Analyst"
    assert by_type["financial"]["provider_method"] == "financial"
    assert by_type["financial"]["result_count"] == 1
    assert by_type["news"]["provider_method"] == "news"
    assert by_type["sec"]["provider_method"] == "filings"
    assert by_type["risk"]["provider_method"] == "local_evidence_analysis"
    assert by_type["risk"]["result_count"] >= 1
    assert by_type["report"]["provider_method"] == "build_context"
    assert by_type["evaluation"]["status"] == "deferred"
    assert state.metadata["generation_trace"]["specialist"] == (
        "Research Synthesis Analyst"
    )
    assert state.metadata["prompt_chain_trace"]
    assert all(stage["status"] == "completed" for stage in state.metadata["prompt_chain_trace"])


def test_evaluation_plan_is_not_marked_completed_before_evaluation_runs():
    state = AgentGraph(InMemoryResearchProvider()).run(
        "Analyze AAPL earnings and risk", ticker="AAPL"
    )
    evaluation_task = next(
        task for task in state.research_plan
        if task.task_type is TaskType.EVALUATION
    )

    assert evaluation_task.status == "deferred"
    assert evaluation_task.task_id not in state.completed_tasks


def test_process_evidence_keeps_a_bounded_preview_not_duplicate_full_text():
    long_text = "Revenue increased 12%. " + ("Detailed filing content. " * 1000)
    evidence = EvidenceRecord(
        source="SEC EDGAR",
        title="Large filing",
        content=long_text,
        ticker="AAPL",
    )

    result = process_evidence(
        [evidence], max_source_chars=800, max_preview_chars=120, max_facts_per_record=2
    )[0]

    assert len(result["preprocessed_text"]) <= 120
    assert len(result["summary"]) <= 120
    assert len(result["extracted_facts"]) <= 2
    assert "raw_text" not in result
    assert len(evidence.content) == len(long_text)  # canonical evidence remains intact


def test_router_caps_news_and_evidence_retained_in_state(monkeypatch):
    from agent.graph import AgentGraph
    from agent.state import EvidenceRecord
    from tools.providers import InMemoryResearchProvider

    monkeypatch.setenv("RESEARCH_MAX_NEWS_RECORDS", "2")
    monkeypatch.setenv("RESEARCH_MAX_EVIDENCE_RECORDS", "2")
    monkeypatch.setenv("RESEARCH_MAX_EVIDENCE_CHARS", "500")
    news = [
        EvidenceRecord(
            source="NewsAPI", title=f"Article {index}",
            content="x" * 2000, ticker="AAPL", document_id=f"news-{index}"
        )
        for index in range(5)
    ]
    provider = InMemoryResearchProvider(news=news)

    state = AgentGraph(provider).run("Analyze AAPL news", ticker="AAPL")

    assert len(state.news_data) <= 2
    assert len(state.evidence) <= 2
    assert all(len(record.content) <= 500 for record in state.evidence)
    assert len(state.processed_evidence) <= 2


def test_fact_extraction_splits_sentences_and_recognizes_numeric_facts():
    from chains.preprocessing import extract_facts

    facts = extract_facts("Background only. Revenue reached 12 billion. Outlook is unclear.")

    assert "Revenue reached 12 billion." in facts
    assert "Background only. Revenue reached 12 billion. Outlook is unclear." not in facts
