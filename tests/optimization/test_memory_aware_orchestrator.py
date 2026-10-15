"""Tests for the Cohort 2/Cohort 3 memory-aware orchestration boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluator_optimizer.orchestration import memory_aware_orchestrator
from evaluator_optimizer.orchestration.memory_aware_orchestrator import (
    MemoryAwareResearchService,
)


def _memory(
    memory_id: str,
    gaps: list[str] | None = None,
    *,
    research_summary: str = "Prior research summary",
    evidence: list[str] | None = None,
    key_findings: list[str] | None = None,
):
    """Build a lightweight memory matching fields consumed by orchestration."""
    return SimpleNamespace(
        memory_id=memory_id,
        identified_gaps=gaps or [],
        research_summary=research_summary,
        evidence=evidence or [],
        key_findings=key_findings or [],
    )


def _trace():
    trace = MagicMock()
    trace.optimized_answer = "Optimized response"
    trace.initial_answer = "Initial response"
    trace.final_evaluation = {"overall_score": 0.9}
    trace.memory_retrieved = []
    trace.memory_applied = []
    return trace


def _service(memories=None):
    router = MagicMock()
    memory_service = MagicMock()
    memory_service.search.return_value = memories or []

    memory_builder = MagicMock()
    memory_builder.create_query_signature.return_value = "query-signature"

    optimization_controller = MagicMock()
    optimization_controller.run.return_value = _trace()

    service = MemoryAwareResearchService(
        router_service=router,
        memory_service=memory_service,
        memory_builder=memory_builder,
        optimization_controller=optimization_controller,
    )
    return (
        service,
        router,
        memory_service,
        memory_builder,
        optimization_controller,
    )


def test_run_gets_initial_response_from_cohort_2_router():
    """The Cohort 2 router response is the answer evaluated by Cohort 3."""
    (
        service,
        router,
        _memory_service,
        _memory_builder,
        optimization_controller,
    ) = _service()

    router.route.return_value = "Cohort 2 initial response"

    service.run(
        exchange_symbol="AAPL",
        query="Assess the company's recent earnings.",
        context="RAG research context",
    )

    # With no retrieved memory, the supplied context is preserved unchanged.
    router.route.assert_called_once_with(
        query="Assess the company's recent earnings.",
        context="RAG research context",
    )
    optimization_controller.run.assert_called_once_with(
        query="Assess the company's recent earnings.",
        answer="Cohort 2 initial response",
        context="RAG research context",
    )


def test_run_no_longer_accepts_external_answer():
    """The public API requires Cohort 2 to produce the initial response."""
    service, router, *_ = _service()
    router.route.return_value = "Router-generated response"

    with pytest.raises(TypeError):
        service.run(
            exchange_symbol="MSFT",
            query="What is the current market trend?",
            answer="Externally supplied response",
            context="Market RAG context",
        )


def test_run_retrieves_memory_before_router_and_persists_final_response():
    """Memory retrieval, generation, optimization, and persistence are ordered."""
    memory = _memory(
        "mem-001",
        ["Needs more valuation evidence"],
        research_summary="Prior valuation analysis",
        evidence=["Prior P/E ratio was 25x"],
    )
    (
        service,
        router,
        memory_service,
        memory_builder,
        optimization_controller,
    ) = _service(memories=[memory])

    router.route.return_value = "Initial Cohort 2 response"
    trace = optimization_controller.run.return_value
    events = []

    memory_service.search.side_effect = (
        lambda query: events.append("memory_search") or [memory]
    )
    router.route.side_effect = (
        lambda **kwargs: events.append("router")
        or "Initial Cohort 2 response"
    )
    optimization_controller.run.side_effect = (
        lambda **kwargs: events.append("optimization") or trace
    )
    memory_service.save.side_effect = lambda value: events.append("save")

    service.run(
        exchange_symbol="NVDA",
        query="Evaluate valuation.",
        context="Financial RAG context",
        topics=["valuation"],
    )

    assert events == [
        "memory_search",
        "router",
        "optimization",
        "save",
    ]
    memory_builder.build.assert_called_once()
    assert (
        memory_builder.build.call_args.kwargs["research_summary"]
        == "Optimized response"
    )

    # Retrieved memory is now passed into both generation and optimization.
    router_context = router.route.call_args.kwargs["context"]
    optimizer_context = optimization_controller.run.call_args.kwargs["context"]
    for enriched_context in (router_context, optimizer_context):
        assert "Financial RAG context" in enriched_context
        assert "Prior research memory (verify against current evidence)" in enriched_context
        assert "Prior valuation analysis" in enriched_context
        assert "Prior P/E ratio was 25x" in enriched_context
        assert "Needs more valuation evidence" in enriched_context


def test_run_records_retrieved_and_applied_memory_ids():
    """The trace retains memory traceability."""
    memory = _memory("mem-123")
    service, router, _, _, _ = _service(memories=[memory])
    router.route.return_value = "Initial response"

    trace, memory_context = service.run(
        exchange_symbol="AMZN",
        query="Analyze revenue growth.",
        context="Revenue context",
    )

    assert memory_context.memories == [memory]
    # The router receives the memory-enriched context, not only the caller's text.
    routed_context = router.route.call_args.kwargs["context"]
    assert "Revenue context" in routed_context
    assert "Prior research memory (verify against current evidence)" in routed_context
    assert "Prior research summary" in routed_context
    trace.record_memory_retrieved.assert_called_once_with(["mem-123"])
    trace.record_memory_applied.assert_called_once_with(["mem-123"])



def test_memory_context_is_added_to_router_and_optimizer_contexts():
    """Historical summary, evidence, and gaps are supplied to both stages."""
    memory = _memory(
        "mem-context",
        ["Confirm latest quarterly figures"],
        research_summary="Revenue grew 8% in FY2025.",
        evidence=["FY2025 annual report"],
    )
    service, router, _, _, optimization_controller = _service(memories=[memory])
    router.route.return_value = "Initial answer"

    service.run(
        exchange_symbol="AAPL",
        query="Summarize revenue growth.",
        context="Current retrieval returned limited details.",
        topics=["revenue"],
    )

    router_context = router.route.call_args.kwargs["context"]
    optimizer_context = optimization_controller.run.call_args.kwargs["context"]
    assert router_context == optimizer_context
    assert "Current retrieval returned limited details." in router_context
    assert "Revenue grew 8% in FY2025." in router_context
    assert "FY2025 annual report" in router_context
    assert "Confirm latest quarterly figures" in router_context
    assert "verify against current evidence" in router_context


def test_run_question_uses_memory_router_and_persists_optimized_answer(monkeypatch):
    """Retrieved memory seeds routing and is available to optimization."""
    memory = _memory("mem-question", ["Verify current valuation"])
    service, _, memory_service, memory_builder, optimization_controller = (
        _service(memories=[memory])
    )
    provider = object()
    service._data_provider = provider
    graph = MagicMock()
    router = graph.get_router_service.return_value
    router.route.return_value = "Memory-seeded initial response"
    trace = optimization_controller.run.return_value
    monkeypatch.setattr(
        memory_aware_orchestrator, "AgentGraph", MagicMock(return_value=graph)
    )

    result_trace, memory_context = service.run_question(
        question="Assess AAPL valuation.",
        exchange_symbol=" aapl ",
    )

    memory_aware_orchestrator.AgentGraph.assert_called_once_with(
        provider=provider,
        answer_generator=None,
    )
    graph.run.assert_not_called()
    router.route.assert_called_once()
    assert router.route.call_args.kwargs["query"] == "Assess AAPL valuation."
    assert "Prior research summary" in router.route.call_args.kwargs["context"]
    optimization_controller.set_router_service.assert_called_once_with(router)
    optimization_controller.run.assert_called_once_with(
        query="Assess AAPL valuation.",
        answer="Memory-seeded initial response",
        context=router.route.call_args.kwargs["context"],
    )
    assert memory_context.memories == [memory]
    assert result_trace is trace
    memory_builder.build.assert_called_once_with(
        exchange_symbol="AAPL",
        query="Assess AAPL valuation.",
        research_summary="Optimized response",
        key_findings=[],
        claims=[],
        evidence=[],
        identified_gaps=["Verify current valuation"],
        evaluation_results=trace.final_evaluation,
        topics=None,
    )
    memory_service.save.assert_called_once()
    trace.record_memory_retrieved.assert_called_once_with(["mem-question"])
    trace.record_memory_applied.assert_called_once_with(["mem-question"])


def test_run_question_runs_agent_graph_when_memory_is_empty(monkeypatch):
    """Without memory, AgentGraph performs retrieval and creates the draft."""
    service, _, _, _, optimization_controller = _service()
    provider = object()
    service._data_provider = provider
    graph = MagicMock()
    router = graph.get_router_service.return_value
    graph.run.return_value = SimpleNamespace(
        draft_report="Fresh graph report",
        metadata={"context": "Fresh retrieved evidence"},
    )
    monkeypatch.setattr(
        memory_aware_orchestrator, "AgentGraph", MagicMock(return_value=graph)
    )

    trace, memory_context = service.run_question(
        question="What are AAPL's recent earnings?",
        exchange_symbol="aapl",
    )

    graph.run.assert_called_once_with(
        query="What are AAPL's recent earnings?",
        ticker="AAPL",
    )
    router.route.assert_not_called()
    optimization_controller.run.assert_called_once_with(
        query="What are AAPL's recent earnings?",
        answer="Fresh graph report",
        context="Fresh retrieved evidence",
    )
    assert memory_context.memories == []
    assert trace is optimization_controller.run.return_value


def test_run_question_validates_question_symbol_and_provider():
    """The question-only API validates required input and provider setup."""
    service, *_ = _service()

    with pytest.raises(ValueError, match="question must not be empty"):
        service.run_question("  ", "AAPL")
    with pytest.raises(ValueError, match="exchange_symbol must not be empty"):
        service.run_question("Analyze the company", "  ")
    with pytest.raises(ValueError, match="data_provider is required"):
        service.run_question("Analyze the company", "AAPL")


def test_retrieve_memory_builds_query_using_requested_topics():
    """Memory search receives the question signature and requested topics."""
    service, _, memory_service, memory_builder, _ = _service()

    service.retrieve_memory(
        exchange_symbol="GOOG",
        query="Compare operating margins.",
        topics=["earnings", "profitability"],
    )

    memory_builder.create_query_signature.assert_called_once_with(
        "Compare operating margins."
    )

    query = memory_service.search.call_args.args[0]
    assert query.exchange_symbol == "GOOG"
    assert query.topics == ["earnings", "profitability"]
    assert query.query_signature == "query-signature"
    assert query.query_text == "Compare operating margins."



def test_persists_findings_current_evidence_and_only_unresolved_gaps():
    """The next run can reuse structured findings and gaps from this run."""
    service, router, _, memory_builder, optimization_controller = _service()
    router.route.return_value = "Initial answer"
    trace = optimization_controller.run.return_value
    trace.optimized_answer = (
        "# Revenue\n"
        "- Revenue increased 8% in FY2025.\n"
        "- SEC filing supports the reported increase."
    )
    trace.final_evaluation = [
        SimpleNamespace(
            passed=False,
            issues=[
                SimpleNamespace(
                    description="Add a comparison with the prior fiscal year."
                )
            ],
        )
    ]

    service.run(
        exchange_symbol="AAPL",
        query="Summarize revenue growth.",
        context="Source: SEC filing\nRevenue increased 8% in FY2025.",
    )

    saved = memory_builder.build.call_args.kwargs
    assert saved["key_findings"] == [
        "Revenue increased 8% in FY2025.",
        "SEC filing supports the reported increase.",
    ]
    assert saved["evidence"] == [
        "Source: SEC filing",
        "Revenue increased 8% in FY2025.",
    ]
    assert saved["identified_gaps"] == [
        "Add a comparison with the prior fiscal year."
    ]


def test_prior_memory_evidence_is_not_persisted_as_current_evidence():
    """Historical evidence must not be mislabelled as current-run evidence."""
    memory = _memory(
        "mem-historical",
        research_summary="Prior analysis",
        evidence=["Old annual report"],
    )
    service, _, _, memory_builder, _ = _service(memories=[memory])
    service._data_provider = object()
    graph = MagicMock()
    router = graph.get_router_service.return_value
    router.route.return_value = "Fresh answer"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        memory_aware_orchestrator, "AgentGraph", MagicMock(return_value=graph)
    )
    try:
        service.run_question(
            question="Analyze AAPL.",
            exchange_symbol="AAPL",
        )
    finally:
        monkeypatch.undo()

    assert memory_builder.build.call_args.kwargs["evidence"] == []


def test_run_with_initial_answer_uses_agent_graph_report_without_regeneration():
    """A report generated by AgentGraph is evaluated without a second draft."""
    service, router, _, _, optimization_controller = _service()
    report = "Report generated by AgentGraph"
    trace = optimization_controller.run.return_value

    result_trace, _ = service.run_with_initial_answer(
        exchange_symbol="AAPL",
        query="Analyze financial performance.",
        context="Current evidence context",
        initial_answer=report,
        topics=["financial-performance"],
    )

    router.route.assert_not_called()
    optimization_controller.run.assert_called_once_with(
        query="Analyze financial performance.",
        answer=report,
        context="Current evidence context",
    )
    assert result_trace is trace


def test_run_with_initial_answer_rejects_blank_report():
    """An empty AgentGraph report must not enter the evaluator workflow."""
    service, router, _, _, optimization_controller = _service()

    with pytest.raises(ValueError, match="initial_answer must not be empty"):
        service.run_with_initial_answer(
            exchange_symbol="AAPL",
            query="Analyze financial performance.",
            context="Current evidence context",
            initial_answer="  ",
        )

    router.route.assert_not_called()
    optimization_controller.run.assert_not_called()
