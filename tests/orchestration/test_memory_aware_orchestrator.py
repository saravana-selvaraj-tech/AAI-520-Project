"""Tests for the Cohort 2/Cohort 3 memory-aware orchestration boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluator_optimizer.orchestration.memory_aware_orchestrator import (
    MemoryAwareResearchService,
)


def _memory(
    memory_id: str,
    gaps: list[str] | None = None,
    *,
    research_summary: str = "Prior research summary",
    evidence: list[str] | None = None,
):
    """Build a lightweight memory matching fields consumed by orchestration."""
    return SimpleNamespace(
        memory_id=memory_id,
        identified_gaps=gaps or [],
        research_summary=research_summary,
        evidence=evidence or [],
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
