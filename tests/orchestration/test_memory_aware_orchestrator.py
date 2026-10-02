"""Tests for the Cohort 2/Cohort 3 memory-aware orchestration boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from evaluator_optimizer.orchestration.memory_aware_orchestrator import (
    MemoryAwareResearchService,
)


def _memory(memory_id: str, gaps: list[str] | None = None):
    return SimpleNamespace(
        memory_id=memory_id,
        identified_gaps=gaps or [],
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

    try:
        service.run(
            exchange_symbol="MSFT",
            query="What is the current market trend?",
            answer="Externally supplied response",
            context="Market RAG context",
        )
    except TypeError:
        return

    raise AssertionError("run() must reject an externally supplied answer")


def test_run_retrieves_memory_before_router_and_persists_final_response():
    """Memory retrieval, generation, optimization, and persistence are ordered."""
    memory = _memory("mem-001", ["Needs more valuation evidence"])
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
    trace.record_memory_retrieved.assert_called_once_with(["mem-123"])
    trace.record_memory_applied.assert_called_once_with(["mem-123"])


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
