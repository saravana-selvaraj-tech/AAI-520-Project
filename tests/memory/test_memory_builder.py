from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.memory.memory_builder import MemoryBuilder


def test_query_signature_is_stable() -> None:
    builder = MemoryBuilder()

    first = builder.create_query_signature(
        "What was Apple's revenue?"
    )

    second = builder.create_query_signature(
        "  WHAT WAS APPLE'S REVENUE?  "
    )

    assert first == second


def test_builder_creates_reusable_memory() -> None:
    builder = MemoryBuilder()

    evaluation = EvaluationResult(
        metric=EvaluationMetric.ACCURACY,
        score=0.95,
        passed=True,
        rationale="Accurate.",
    )

    memory = builder.build(
        exchange_symbol="NASDAQ:AAPL",
        query="What was Apple's revenue?",
        research_summary="Revenue increased.",
        key_findings=[
            "Revenue increased year over year."
        ],
        claims=[
            "Revenue was approximately $416B."
        ],
        evidence=[
            "FY2025 annual report"
        ],
        identified_gaps=[
            "Segment-level detail unavailable."
        ],
        evaluation_results=[evaluation],
        topics=["revenue", "financials"],
    )

    assert memory.exchange_symbol == "NASDAQ:AAPL"
    assert memory.key_findings
    assert memory.claims
    assert memory.evidence
    assert memory.identified_gaps

    assert memory.evaluation_summary == {
        "accuracy": 0.95,
    }