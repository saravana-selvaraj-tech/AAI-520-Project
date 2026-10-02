from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.orchestration.evaluation_trace import (
    EvaluationTrace,
)


def test_evaluation_trace_records_evaluation_results() -> None:
    trace = EvaluationTrace(
        query="What was revenue?",
        initial_answer="Revenue was $100B.",
        retrieved_context="Revenue was $100B.",
    )

    result = EvaluationResult(
        metric=EvaluationMetric.ACCURACY,
        score=0.95,
        passed=True,
        rationale="Supported.",
    )

    trace.add_evaluation_results([result])

    assert len(trace.evaluation_results) == 1
    assert (
        trace.evaluation_results[0].metric
        == EvaluationMetric.ACCURACY
    )


def test_evaluation_trace_records_final_evaluation() -> None:
    trace = EvaluationTrace(
        query="Question",
        initial_answer="Initial answer",
        retrieved_context="Evidence",
    )

    result = EvaluationResult(
        metric=EvaluationMetric.CLARITY,
        score=0.90,
        passed=True,
        rationale="Clear.",
    )

    trace.add_final_evaluation([result])

    assert len(trace.final_evaluation) == 1
    assert trace.final_evaluation[0].score == 0.90