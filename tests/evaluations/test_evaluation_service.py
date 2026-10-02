from unittest.mock import Mock

from evaluator_optimizer.evaluation.evaluation_service import (
    EvaluationService,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


def _result(metric: EvaluationMetric) -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        score=0.90,
        passed=True,
        rationale="Passed.",
    )


def test_evaluation_service_runs_all_metric_evaluators() -> None:
    relevance = Mock()
    groundedness = Mock()
    accuracy = Mock()
    temporal = Mock()
    clarity = Mock()

    relevance.evaluate.return_value = _result(
        EvaluationMetric.RELEVANCE
    )
    groundedness.evaluate.return_value = _result(
        EvaluationMetric.GROUNDEDNESS
    )
    accuracy.evaluate.return_value = _result(
        EvaluationMetric.ACCURACY
    )
    temporal.evaluate.return_value = _result(
        EvaluationMetric.TEMPORAL_VALIDITY
    )
    clarity.evaluate.return_value = _result(
        EvaluationMetric.CLARITY
    )

    service = EvaluationService(
        relevance_evaluator=relevance,
        groundedness_evaluator=groundedness,
        accuracy_evaluator=accuracy,
        temporal_validity_evaluator=temporal,
        clarity_evaluator=clarity,
    )

    results = service.evaluate(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
    )

    assert len(results) == 5

    assert {
        result.metric
        for result in results
    } == set(EvaluationMetric)

    relevance.evaluate.assert_called_once()
    groundedness.evaluate.assert_called_once()
    accuracy.evaluate.assert_called_once()
    temporal.evaluate.assert_called_once()
    clarity.evaluate.assert_called_once()