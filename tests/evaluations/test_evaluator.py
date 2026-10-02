from unittest.mock import Mock

import pytest

from evaluator_optimizer.evaluation.evaluator import (
    EvaluatorLLM,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity


def make_result(
    metric: EvaluationMetric,
    score: float,
) -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        score=score,
        passed=False,
        rationale="Evaluation rationale.",
        issues=[
            EvaluationIssue(
                description="Example issue.",
                severity=IssueSeverity.MEDIUM,
            )
        ],
    )


def test_evaluator_calls_structured_response_api() -> None:
    client = Mock()

    client.responses.parse.return_value.output_parsed = (
        make_result(
            EvaluationMetric.ACCURACY,
            0.91,
        )
    )

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    result = evaluator.evaluate(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
        metric=EvaluationMetric.ACCURACY,
    )

    assert result.metric == EvaluationMetric.ACCURACY
    assert result.score == 0.91
    assert result.passed is True

    client.responses.parse.assert_called_once()


def test_evaluator_forces_requested_metric() -> None:
    client = Mock()

    # Simulate an LLM accidentally returning the wrong metric.
    client.responses.parse.return_value.output_parsed = (
        make_result(
            EvaluationMetric.CLARITY,
            0.90,
        )
    )

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    result = evaluator.evaluate(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
        metric=EvaluationMetric.ACCURACY,
    )

    assert result.metric == EvaluationMetric.ACCURACY


def test_evaluator_recalculates_pass_status() -> None:
    client = Mock()

    client.responses.parse.return_value.output_parsed = (
        make_result(
            EvaluationMetric.GROUNDEDNESS,
            0.55,
        )
    )

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
        pass_threshold=0.70,
    )

    result = evaluator.evaluate(
        query="What happened?",
        answer="Something happened.",
        context="Evidence.",
        metric=EvaluationMetric.GROUNDEDNESS,
    )

    assert result.score == 0.55
    assert result.passed is False


def test_evaluator_rejects_empty_query() -> None:
    client = Mock()

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            query="",
            answer="Answer",
            context="Context",
            metric=EvaluationMetric.RELEVANCE,
        )


def test_evaluator_rejects_empty_answer() -> None:
    client = Mock()

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            query="Question",
            answer="",
            context="Context",
            metric=EvaluationMetric.RELEVANCE,
        )


def test_evaluator_rejects_empty_context() -> None:
    client = Mock()

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            query="Question",
            answer="Answer",
            context="",
            metric=EvaluationMetric.RELEVANCE,
        )


def test_evaluate_all_evaluates_every_metric() -> None:
    client = Mock()

    def parse_response(*args, **kwargs):
        return Mock(
            output_parsed=EvaluationResult(
                metric=EvaluationMetric.RELEVANCE,
                score=0.90,
                passed=True,
                rationale="Good.",
            )
        )

    client.responses.parse.side_effect = parse_response

    evaluator = EvaluatorLLM(
        model="test-model",
        client=client,
    )

    results = evaluator.evaluate_all(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
    )

    assert len(results) == len(EvaluationMetric)

    assert client.responses.parse.call_count == len(
        EvaluationMetric
    )