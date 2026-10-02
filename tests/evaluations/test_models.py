import pytest
from pydantic import ValidationError

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity


def test_evaluation_result_accepts_valid_score() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.ACCURACY,
        score=0.91,
        passed=True,
        rationale="The numerical values match the evidence.",
    )

    assert result.score == 0.91
    assert result.passed is True
    assert result.issues == []


def test_evaluation_result_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            metric=EvaluationMetric.ACCURACY,
            score=1.2,
            passed=True,
            rationale="Invalid score.",
        )


def test_evaluation_result_rejects_negative_score() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            metric=EvaluationMetric.ACCURACY,
            score=-0.1,
            passed=False,
            rationale="Invalid score.",
        )


def test_evaluation_issue_contains_severity() -> None:
    issue = EvaluationIssue(
        description="The answer contains unsupported claims.",
        severity=IssueSeverity.HIGH,
    )

    assert issue.severity == IssueSeverity.HIGH