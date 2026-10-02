from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity
from evaluator_optimizer.reflection.models import (
    ReflectionPriority,
)
from evaluator_optimizer.reflection.reflection_service import (
    ReflectionService,
)


def test_reflection_creates_finding_for_failed_metric() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.GROUNDEDNESS,
        score=0.42,
        passed=False,
        rationale="Several claims are unsupported.",
        issues=[
            EvaluationIssue(
                description="Unsupported revenue claim.",
                severity=IssueSeverity.HIGH,
            )
        ],
    )

    service = ReflectionService()

    reflection = service.reflect([result])

    assert reflection.should_optimize is True
    assert len(reflection.findings) == 1

    finding = reflection.findings[0]

    assert finding.metric == EvaluationMetric.GROUNDEDNESS
    assert finding.priority == ReflectionPriority.HIGH
    assert "Unsupported revenue claim" in finding.problem


def test_reflection_does_not_create_findings_for_passing_metric() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.ACCURACY,
        score=0.94,
        passed=True,
        rationale="Facts match evidence.",
    )

    service = ReflectionService()

    reflection = service.reflect([result])

    assert reflection.should_optimize is False
    assert reflection.findings == []


def test_reflection_generates_metric_specific_recommendation() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.TEMPORAL_VALIDITY,
        score=0.35,
        passed=False,
        rationale="Evidence is stale.",
        issues=[
            EvaluationIssue(
                description="FY2024 evidence used.",
                severity=IssueSeverity.HIGH,
            )
        ],
    )

    reflection = ReflectionService().reflect([result])

    assert (
        "reporting period"
        in reflection.findings[0].recommended_change
    )