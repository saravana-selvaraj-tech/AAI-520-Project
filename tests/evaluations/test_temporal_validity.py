from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity
from evaluator_optimizer.evaluation.temporal_validity import (
    TemporalValidityEvaluator,
)


class FakeEvaluatorLLM:
    def __init__(self, result: EvaluationResult) -> None:
        self.result = result
        self.calls = []

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        metric: EvaluationMetric,
    ) -> EvaluationResult:
        self.calls.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
                "metric": metric,
            }
        )

        return self.result


def test_temporal_validity_uses_correct_metric() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.TEMPORAL_VALIDITY,
        score=0.92,
        passed=True,
        rationale=(
            "The answer uses evidence from the requested period."
        ),
    )

    fake_llm = FakeEvaluatorLLM(result)

    evaluator = TemporalValidityEvaluator(fake_llm)

    returned = evaluator.evaluate(
        query="What was Apple's FY2025 revenue?",
        answer="Apple reported revenue in FY2025.",
        context="FY2025 annual report.",
    )

    assert returned.metric == (
        EvaluationMetric.TEMPORAL_VALIDITY
    )
    assert returned.score == 0.92
    assert returned.passed is True

    assert (
        fake_llm.calls[0]["metric"]
        == EvaluationMetric.TEMPORAL_VALIDITY
    )


def test_temporal_validity_preserves_stale_evidence_issue() -> None:
    issue = EvaluationIssue(
        description=(
            "The answer uses FY2024 evidence although the question "
            "requests the latest annual result."
        ),
        severity=IssueSeverity.HIGH,
    )

    result = EvaluationResult(
        metric=EvaluationMetric.TEMPORAL_VALIDITY,
        score=0.25,
        passed=False,
        rationale=(
            "The evidence is stale for the requested period."
        ),
        issues=[issue],
    )

    fake_llm = FakeEvaluatorLLM(result)

    evaluator = TemporalValidityEvaluator(fake_llm)

    returned = evaluator.evaluate(
        query="What is the latest annual revenue?",
        answer="Revenue was $X in FY2024.",
        context="FY2024 annual report.",
    )

    assert returned.score == 0.25
    assert returned.passed is False
    assert len(returned.issues) == 1
    assert returned.issues[0].severity == IssueSeverity.HIGH