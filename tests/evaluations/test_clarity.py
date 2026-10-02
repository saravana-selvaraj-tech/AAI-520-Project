from evaluator_optimizer.evaluation.clarity import ClarityEvaluator
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity


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


def test_clarity_uses_correct_metric() -> None:
    result = EvaluationResult(
        metric=EvaluationMetric.CLARITY,
        score=0.91,
        passed=True,
        rationale="The answer is concise and well structured.",
    )

    fake_llm = FakeEvaluatorLLM(result)

    evaluator = ClarityEvaluator(fake_llm)

    returned = evaluator.evaluate(
        query="Compare revenue growth.",
        answer=(
            "Revenue increased 12% year over year. "
            "The primary driver was international growth."
        ),
        context="Financial results.",
    )

    assert returned.metric == EvaluationMetric.CLARITY
    assert returned.score == 0.91
    assert returned.passed is True

    assert (
        fake_llm.calls[0]["metric"]
        == EvaluationMetric.CLARITY
    )


def test_clarity_preserves_identified_issue() -> None:
    issue = EvaluationIssue(
        description=(
            "The answer combines multiple reporting periods "
            "without clearly identifying each period."
        ),
        severity=IssueSeverity.MEDIUM,
    )

    result = EvaluationResult(
        metric=EvaluationMetric.CLARITY,
        score=0.55,
        passed=False,
        rationale="The response contains period ambiguity.",
        issues=[issue],
    )

    fake_llm = FakeEvaluatorLLM(result)

    evaluator = ClarityEvaluator(fake_llm)

    returned = evaluator.evaluate(
        query="Compare FY2024 and FY2025 revenue.",
        answer="Revenue increased significantly.",
        context=(
            "FY2024 revenue: ...\n"
            "FY2025 revenue: ..."
        ),
    )

    assert returned.passed is False
    assert returned.score == 0.55
    assert returned.issues[0].severity == IssueSeverity.MEDIUM
    assert "reporting periods" in returned.issues[0].description