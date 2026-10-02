import pytest

from evaluator_optimizer.evaluation.accuracy import AccuracyEvaluator
from evaluator_optimizer.evaluation.clarity import ClarityEvaluator
from evaluator_optimizer.evaluation.groundedness import (
    GroundednessEvaluator,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.evaluation.relevance import RelevanceEvaluator
from evaluator_optimizer.evaluation.temporal_validity import (
    TemporalValidityEvaluator,
)


class FakeEvaluatorLLM:
    """Test double for EvaluatorLLM."""

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


@pytest.mark.parametrize(
    ("evaluator_class", "metric"),
    [
        (RelevanceEvaluator, EvaluationMetric.RELEVANCE),
        (
            GroundednessEvaluator,
            EvaluationMetric.GROUNDEDNESS,
        ),
        (AccuracyEvaluator, EvaluationMetric.ACCURACY),
        (
            TemporalValidityEvaluator,
            EvaluationMetric.TEMPORAL_VALIDITY,
        ),
        (ClarityEvaluator, EvaluationMetric.CLARITY),
    ],
)
def test_metric_evaluator_delegates_to_evaluator_llm(
    evaluator_class,
    metric,
) -> None:
    result = EvaluationResult(
        metric=metric,
        score=0.91,
        passed=True,
        rationale="Good evaluation.",
    )

    fake_llm = FakeEvaluatorLLM(result)
    evaluator = evaluator_class(fake_llm)

    returned = evaluator.evaluate(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
    )

    assert returned.metric == metric
    assert returned.score == 0.91
    assert returned.passed is True

    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["metric"] == metric