"""
Accuracy evaluator.

Evaluates factual and numerical consistency between the generated
answer and supplied evidence.
"""

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


class AccuracyEvaluator:
    """Evaluate factual and numerical accuracy."""

    metric = EvaluationMetric.ACCURACY

    def __init__(self, evaluator_llm: EvaluatorLLM) -> None:
        self._evaluator_llm = evaluator_llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """Evaluate answer accuracy."""
        return self._evaluator_llm.evaluate(
            query=query,
            answer=answer,
            context=context,
            metric=self.metric,
        )