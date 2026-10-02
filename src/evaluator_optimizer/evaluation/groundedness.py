"""
Groundedness evaluator.

Determines whether generated claims are supported by retrieved
evidence.
"""

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


class GroundednessEvaluator:
    """Evaluate whether the answer is supported by retrieved context."""

    metric = EvaluationMetric.GROUNDEDNESS

    def __init__(self, evaluator_llm: EvaluatorLLM) -> None:
        self._evaluator_llm = evaluator_llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """Evaluate answer groundedness."""
        return self._evaluator_llm.evaluate(
            query=query,
            answer=answer,
            context=context,
            metric=self.metric,
        )