"""
Clarity evaluator.

Evaluates whether a generated investment-research answer is clear,
precise, direct, and logically organized.
"""

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


class ClarityEvaluator:
    """Evaluate answer clarity."""

    metric = EvaluationMetric.CLARITY

    def __init__(self, evaluator_llm: EvaluatorLLM) -> None:
        self._evaluator_llm = evaluator_llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """Evaluate answer clarity."""
        return self._evaluator_llm.evaluate(
            query=query,
            answer=answer,
            context=context,
            metric=self.metric,
        )