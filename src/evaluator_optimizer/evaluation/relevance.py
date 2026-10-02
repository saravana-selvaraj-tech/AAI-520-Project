"""
Relevance evaluator.

Thin adapter around the generic EvaluatorLLM.
"""

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


class RelevanceEvaluator:
    """Evaluate whether an answer addresses the user's question."""

    metric = EvaluationMetric.RELEVANCE

    def __init__(self, evaluator_llm: EvaluatorLLM) -> None:
        self._evaluator_llm = evaluator_llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """Evaluate answer relevance."""
        return self._evaluator_llm.evaluate(
            query=query,
            answer=answer,
            context=context,
            metric=self.metric,
        )