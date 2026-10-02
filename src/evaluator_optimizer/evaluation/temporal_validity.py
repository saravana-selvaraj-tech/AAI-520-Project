"""
Temporal validity evaluator.

Evaluates whether the evidence and claims used in an answer are
appropriate for the requested reporting period and time context.

Temporal validity is not the same as freshness.

For example:

    "What was Apple's FY2020 revenue?"
        -> FY2020 evidence is appropriate.

    "What is Apple's latest annual revenue?"
        -> FY2020 evidence is potentially stale and inappropriate
           when newer annual evidence is required.
"""

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult


class TemporalValidityEvaluator:
    """Evaluate temporal appropriateness of research evidence."""

    metric = EvaluationMetric.TEMPORAL_VALIDITY

    def __init__(self, evaluator_llm: EvaluatorLLM) -> None:
        self._evaluator_llm = evaluator_llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """Evaluate temporal validity."""
        return self._evaluator_llm.evaluate(
            query=query,
            answer=answer,
            context=context,
            metric=self.metric,
        )