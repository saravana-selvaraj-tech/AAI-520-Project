"""
Evaluation orchestration service.

EvaluationService coordinates all metric-specific evaluators.

The service deliberately owns orchestration rather than EvaluatorLLM.
EvaluatorLLM is responsible for performing one LLM-as-a-Judge
evaluation, while this service decides which metrics are evaluated
and aggregates their results.
"""

from __future__ import annotations

from evaluator_optimizer.evaluation.accuracy import AccuracyEvaluator
from evaluator_optimizer.evaluation.clarity import ClarityEvaluator
from evaluator_optimizer.evaluation.groundedness import (
    GroundednessEvaluator,
)
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.evaluation.relevance import RelevanceEvaluator
from evaluator_optimizer.evaluation.temporal_validity import (
    TemporalValidityEvaluator,
)


from evaluator_optimizer.logging_config import get_logger


logger = get_logger("evaluation.service")


class EvaluationService:
    """
    Evaluate a generated answer across all configured quality metrics.

    All metric evaluators share the same EvaluatorLLM dependency.
    """

    def __init__(
        self,
        relevance_evaluator: RelevanceEvaluator,
        groundedness_evaluator: GroundednessEvaluator,
        accuracy_evaluator: AccuracyEvaluator,
        temporal_validity_evaluator: TemporalValidityEvaluator,
        clarity_evaluator: ClarityEvaluator,
    ) -> None:
        self._evaluators = [
            relevance_evaluator,
            groundedness_evaluator,
            accuracy_evaluator,
            temporal_validity_evaluator,
            clarity_evaluator,
        ]

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:
        """
        Evaluate the answer using all configured metric evaluators.

        Args:
            query: Original research question.
            answer: Generated answer.
            context: Retrieved evidence.

        Returns:
            Evaluation results, one per configured metric.
        """
        logger.info("Evaluation started | metrics=%d", len(self._evaluators))
        results = [
            evaluator.evaluate(
                query=query,
                answer=answer,
                context=context,
            )
            for evaluator in self._evaluators
        ]
        logger.info("Evaluation completed | metrics=%d", len(results))
        return results