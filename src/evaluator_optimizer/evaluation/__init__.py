"""
Evaluation package.

Provides the LLM-as-a-Judge evaluator, evaluation metrics,
structured evaluation results, severity mapping, and
metric-specific evaluator adapters.
"""

from evaluator_optimizer.evaluation.accuracy import AccuracyEvaluator
from evaluator_optimizer.evaluation.clarity import ClarityEvaluator
from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.groundedness import (
    GroundednessEvaluator,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.relevance import RelevanceEvaluator
from evaluator_optimizer.evaluation.severity import (
    IssueSeverity,
    determine_severity,
)
from evaluator_optimizer.evaluation.temporal_validity import (
    TemporalValidityEvaluator,
)

__all__ = [
    "AccuracyEvaluator",
    "ClarityEvaluator",
    "EvaluatorLLM",
    "EvaluationIssue",
    "EvaluationMetric",
    "EvaluationResult",
    "GroundednessEvaluator",
    "IssueSeverity",
    "RelevanceEvaluator",
    "TemporalValidityEvaluator",
    "determine_severity",
]