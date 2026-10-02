"""
Evaluation metric definitions.

These metrics represent the dimensions used by the Evaluator LLM
to judge the quality of an investment-research response.
"""

from enum import Enum


class EvaluationMetric(str, Enum):
    """Supported response-quality evaluation dimensions."""

    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    ACCURACY = "accuracy"
    TEMPORAL_VALIDITY = "temporal_validity"
    CLARITY = "clarity"