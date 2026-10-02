"""
Evaluation domain models.

The Evaluator LLM returns structured evaluation results rather than
free-form text. These models provide a stable contract between the
LLM evaluator and downstream reflection/optimization components.
"""

from typing import List

from pydantic import BaseModel, Field

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.severity import IssueSeverity


class EvaluationIssue(BaseModel):
    """
    A concrete quality issue identified during evaluation.
    """

    description: str = Field(
        ...,
        min_length=1,
        description="Concrete explanation of the identified issue.",
    )

    severity: IssueSeverity = Field(
        ...,
        description="Severity of the identified issue.",
    )


class EvaluationResult(BaseModel):
    """
    Structured result produced by the Evaluator LLM.

    score:
        Normalized LLM-as-Judge score between 0.0 and 1.0.

    passed:
        Indicates whether the response satisfies the evaluation
        threshold for the metric.

    rationale:
        Explanation supporting the score.

    issues:
        Concrete issues that can later be consumed by reflection
        and optimization.
    """

    metric: EvaluationMetric

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    passed: bool

    rationale: str = Field(
        ...,
        min_length=1,
    )

    issues: List[EvaluationIssue] = Field(
        default_factory=list,
    )