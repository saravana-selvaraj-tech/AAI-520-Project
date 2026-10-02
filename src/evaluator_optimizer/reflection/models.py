"""
Reflection domain models.

Reflection converts evaluator findings into a concise diagnosis
that can be consumed by the optimization layer.
"""

from enum import Enum

from pydantic import BaseModel, Field

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.severity import IssueSeverity


class ReflectionPriority(str, Enum):
    """Priority assigned to a reflection finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReflectionFinding(BaseModel):
    """
    One diagnosed quality problem.
    """

    metric: EvaluationMetric

    problem: str = Field(
        ...,
        min_length=1,
    )

    evidence: str = Field(
        ...,
        min_length=1,
    )

    priority: ReflectionPriority

    recommended_change: str = Field(
        ...,
        min_length=1,
    )


class ReflectionResult(BaseModel):
    """
    Structured reflection output.
    """

    summary: str = Field(
        ...,
        min_length=1,
    )

    findings: list[ReflectionFinding] = Field(
        default_factory=list,
    )

    should_optimize: bool = False

    @property
    def highest_priority(
        self,
    ) -> ReflectionPriority | None:
        """Return the highest priority finding."""
        priorities = {
            ReflectionPriority.CRITICAL: 4,
            ReflectionPriority.HIGH: 3,
            ReflectionPriority.MEDIUM: 2,
            ReflectionPriority.LOW: 1,
        }

        if not self.findings:
            return None

        return max(
            (
                finding.priority
                for finding in self.findings
            ),
            key=lambda priority: priorities[priority],
        )