"""
Optimization domain models.

OptimizationAction represents a concrete change that the answer
optimizer should make to address a reflection finding.
"""

from enum import Enum

from pydantic import BaseModel, Field

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.reflection.models import ReflectionPriority


class OptimizationActionType(str, Enum):
    """Supported answer optimization operations."""

    REWRITE = "rewrite"
    ADD_EVIDENCE = "add_evidence"
    REMOVE_UNSUPPORTED_CLAIMS = "remove_unsupported_claims"
    CLARIFY = "clarify"
    UPDATE_TEMPORAL_CONTEXT = "update_temporal_context"
    CORRECT_FACTS = "correct_facts"


class OptimizationAction(BaseModel):
    """One concrete optimization instruction."""

    action_type: OptimizationActionType

    metric: EvaluationMetric

    instruction: str = Field(
        ...,
        min_length=1,
    )

    priority: ReflectionPriority