"""
Issue severity calculation.

Severity is derived from the LLM evaluation score.

The score itself remains the primary quantitative metric.
Severity provides a categorical interpretation that can be consumed
by reflection and optimization components in later weeks.
"""

from enum import Enum

from evaluator_optimizer.config import (
    MAX_SCORE,
    MIN_SCORE,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_LOW_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
    SEVERITY_NONE_THRESHOLD,
)


class IssueSeverity(str, Enum):
    """Severity associated with an evaluation result."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def determine_severity(score: float) -> IssueSeverity:
    """
    Convert an evaluation score into an issue severity.

    Thresholds:
        >= 0.80 -> NONE
        >= 0.65 -> LOW
        >= 0.50 -> MEDIUM
        >= 0.30 -> HIGH
        <  0.30 -> CRITICAL

    Args:
        score: Normalized evaluation score in [0.0, 1.0].

    Returns:
        Corresponding IssueSeverity.

    Raises:
        ValueError: If score is outside [0.0, 1.0].
    """
    if not MIN_SCORE <= score <= MAX_SCORE:
        raise ValueError("score must be between MIN_SCORE and MAX_SCORE")

    if score >= SEVERITY_NONE_THRESHOLD:
        return IssueSeverity.NONE

    if score >= SEVERITY_LOW_THRESHOLD:
        return IssueSeverity.LOW

    if score >= SEVERITY_MEDIUM_THRESHOLD:
        return IssueSeverity.MEDIUM

    if score >= SEVERITY_HIGH_THRESHOLD:
        return IssueSeverity.HIGH

    return IssueSeverity.CRITICAL