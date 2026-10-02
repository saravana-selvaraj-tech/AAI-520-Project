"""
Reflection service.

Transforms EvaluationResult objects into actionable reflection
findings.

Reflection does not rewrite the answer. It diagnoses what should
change and why.
"""

from __future__ import annotations

from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.logging_config import get_logger
from evaluator_optimizer.evaluation.severity import IssueSeverity
from evaluator_optimizer.reflection.models import (
    ReflectionFinding,
    ReflectionPriority,
    ReflectionResult,
)


logger = get_logger("reflection.legacy_service")


class ReflectionService:
    """
    Convert evaluation results into optimization-oriented findings.
    """

    _PRIORITY_MAP = {
        IssueSeverity.NONE: ReflectionPriority.LOW,
        IssueSeverity.LOW: ReflectionPriority.LOW,
        IssueSeverity.MEDIUM: ReflectionPriority.MEDIUM,
        IssueSeverity.HIGH: ReflectionPriority.HIGH,
        IssueSeverity.CRITICAL: ReflectionPriority.CRITICAL,
    }

    def reflect(
        self,
        evaluation_results: list[EvaluationResult],
    ) -> ReflectionResult:
        """
        Analyze evaluation results and identify improvements.

        Passing metrics do not produce optimization findings.
        Failed metrics and their issues become reflection findings.
        """
        logger.info("Reflection started | metrics=%d", len(evaluation_results))
        findings: list[ReflectionFinding] = []

        for result in evaluation_results:
            if result.passed:
                continue

            if result.issues:
                for issue in result.issues:
                    findings.append(
                        ReflectionFinding(
                            metric=result.metric,
                            problem=issue.description,
                            evidence=result.rationale,
                            priority=self._PRIORITY_MAP[
                                issue.severity
                            ],
                            recommended_change=(
                                self._recommended_change(
                                    result.metric.value
                                )
                            ),
                        )
                    )
            else:
                findings.append(
                    ReflectionFinding(
                        metric=result.metric,
                        problem=(
                            f"{result.metric.value} score is "
                            f"{result.score:.2f}, below the "
                            "required threshold."
                        ),
                        evidence=result.rationale,
                        priority=ReflectionPriority.MEDIUM,
                        recommended_change=(
                            self._recommended_change(
                                result.metric.value
                            )
                        ),
                    )
                )

        should_optimize = bool(findings)

        summary = self._build_summary(
            evaluation_results=evaluation_results,
            findings=findings,
        )

        result = ReflectionResult(
            summary=summary,
            findings=findings,
            should_optimize=should_optimize,
        )
        logger.info(
            "Reflection completed | findings=%d | optimize=%s",
            len(result.findings),
            result.should_optimize,
        )
        return result

    @staticmethod
    def _recommended_change(metric: str) -> str:
        """Return an optimization recommendation for a metric."""
        recommendations = {
            "relevance": (
                "Focus the response on the user's requested "
                "question and remove unrelated content."
            ),
            "groundedness": (
                "Remove unsupported claims and ensure important "
                "claims are supported by retrieved evidence."
            ),
            "accuracy": (
                "Recheck factual values, calculations, and "
                "comparisons against the supplied evidence."
            ),
            "temporal_validity": (
                "Use evidence from the requested reporting period "
                "and explicitly distinguish historical information."
            ),
            "clarity": (
                "Improve organization, precision, readability, "
                "and explicit identification of periods and values."
            ),
        }

        return recommendations.get(
            metric,
            "Revise the response to address the identified issue.",
        )

    @staticmethod
    def _build_summary(
        evaluation_results: list[EvaluationResult],
        findings: list[ReflectionFinding],
    ) -> str:
        """Build a concise reflection summary."""
        if not findings:
            return (
                "All configured evaluation metrics passed. "
                "No optimization is required."
            )

        failed_metrics = {
            result.metric.value
            for result in evaluation_results
            if not result.passed
        }

        metrics = ", ".join(sorted(failed_metrics))

        return (
            f"Optimization is required for the following metrics: "
            f"{metrics}. "
            f"{len(findings)} issue(s) were identified."
        )