"""
Evaluation trace.

EvaluationTrace records the evaluator-optimizer lifecycle so that
later components can inspect what was evaluated, what problems were
identified, what optimization actions were taken, and what the final
evaluation produced.

The trace also provides the foundation for Week 2 memory integration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from evaluator_optimizer.evaluation.models import EvaluationResult


class EvaluationTrace(BaseModel):
    """
    Diagnostic trace for one evaluator-optimizer execution.
    """

    query: str

    initial_answer: str

    retrieved_context: str

    evaluation_results: list[EvaluationResult] = Field(
        default_factory=list,
    )

    identified_issues: list[str] = Field(
        default_factory=list,
    )

    reflection: str | None = None

    optimization_actions: list[str] = Field(
        default_factory=list,
    )

    memory_retrieved: list[str] = Field(
        default_factory=list,
    )

    memory_applied: list[str] = Field(
        default_factory=list,
    )

    optimized_answer: str | None = None

    final_evaluation: list[EvaluationResult] = Field(
        default_factory=list,
    )

    iteration_count: int = 0

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def add_evaluation_results(
        self,
        results: list[EvaluationResult],
    ) -> None:
        """Append evaluation results to the trace."""
        self.evaluation_results.extend(results)

    def add_final_evaluation(
        self,
        results: list[EvaluationResult],
    ) -> None:
        """Record the final evaluation results."""
        self.final_evaluation = results

    def record_memory_retrieved(
    self,
    memory_ids: list[str],
    ) -> None:
        """Record memory items retrieved for this run."""
        self.memory_retrieved.extend(
            memory_id
            for memory_id in memory_ids
            if memory_id not in self.memory_retrieved
        )


    def record_memory_applied(
        self,
        memory_ids: list[str],
    ) -> None:
        """Record memory items actually applied to the answer."""
        self.memory_applied.extend(
            memory_id
            for memory_id in memory_ids
            if memory_id not in self.memory_applied
        )