"""
Answer optimization contract.

The controller depends on this abstraction rather than a concrete
LLM, LangChain chain, or research-agent implementation.

This allows the Cohort 3 evaluator-optimizer package to integrate
with the main research agent without tightly coupling the two.
"""

from __future__ import annotations

from typing import Protocol

from evaluator_optimizer.optimization.models import (
    OptimizationAction,
)


class AnswerOptimizer(Protocol):
    """
    Contract implemented by the research-answer generation layer.
    """

    def optimize(
        self,
        query: str,
        answer: str,
        context: str,
        actions: list[OptimizationAction],
    ) -> str:
        """
        Produce a revised answer using optimization actions.
        """
        ...