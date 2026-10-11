"""Suggestion-driven answer optimization through the Cohort 2 router."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator_optimizer.optimization.models import OptimizationAction
if TYPE_CHECKING:
    from routing.router import RouterService
from logging_config import get_logger


logger = get_logger("optimization.answer_optimizer")



class AnswerOptimizer:
    """Revise an answer by applying all evaluator/reflection actions.

    The optimizer delegates generation to the same ``RouterService`` used
    for the initial research answer. No separate model client or duplicate
    generation path is created here.
    """

    def __init__(self, router_service: RouterService) -> None:
        self._router_service = router_service

    def optimize(
        self,
        query: str,
        answer: str,
        context: str,
        actions: list[OptimizationAction],
    ) -> str:
        """Return a revised answer guided by every supplied action.

        Actions are ordered by priority and rendered with their type, metric,
        and full instruction. The model is instructed to use supplied context,
        distinguish historical memory from current evidence, and return only
        the revised answer. If no actions are supplied or the router returns
        an empty response, the original answer is preserved.
        """
        if not actions:
            logger.info("Optimization skipped because no actions were supplied")
            return answer

        prompt = self._build_prompt(
            query=query,
            answer=answer,
            actions=actions,
        )
        logger.info("Generating optimized answer | action_count=%d", len(actions))
        revised_answer = self._router_service.route(
            query=prompt,
            context=context,
        )
        if not revised_answer or not revised_answer.strip():
            logger.warning("Router returned an empty optimized answer; preserving draft")
            return answer
        return revised_answer.strip()

    @staticmethod
    def _build_prompt(
        query: str,
        answer: str,
        actions: list[OptimizationAction],
    ) -> str:
        """Build a complete, explicit prompt from all optimization actions."""
        priority_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }
        ordered_actions = sorted(
            enumerate(actions),
            key=lambda pair: (
                priority_order.get(pair[1].priority.value, 4),
                pair[0],
            ),
        )
        action_lines = []
        for index, (_, action) in enumerate(ordered_actions, start=1):
            action_lines.append(
                f"{index}. Priority: {action.priority.value}; "
                f"Metric: {action.metric.value}; "
                f"Action: {action.action_type.value}; "
                f"Instruction: {action.instruction}"
            )

        return (
            "You are revising an investment research answer after evaluator "
            "and reflection feedback. Apply every action listed below; do not "
            "ignore lower-priority actions unless they conflict with a higher-"
            "priority correction. Preserve accurate supported details, remove "
            "unsupported claims, correct factual errors, clarify reasoning, "
            "and respect reporting periods. Use only the supplied context as "
            "evidence. Treat prior research memory as historical and verify it "
            "against current evidence before presenting it as current. Do not "
            "invent facts, numbers, dates, or citations. If evidence is missing, "
            "state the limitation. Return only the complete revised answer.\n\n"
            f"Original user query:\n{query}\n\n"
            f"Current answer to revise:\n{answer}\n\n"
            "Required optimization actions (apply all):\n"
            + "\n".join(action_lines)
        )
