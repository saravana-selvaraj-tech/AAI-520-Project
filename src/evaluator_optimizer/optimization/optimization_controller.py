"""
Optimization controller.

Coordinates the bounded Evaluator -> Reflection -> Optimization
loop.

The controller prevents uncontrolled recursive optimization by
enforcing a maximum number of iterations.
"""

from __future__ import annotations

import uuid

from evaluator_optimizer.config import DEFAULT_MAX_OPTIMIZATION_ITERATIONS
from evaluator_optimizer.evaluation.evaluation_service import (
    EvaluationService,
)
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.optimization.models import (
    OptimizationAction,
    OptimizationActionType,
)
from evaluator_optimizer.optimization.optimizer import AnswerOptimizer
from evaluator_optimizer.orchestration.evaluation_trace import (
    EvaluationTrace,
)
from evaluator_optimizer.reflection.models import ReflectionResult
from evaluator_optimizer.reflection.reflection_service import (
    ReflectionService,
)
from evaluator_optimizer.logging_config import get_logger


logger = get_logger("optimization.controller")


class OptimizationController:
    """
    Run a bounded evaluator-optimizer loop.

    Flow:

        evaluate
           |
        passed?
        /    \
      yes     no
       |       |
      end   reflect
               |
            actions
               |
            optimize
               |
            re-evaluate

    Args:
        evaluation_service:
            Evaluates generated answers.

        reflection_service:
            Diagnoses evaluation failures.

        answer_optimizer:
            Applies optimization actions.

        max_iterations:
            Maximum number of optimization cycles.
    """

    def __init__(
        self,
        evaluation_service: EvaluationService,
        reflection_service: ReflectionService,
        answer_optimizer: AnswerOptimizer,
        max_iterations: int = DEFAULT_MAX_OPTIMIZATION_ITERATIONS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1"
            )

        self._evaluation_service = evaluation_service
        self._reflection_service = reflection_service
        self._answer_optimizer = answer_optimizer
        self._max_iterations = max_iterations

    def run(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> EvaluationTrace:
        """
        Run the evaluator-optimizer loop.

        Returns:
            EvaluationTrace containing the complete lifecycle.
        """
        run_id = uuid.uuid4().hex[:12]
        logger.info("Workflow started | run_id=%s", run_id)
        trace = EvaluationTrace(
            query=query,
            initial_answer=answer,
            retrieved_context=context,
            metadata={"run_id": run_id},
        )

        current_answer = answer

        logger.info("Evaluation started | run_id=%s | iteration=0", run_id)
        initial_results = self._evaluation_service.evaluate(
            query=query,
            answer=current_answer,
            context=context,
        )

        trace.add_evaluation_results(initial_results)
        logger.info(
            "Evaluation completed | run_id=%s | iteration=0 | passed=%s",
            run_id,
            self._all_passed(initial_results),
        )

        if self._all_passed(initial_results):
            trace.final_evaluation = initial_results
            trace.optimized_answer = current_answer
            logger.info(
                "Workflow completed | run_id=%s | iterations=0 | passed=True",
                run_id,
            )
            return trace

        for iteration in range(1, self._max_iterations + 1):
            trace.iteration_count = iteration

            logger.info(
                "Reflection started | run_id=%s | iteration=%d",
                run_id,
                iteration,
            )
            reflection = self._reflection_service.reflect(
                initial_results
            )

            self._record_reflection(
                trace=trace,
                reflection=reflection,
            )
            logger.info(
                "Reflection completed | run_id=%s | iteration=%d | findings=%d",
                run_id, iteration, len(reflection.findings),
            )

            if not reflection.should_optimize:
                break

            actions = self._create_actions(reflection)

            trace.optimization_actions.extend(
                action.instruction
                for action in actions
            )

            logger.info(
                "Optimization started | run_id=%s | iteration=%d",
                run_id,
                iteration,
            )
            current_answer = self._answer_optimizer.optimize(
                query=query,
                answer=current_answer,
                context=context,
                actions=actions,
            )

            trace.optimized_answer = current_answer
            logger.info(
                "Optimization completed | run_id=%s | iteration=%d",
                run_id,
                iteration,
            )

            logger.info(
                "Evaluation started | run_id=%s | iteration=%d",
                run_id,
                iteration,
            )
            results = self._evaluation_service.evaluate(
                query=query,
                answer=current_answer,
                context=context,
            )

            trace.final_evaluation = results
            logger.info(
                "Evaluation completed | run_id=%s | iteration=%d | passed=%s",
                run_id, iteration, self._all_passed(results),
            )

            if self._all_passed(results):
                break

            initial_results = results

        logger.info(
            "Workflow completed | run_id=%s | iterations=%d | passed=%s",
            run_id, trace.iteration_count, self._all_passed(trace.final_evaluation),
        )
        return trace

    @staticmethod
    def _all_passed(
        results: list[EvaluationResult],
    ) -> bool:
        """Return True when every evaluation metric passes."""
        return bool(results) and all(
            result.passed
            for result in results
        )

    @staticmethod
    def _record_reflection(
        trace: EvaluationTrace,
        reflection: ReflectionResult,
    ) -> None:
        """Add reflection information to the trace."""
        trace.reflection = reflection.summary

        trace.identified_issues.extend(
            finding.problem
            for finding in reflection.findings
        )

    @staticmethod
    def _create_actions(
        reflection: ReflectionResult,
    ) -> list[OptimizationAction]:
        """Convert reflection findings into optimization actions."""
        actions: list[OptimizationAction] = []

        action_type_map = {
            "relevance": OptimizationActionType.REWRITE,
            "groundedness": (
                OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS
            ),
            "accuracy": OptimizationActionType.CORRECT_FACTS,
            "temporal_validity": (
                OptimizationActionType.UPDATE_TEMPORAL_CONTEXT
            ),
            "clarity": OptimizationActionType.CLARIFY,
        }

        for finding in reflection.findings:
            action_type = action_type_map.get(
                finding.metric.value,
                OptimizationActionType.REWRITE,
            )

            actions.append(
                OptimizationAction(
                    action_type=action_type,
                    metric=finding.metric,
                    instruction=finding.recommended_change,
                    priority=finding.priority,
                )
            )

        return actions