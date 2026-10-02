from unittest.mock import Mock

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import (
    EvaluationIssue,
    EvaluationResult,
)
from evaluator_optimizer.evaluation.severity import IssueSeverity
from evaluator_optimizer.optimization.optimization_controller import (
    OptimizationController,
)
from evaluator_optimizer.reflection.reflection_service import (
    ReflectionService,
)


def _evaluation_result(
    score: float,
    passed: bool,
) -> EvaluationResult:
    return EvaluationResult(
        metric=EvaluationMetric.GROUNDEDNESS,
        score=score,
        passed=passed,
        rationale="Groundedness evaluation.",
        issues=(
            [
                EvaluationIssue(
                    description="Unsupported claim.",
                    severity=IssueSeverity.HIGH,
                )
            ]
            if not passed
            else []
        ),
    )


def test_controller_stops_when_initial_answer_passes() -> None:
    evaluation_service = Mock()

    evaluation_service.evaluate.return_value = [
        _evaluation_result(
            score=0.95,
            passed=True,
        )
    ]

    optimizer = Mock()

    controller = OptimizationController(
        evaluation_service=evaluation_service,
        reflection_service=ReflectionService(),
        answer_optimizer=optimizer,
        max_iterations=2,
    )

    trace = controller.run(
        query="What was revenue?",
        answer="Revenue was $100B.",
        context="Revenue was $100B.",
    )

    assert trace.iteration_count == 0
    assert trace.optimized_answer == "Revenue was $100B."
    assert len(trace.final_evaluation) == 1

    optimizer.optimize.assert_not_called()


def test_controller_optimizes_failed_answer() -> None:
    evaluation_service = Mock()

    evaluation_service.evaluate.side_effect = [
        [
            _evaluation_result(
                score=0.40,
                passed=False,
            )
        ],
        [
            _evaluation_result(
                score=0.90,
                passed=True,
            )
        ],
    ]

    optimizer = Mock()

    optimizer.optimize.return_value = (
        "Revenue was $100B based on the supplied evidence."
    )

    controller = OptimizationController(
        evaluation_service=evaluation_service,
        reflection_service=ReflectionService(),
        answer_optimizer=optimizer,
        max_iterations=2,
    )

    trace = controller.run(
        query="What was revenue?",
        answer="Revenue was approximately $100B.",
        context="Revenue was $100B.",
    )

    assert trace.iteration_count == 1
    assert trace.optimized_answer == (
        "Revenue was $100B based on the supplied evidence."
    )

    optimizer.optimize.assert_called_once()

    assert len(trace.identified_issues) == 1
    assert len(trace.optimization_actions) == 1


def test_controller_respects_max_iterations() -> None:
    evaluation_service = Mock()

    evaluation_service.evaluate.return_value = [
        _evaluation_result(
            score=0.40,
            passed=False,
        )
    ]

    optimizer = Mock()

    optimizer.optimize.return_value = (
        "Still insufficient answer."
    )

    controller = OptimizationController(
        evaluation_service=evaluation_service,
        reflection_service=ReflectionService(),
        answer_optimizer=optimizer,
        max_iterations=2,
    )

    trace = controller.run(
        query="Question",
        answer="Initial answer",
        context="Evidence",
    )

    assert trace.iteration_count == 2
    assert optimizer.optimize.call_count == 2

    # Initial evaluation + two re-evaluations.
    assert evaluation_service.evaluate.call_count == 3


def test_controller_rejects_invalid_iteration_count() -> None:
    evaluation_service = Mock()
    optimizer = Mock()

    try:
        OptimizationController(
            evaluation_service=evaluation_service,
            reflection_service=ReflectionService(),
            answer_optimizer=optimizer,
            max_iterations=0,
        )
    except ValueError as exc:
        assert "max_iterations" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for max_iterations=0"
        )