"""Integration tests for the production evaluator-optimizer workflow.

These tests use the real EvaluationService, ReflectionService,
OptimizationController, MemoryAwareResearchService, MemoryService,
MemoryCache, and PickleMemoryRepository. Only the LLM judge, Cohort 2
router, and answer-generation optimizer are deterministic fakes.

Run from the extracted source root:
    PYTHONPATH=. pytest -q tests/integration/test_evaluator_optimizer_workflow.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluator_optimizer.evaluation.accuracy import AccuracyEvaluator
from evaluator_optimizer.evaluation.clarity import ClarityEvaluator
from evaluator_optimizer.evaluation.evaluation_service import EvaluationService
from evaluator_optimizer.evaluation.groundedness import GroundednessEvaluator
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationIssue, EvaluationResult
from evaluator_optimizer.evaluation.relevance import RelevanceEvaluator
from evaluator_optimizer.evaluation.temporal_validity import (
    TemporalValidityEvaluator,
)
from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.memory_builder import MemoryBuilder
from evaluator_optimizer.memory.memory_service import MemoryService
from evaluator_optimizer.memory.repository import PickleMemoryRepository
from evaluator_optimizer.optimization.optimization_controller import (
    OptimizationController,
)
from evaluator_optimizer.orchestration.memory_aware_orchestrator import (
    MemoryAwareResearchService,
)
from evaluator_optimizer.reflection.reflection_service import ReflectionService
from evaluator_optimizer.optimization.models import OptimizationActionType
from evaluator_optimizer.evaluation.severity import IssueSeverity


class ScriptedEvaluatorLLM:
    """Deterministic substitute for the LLM judge; preserves real adapters."""

    REQUIRED_TERMS = ("revenue", "fy2025", "sec filing")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        metric: EvaluationMetric,
    ) -> EvaluationResult:
        self.calls.append(
            {"query": query, "answer": answer, "context": context, "metric": metric}
        )
        answer_lower = answer.casefold()
        context_lower = context.casefold()
        grounded = "revenue increased 8%" in answer_lower and (
            "revenue increased 8%" in context_lower
        )
        complete = all(term in answer_lower for term in self.REQUIRED_TERMS)
        passed = grounded and complete
        issues = []
        if not complete:
            issues.append(
                EvaluationIssue(
                    description=(
                        "Answer must identify the reporting period and "
                        "cite the primary source."
                    ),
                    severity=IssueSeverity.HIGH,
                )
            )
        if not grounded:
            issues.append(
                EvaluationIssue(
                    description="Revenue claim is not supported by supplied evidence.",
                    severity=IssueSeverity.CRITICAL,
                )
            )
        return EvaluationResult(
            metric=metric,
            score=0.95 if passed else 0.35,
            passed=passed,
            rationale=(
                "Required period, source, and evidence are present."
                if passed
                else "The response omits required evidence details."
            ),
            issues=issues,
        )


class ScriptedAnswerOptimizer:
    """Applies deterministic revisions based on real OptimizationAction objects."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def optimize(
        self,
        query: str,
        answer: str,
        context: str,
        actions: list[Any],
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
                "actions": actions,
            }
        )
        revised = answer
        context_lower = context.casefold()
        has_revenue_claim = "revenue increased 8%" in revised.casefold()
        if "revenue increased 8%" in context_lower and not has_revenue_claim:
            revised += " Revenue increased 8%."
        if "fy2025" in context_lower and "fy2025" not in revised.casefold():
            revised += " Period: FY2025."
        if "sec filing" in context_lower and "sec filing" not in revised.casefold():
            revised += " Source: SEC filing."
        return revised


class ScriptedRouter:
    """Provides repeatable Cohort 2 answers without network calls."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.calls: list[dict[str, str]] = []

    def route(self, query: str, context: str) -> str:
        self.calls.append({"query": query, "context": context})
        return next(self.answers)


class SuggestionDrivenAnswerOptimizer:
    """Simulate an LLM optimizer that follows controller suggestions.

    The fake deliberately discards unsupported initial wording and only adds
    claims found in the supplied context. Its actions determine which repairs
    it attempts, making the optimizer -> evaluator loop deterministic.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def optimize(
        self,
        query: str,
        answer: str,
        context: str,
        actions: list[Any],
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
                "actions": actions,
            }
        )
        action_types = {action.action_type for action in actions}
        context_lower = context.casefold()
        repaired_claims: list[str] = []

        if (
            OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS in action_types
            and "revenue increased 8%" in context_lower
        ):
            repaired_claims.append("Revenue increased 8%.")
        if (
            OptimizationActionType.CORRECT_FACTS in action_types
            and "revenue increased 8%" in context_lower
            and "Revenue increased 8%." not in repaired_claims
        ):
            repaired_claims.append("Revenue increased 8%.")
        if (
            OptimizationActionType.UPDATE_TEMPORAL_CONTEXT in action_types
            and "fy2025" in context_lower
        ):
            repaired_claims.append("Reporting period: FY2025.")
        if (
            OptimizationActionType.CLARIFY in action_types
            and "sec filing" in context_lower
        ):
            repaired_claims.append("Source: SEC filing.")

        # Do not repeat the unsupported draft when a rewrite/removal was
        # requested. Retain it only when no corrective action was suggested.
        corrective_types = {
            OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS,
            OptimizationActionType.CORRECT_FACTS,
            OptimizationActionType.UPDATE_TEMPORAL_CONTEXT,
            OptimizationActionType.CLARIFY,
        }
        if not action_types.intersection(corrective_types):
            return answer
        return " ".join(repaired_claims) or answer


def build_workflow(
    tmp_path: Path,
    answers: list[str],
    *,
    max_iterations: int = 2,
    answer_optimizer: Any | None = None,
):
    judge = ScriptedEvaluatorLLM()
    optimizer = answer_optimizer or ScriptedAnswerOptimizer()

    evaluation_service = EvaluationService(
        relevance_evaluator=RelevanceEvaluator(judge),
        groundedness_evaluator=GroundednessEvaluator(judge),
        accuracy_evaluator=AccuracyEvaluator(judge),
        temporal_validity_evaluator=TemporalValidityEvaluator(judge),
        clarity_evaluator=ClarityEvaluator(judge),
    )
    controller = OptimizationController(
        evaluation_service=evaluation_service,
        reflection_service=ReflectionService(),
        answer_optimizer=optimizer,
        max_iterations=max_iterations,
    )

    repository = PickleMemoryRepository(tmp_path / "memory.pkl")
    memory_service = MemoryService(
        repository=repository,
        cache=MemoryCache(max_size=20),
    )
    memory_builder = MemoryBuilder()
    router = ScriptedRouter(answers)
    service = MemoryAwareResearchService(
        router_service=router,
        memory_service=memory_service,
        memory_builder=memory_builder,
        optimization_controller=controller,
    )
    return service, router, judge, optimizer, memory_service, repository


RESEARCH_CONTEXT = (
    "FY2025 annual report. SEC filing. "
    "Revenue increased 8% year over year."
)
QUERY = "Summarize revenue growth for FY2025."
TOPICS = ["earnings", "revenue"]


def test_end_to_end_failed_evaluation_reflection_optimization_and_reevaluation(
    tmp_path: Path,
) -> None:
    """Low-scoring answer is diagnosed, revised, and passes a second evaluation."""
    service, router, judge, optimizer, memory_service, repository = build_workflow(
        tmp_path,
        ["Revenue grew, but the period and source are not stated."],
    )

    trace, memory_context = service.run(
        exchange_symbol="AAPL",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )

    assert len(router.calls) == 1
    assert trace.initial_answer == (
        "Revenue grew, but the period and source are not stated."
    )
    assert len(trace.evaluation_results) == 5
    assert any(not result.passed for result in trace.evaluation_results)
    assert trace.reflection
    assert trace.identified_issues
    assert trace.optimization_actions
    assert optimizer.calls
    assert any(
        action.action_type == OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS
        for action in optimizer.calls[0]["actions"]
    )
    assert trace.optimized_answer is not None
    assert "Revenue increased 8%" in trace.optimized_answer
    assert "FY2025" in trace.optimized_answer
    assert "SEC filing" in trace.optimized_answer
    assert len(trace.final_evaluation) == 5
    assert all(result.passed for result in trace.final_evaluation)
    assert trace.iteration_count == 1
    assert len(judge.calls) == 10  # five initial metrics + five re-evaluations
    assert memory_context.memories == []
    assert len(repository.load_all()) == 1
    assert repository.load_all()[0].research_summary == trace.optimized_answer
    assert trace.memory_retrieved == []
    assert trace.memory_applied == []


def test_passing_answer_skips_reflection_and_optimization(tmp_path: Path) -> None:
    """An answer that passes all five real metric adapters is not rewritten."""
    answer = "Revenue increased 8% in FY2025. Source: SEC filing."
    service, _, judge, optimizer, _, repository = build_workflow(
        tmp_path, [answer]
    )

    trace, _ = service.run(
        exchange_symbol="MSFT",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )

    assert trace.initial_answer == answer
    assert trace.optimized_answer == answer
    assert trace.iteration_count == 0
    assert trace.reflection is None
    assert trace.optimization_actions == []
    assert optimizer.calls == []
    assert len(judge.calls) == 5
    assert all(result.passed for result in trace.final_evaluation)
    assert len(repository.load_all()) == 1


def test_iteration_limit_stops_after_configured_re_evaluations(
    tmp_path: Path,
) -> None:
    """A response that cannot be improved must stop at the iteration bound."""
    service, _, judge, optimizer, _, _ = build_workflow(
        tmp_path,
        ["Unsupported claim with no relevant terms."],
        max_iterations=2,
    )
    # Context intentionally lacks the source phrases the optimizer can add.
    trace, _ = service.run(
        exchange_symbol="NVDA",
        query="Assess the company's revenue.",
        context="No relevant revenue period or source is available.",
        topics=["revenue"],
    )

    assert trace.iteration_count == 2
    assert len(optimizer.calls) == 2
    assert len(judge.calls) == 15  # initial + two rounds, five metrics each
    assert len(trace.final_evaluation) == 5
    assert not all(result.passed for result in trace.final_evaluation)


def test_memory_is_persisted_and_retrieved_on_matching_follow_up_run(
    tmp_path: Path,
) -> None:
    """A saved memory can be retrieved and traced by a subsequent matching run."""
    service, _, _, _, memory_service, repository = build_workflow(
        tmp_path,
        [
            "Revenue increased 8% in FY2025. Source: SEC filing.",
            "Revenue increased 8% in FY2025. Source: SEC filing.",
        ],
    )

    first_trace, _ = service.run(
        exchange_symbol="GOOGL",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )
    first_memory = repository.load_all()[0]
    assert first_memory.research_summary == first_trace.optimized_answer

    second_trace, second_context = service.run(
        exchange_symbol="GOOGL",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )

    assert any(
        memory.memory_id == first_memory.memory_id
        for memory in second_context.memories
    )
    assert first_memory.memory_id in second_trace.memory_retrieved
    assert first_memory.memory_id in second_trace.memory_applied
    assert len(repository.load_all()) == 2
    assert len(memory_service._repository.load_all()) == 2


def test_prior_memory_gaps_are_propagated_to_new_memory(tmp_path: Path) -> None:
    """Identified gaps from retrieved memory survive into the next memory item."""
    service, _, _, _, memory_service, repository = build_workflow(
        tmp_path,
        [
            "Revenue increased 8% in FY2025. Source: SEC filing.",
            "Revenue increased 8% in FY2025. Source: SEC filing.",
        ],
    )

    service.run(
        exchange_symbol="TSLA",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )
    memories = repository.load_all()
    memories[0].identified_gaps = ["Compare with prior-year operating margin."]
    # Persist the seeded gap into the same test repository.
    memory_service.save(memories[0])

    _, memory_context = service.run(
        exchange_symbol="TSLA",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )
    latest_memory = repository.load_all()[-1]

    assert "Compare with prior-year operating margin." in memory_context.research_gaps
    assert "Compare with prior-year operating margin." in latest_memory.identified_gaps



def test_suggestion_driven_optimizer_uses_retrieved_memory_and_passes_reevaluation(
    tmp_path: Path,
) -> None:
    """Prior optimized findings inform a later repair and pass re-evaluation."""
    optimizer = SuggestionDrivenAnswerOptimizer()
    service, router, judge, _, _, repository = build_workflow(
        tmp_path,
        [
            "The company had revenue growth.",
            "Revenue grew, but the period and source are not stated.",
        ],
        answer_optimizer=optimizer,
    )

    # First run creates a successful, reusable memory from current context.
    first_trace, _ = service.run(
        exchange_symbol="AAPL",
        query=QUERY,
        context=RESEARCH_CONTEXT,
        topics=TOPICS,
    )
    saved_memory = repository.load_all()[0]
    assert first_trace.optimized_answer == saved_memory.research_summary
    assert all(result.passed for result in first_trace.final_evaluation)

    # The follow-up's current context omits the key details. Retrieved memory
    # is added by the orchestrator and is visible to the router, optimizer,
    # and evaluator. The optimizer responds to actual action suggestions.
    sparse_context = "Prepare a concise follow-up summary of revenue performance."
    second_trace, memory_context = service.run(
        exchange_symbol="AAPL",
        query=QUERY,
        context=sparse_context,
        topics=TOPICS,
    )

    assert saved_memory.memory_id in {
        item.memory_id for item in memory_context.memories
    }
    assert "Prior research memory" in router.calls[1]["context"]
    assert "Revenue increased 8%" in optimizer.calls[-1]["context"]
    assert optimizer.calls[-1]["actions"]
    assert any(
        action.action_type == OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS
        for action in optimizer.calls[-1]["actions"]
    )
    assert "Revenue increased 8%" in second_trace.optimized_answer
    assert "FY2025" in second_trace.optimized_answer
    assert "SEC filing" in second_trace.optimized_answer
    assert all(result.passed for result in second_trace.final_evaluation)
    assert saved_memory.memory_id in second_trace.memory_retrieved
    assert saved_memory.memory_id in second_trace.memory_applied
    assert len(judge.calls) == 20
    assert len(repository.load_all()) == 2
