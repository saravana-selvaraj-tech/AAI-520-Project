"""Tests for suggestion-driven answer optimization through RouterService."""

from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.optimization.models import (
    OptimizationAction,
    OptimizationActionType,
)
from evaluator_optimizer.optimization.optimizer import AnswerOptimizer
from evaluator_optimizer.reflection.models import ReflectionPriority


class RecordingRouter:
    """Capture prompts and return a deterministic optimized answer."""

    def __init__(self, response: str = "Revised, evidence-based answer.") -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def route(self, query: str, context: str) -> str:
        self.calls.append({"query": query, "context": context})
        return self.response


def _action(
    action_type: OptimizationActionType,
    metric: EvaluationMetric,
    instruction: str,
    priority: ReflectionPriority,
) -> OptimizationAction:
    return OptimizationAction(
        action_type=action_type,
        metric=metric,
        instruction=instruction,
        priority=priority,
    )


def test_optimizer_routes_prompt_containing_every_action_and_original_answer() -> None:
    router = RecordingRouter()
    optimizer = AnswerOptimizer(router)
    actions = [
        _action(
            OptimizationActionType.CLARIFY,
            EvaluationMetric.CLARITY,
            "Define the comparison period.",
            ReflectionPriority.LOW,
        ),
        _action(
            OptimizationActionType.REMOVE_UNSUPPORTED_CLAIMS,
            EvaluationMetric.GROUNDEDNESS,
            "Remove claims not supported by evidence.",
            ReflectionPriority.CRITICAL,
        ),
        _action(
            OptimizationActionType.UPDATE_TEMPORAL_CONTEXT,
            EvaluationMetric.TEMPORAL_VALIDITY,
            "Label figures as FY2025.",
            ReflectionPriority.HIGH,
        ),
    ]

    result = optimizer.optimize(
        query="Summarize revenue.",
        answer="Revenue is booming.",
        context="SEC filing: Revenue increased 8% in FY2025.",
        actions=actions,
    )

    assert result == "Revised, evidence-based answer."
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["context"] == "SEC filing: Revenue increased 8% in FY2025."
    prompt = call["query"]
    assert "Summarize revenue." in prompt
    assert "Revenue is booming." in prompt
    assert "Define the comparison period." in prompt
    assert "Remove claims not supported by evidence." in prompt
    assert "Label figures as FY2025." in prompt
    assert prompt.index("Priority: critical") < prompt.index("Priority: high")
    assert prompt.index("Priority: high") < prompt.index("Priority: low")
    assert "Apply every action" in prompt
    assert "Do not invent facts" in prompt


def test_optimizer_returns_original_answer_when_actions_are_empty() -> None:
    router = RecordingRouter()
    optimizer = AnswerOptimizer(router)

    answer = optimizer.optimize(
        query="Query", answer="Original", context="Context", actions=[]
    )

    assert answer == "Original"
    assert router.calls == []


def test_optimizer_preserves_original_answer_when_router_returns_blank() -> None:
    router = RecordingRouter(response="  ")
    optimizer = AnswerOptimizer(router)
    action = _action(
        OptimizationActionType.CLARIFY,
        EvaluationMetric.CLARITY,
        "Clarify the answer.",
        ReflectionPriority.MEDIUM,
    )

    answer = optimizer.optimize(
        query="Query", answer="Original", context="Context", actions=[action]
    )

    assert answer == "Original"
    assert len(router.calls) == 1


def test_optimizer_reuses_router_service_exposed_by_agent_graph() -> None:
    """Initial and optimized responses can share the graph's configured router."""
    from agent.graph import AgentGraph
    from tools.providers import InMemoryResearchProvider

    class PromptAwareGenerator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate(self, query: str, context: str) -> str:
            self.calls.append((query, context))
            if "Required optimization actions" in query:
                return "Optimized answer with source and period."
            return "Initial answer."

    generator = PromptAwareGenerator()
    graph = AgentGraph(
        InMemoryResearchProvider(),
        answer_generator=generator,
    )
    router_service = graph.get_router_service()
    optimizer = AnswerOptimizer(router_service)
    action = _action(
        OptimizationActionType.CLARIFY,
        EvaluationMetric.CLARITY,
        "Include the source and reporting period.",
        ReflectionPriority.HIGH,
    )

    initial = router_service.route("Summarize revenue.", "SEC evidence")
    revised = optimizer.optimize(
        query="Summarize revenue.",
        answer=initial,
        context="SEC evidence",
        actions=[action],
    )

    assert initial == "Initial answer."
    assert revised == "Optimized answer with source and period."
    assert len(generator.calls) == 2
    assert generator.calls[0] == ("Summarize revenue.", "SEC evidence")
    assert "Include the source and reporting period." in generator.calls[1][0]
    assert generator.calls[1][1] == "SEC evidence"
