"""Agent graph coordinating planning, specialist routing, and answer generation."""

from __future__ import annotations

from agent.state import ResearchState
from chains.preprocessing import process_evidence
from planning.planner import Planner
from routing.router import RouterService
from tools.providers import ResearchProvider


class AgentGraph:
    """Execute planning, routing, prompt-chain processing, and synthesis."""

    def __init__(
        self,
        provider: ResearchProvider,
        planner: Planner | None = None,
        answer_generator=None,
    ) -> None:
        self._planner = planner or Planner()
        self._router = RouterService(provider, answer_generator)

    def run(self, query: str, ticker: str | None = None) -> ResearchState:
        resolved_ticker = ticker or self._planner.extract_ticker(query)
        state = ResearchState(user_query=query, ticker=resolved_ticker)
        state.research_plan = self._planner.plan(query, resolved_ticker)
        self._router.execute(state)

        state.processed_evidence = process_evidence(state.evidence)
        processed_count = len(state.processed_evidence)
        stage_names = ("preprocess", "classify", "extract", "summarize")
        state.metadata["prompt_chain_trace"] = [
            {
                "stage": stage,
                "input_count": len(state.evidence),
                "output_count": processed_count,
                "status": (
                    "completed" if processed_count else "no_input"
                ),
                "implementation": {
                    "preprocess": "chains.preprocessing.preprocess_text",
                    "classify": "chains.preprocessing.classify_text",
                    "extract": "chains.preprocessing.extract_facts",
                    "summarize": "chains.preprocessing.summarize_text",
                }[stage],
            }
            for stage in stage_names
        ]

        context = state.metadata.get("context") or self._router.build_context(state)
        state.metadata["context"] = context
        state.draft_report = self._router.route(query, context)
        state.metadata["generation_trace"] = {
            "specialist": "Research Synthesis Analyst",
            "route": "RouterService.route",
            "answer_generator": (
                type(self._router._answer_generator).__name__
                if self._router._answer_generator is not None
                else "RouterService evidence-context fallback"
            ),
            "status": "completed" if state.draft_report.strip() else "empty",
            "answer_characters": len(state.draft_report),
        }
        return state

    def get_router_service(self) -> RouterService:
        """Return the configured router shared with downstream workflows."""
        return self._router

    def route(self, query: str, context: str) -> str:
        """Expose the ResearchRouterService protocol."""
        return self.get_router_service().route(query, context)
