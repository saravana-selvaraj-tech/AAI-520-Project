"""Bounded orchestration for planning, retrieval, and answer generation."""

from __future__ import annotations

from agent.state import ResearchState
from chains.preprocessing import process_evidence
from planning.planner import Planner
from routing.router import RouterService
from tools.providers import ResearchProvider


class AgentGraph:
    """Execute planning, routing, prompt processing, and generation in order."""

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
        state.draft_report = self._router.route(query, state.metadata["context"])
        return state

    def route(self, query: str, context: str) -> str:
        """Expose the Cohort 3 ResearchRouterService protocol."""
        return self._router.route(query, context)