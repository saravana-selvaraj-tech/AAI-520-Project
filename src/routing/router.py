"""Cohort 2 router compatible with Cohort 3's service boundary."""

from __future__ import annotations

from agent.state import EvidenceRecord, ResearchState
from planning.planner import ResearchTask, TaskType
from tools.providers import ResearchProvider


class RouterService:
    """Select provider capabilities and generate a grounded initial answer."""

    def __init__(self, provider: ResearchProvider) -> None:
        self._provider = provider

    def route(self, query: str, context: str) -> str:
        """Generate the Cohort 3-compatible initial response from context."""
        if not context.strip():
            return f"No retrieved evidence is available to answer: {query}"
        return f"Research request: {query}\n\n{context.strip()}"

    def execute(self, state: ResearchState) -> ResearchState:
        """Run planned retrieval tasks and construct provenance-rich context."""
        for task in state.research_plan:
            try:
                self._execute_task(state, task)
                task.status = "completed"
                state.completed_tasks.append(task.task_id)
            except (RuntimeError, TypeError, ValueError) as exc:
                task.status = "failed"
                state.errors.append(f"{task.task_type}: {exc}")
        state.evidence = self._deduplicate(state.evidence)
        state.retrieved_documents = list(state.evidence)
        state.processed_evidence = [
            {"source": record.citation(), "content": record.content}
            for record in state.evidence
        ]
        state.metadata["context"] = self.build_context(state)
        return state

    def _execute_task(self, state: ResearchState, task: ResearchTask) -> None:
        if task.task_type in (TaskType.FINANCIAL, TaskType.MARKET):
            state.financial_data.update(self._provider.financial(state.ticker, state.user_query))
        elif task.task_type is TaskType.NEWS:
            state.news_data.extend(self._provider.news(state.ticker, state.user_query))
            state.evidence.extend(state.news_data)
        elif task.task_type is TaskType.SEC:
            state.filings.extend(self._provider.filings(state.ticker, state.user_query))
            state.evidence.extend(state.filings)

    @staticmethod
    def build_context(state: ResearchState) -> str:
        sections = []
        if state.financial_data:
            sections.append("Financial data:\n" + "\n".join(f"- {key}: {value}" for key, value in state.financial_data.items()))
        for record in state.evidence:
            sections.append(f"Evidence: {record.citation()}\n{record.content}")
        return "\n\n".join(sections)

    @staticmethod
    def _deduplicate(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
        unique = {}
        for record in records:
            unique[record.document_id or record.url or record.content] = record
        return list(unique.values())