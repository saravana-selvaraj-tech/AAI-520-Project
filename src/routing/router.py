"""Research routing with explicit specialist and provider execution traces."""

from __future__ import annotations

import os
import re
from typing import Any

from agent.state import EvidenceRecord, ResearchState
from planning.planner import ResearchTask, TaskType
from tools.providers import ResearchProvider


class RouterService:
    """Dispatch planned research tasks to specialist/provider capabilities."""

    def __init__(self, provider: ResearchProvider, answer_generator=None) -> None:
        self._provider = provider
        self._answer_generator = answer_generator

    def route(self, query: str, context: str, system_context: str = None) -> str:
        """Generate an answer from the supplied evidence context."""
        if not context.strip():
            return f"No retrieved evidence is available to answer: {query}"
        if self._answer_generator is not None:
            return self._answer_generator.generate(query, context, system_context)
        return (
            "Based on the retrieved evidence, the available findings are:\n\n"
            + context.strip()
        )

    def execute(self, state: ResearchState) -> ResearchState:
        """Run planned tasks and record the selected route and actual results."""
        routing_trace: list[dict[str, Any]] = []
        max_evidence_records = max(1, int(os.getenv("RESEARCH_MAX_EVIDENCE_RECORDS", "12")))
        max_record_chars = max(500, int(os.getenv("RESEARCH_MAX_EVIDENCE_CHARS", "8000")))
        for task in state.research_plan:
            evidence_before = len(state.evidence)
            try:
                print(f"Executing task type: {task.task_type}")
                trace_entry = self._execute_task(state, task)
                trace_entry.update({
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "description": task.description,
                })
                if trace_entry.get("status") == "deferred":
                    task.status = "deferred"
                else:
                    task.status = "completed"
                    state.completed_tasks.append(task.task_id)
                trace_entry["evidence_count"] = len(state.evidence) - evidence_before
            except (RuntimeError, TypeError, ValueError) as exc:
                task.status = "failed"
                error = f"{task.task_type}: {exc}"
                state.errors.append(error)
                trace_entry = {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "description": task.description,
                    "specialist": self._specialist_for(task.task_type),
                    "provider_method": self._provider_method_for(task.task_type),
                    "status": "failed",
                    "evidence_count": len(state.evidence) - evidence_before,
                    "error": str(exc),
                }
            routing_trace.append(trace_entry)

        state.evidence = self._deduplicate(state.evidence)
        # Bound downstream state before creating derived context or token inputs.
        state.evidence = [
            record.model_copy(update={"content": (record.content or "")[:max_record_chars]})
            for record in state.evidence[:max_evidence_records]
        ]
        # These are aliases to the canonical evidence objects, not extra copies
        # of each full document. Derived notebook views are built once by graph.
        state.retrieved_documents = state.evidence
        state.metadata["routing_trace"] = routing_trace
        state.metadata["context"] = self.build_context(state)
        return state

    def _execute_task(
        self,
        state: ResearchState,
        task: ResearchTask,
    ) -> dict[str, Any]:
        task_type = task.task_type
        if task_type in (TaskType.FINANCIAL, TaskType.MARKET):
            data = self._provider.financial(state.ticker, state.user_query)
            state.financial_data.update(data)
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "financial",
                "status": "completed",
                "result_count": len(data),
                "result_summary": f"{len(data)} financial/market fields",
            }

        if task_type is TaskType.NEWS:
            records = self._provider.news(state.ticker, state.user_query)
            max_records = max(1, int(os.getenv("RESEARCH_MAX_NEWS_RECORDS", "12")))
            max_chars = max(500, int(os.getenv("RESEARCH_MAX_EVIDENCE_CHARS", "8000")))
            records = [
                record.model_copy(update={"content": (record.content or "")[:max_chars]})
                for record in records[:max_records]
            ]
            state.news_data.extend(records)
            state.evidence.extend(records)
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "news",
                "status": "completed",
                "result_count": len(records),
                "result_summary": f"{len(records)} news evidence records",
            }

        if task_type is TaskType.SEC:
            records = self._provider.filings(state.ticker, state.user_query)
            max_records = max(1, int(os.getenv("RESEARCH_MAX_SEC_FILINGS", "4")))
            max_chars = max(500, int(os.getenv("RESEARCH_MAX_EVIDENCE_CHARS", "8000")))
            records = [
                record.model_copy(update={"content": (record.content or "")[:max_chars]})
                for record in records[:max_records]
            ]
            state.filings.extend(records)
            state.evidence.extend(records)
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "filings",
                "status": "completed",
                "result_count": len(records),
                "result_summary": f"{len(records)} filing evidence records",
            }

        if task_type is TaskType.RISK:
            # A lightweight specialist pass over the evidence retrieved so far.
            risk_terms = (
                "risk", "uncertain", "decline", "loss", "debt",
                "litigation", "regulatory", "weak", "volatility",
            )
            findings = []
            max_findings = max(1, int(os.getenv("RESEARCH_MAX_RISK_FINDINGS", "20")))
            for record in state.evidence:
                for sentence in re.split(r"(?<=[.!?])\s+", (record.content or "")[:12000]):
                    if any(term in sentence.casefold() for term in risk_terms):
                        findings.append({
                            "source": record.citation(),
                            "finding": sentence.strip()[:500],
                            "url": record.url,
                        })
                        if len(findings) >= max_findings:
                            break
                if len(findings) >= max_findings:
                    break
            state.metadata["risk_analysis"] = findings
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "local_evidence_analysis",
                "status": "completed",
                "result_count": len(findings),
                "result_summary": f"{len(findings)} candidate risk statements",
            }

        if task_type is TaskType.REPORT:
            # Context construction is the concrete action for this planned step.
            context = self.build_context(state)
            state.metadata["context"] = context
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "build_context",
                "status": "completed",
                "result_count": len(context),
                "result_summary": f"{len(context)} context characters prepared",
            }

        if task_type is TaskType.EVALUATION:
            # The real evaluation is performed by EvaluationService after
            # generation; don't falsely label a not-yet-run evaluator as done.
            return {
                "specialist": self._specialist_for(task_type),
                "provider_method": "EvaluationService (downstream handoff)",
                "status": "deferred",
                "result_count": 0,
                "result_summary": "Deferred until an answer is generated",
            }

        raise ValueError(f"Unsupported research task type: {task_type}")

    @staticmethod
    def _specialist_for(task_type: TaskType) -> str:
        return {
            TaskType.FINANCIAL: "Financial Analyst",
            TaskType.MARKET: "Market Analyst",
            TaskType.NEWS: "News Analyst",
            TaskType.SEC: "SEC Filing Analyst",
            TaskType.RISK: "Risk Analyst",
            TaskType.REPORT: "Research Synthesis Analyst",
            TaskType.EVALUATION: "Answer Quality Evaluator",
        }.get(task_type, "Research Analyst")

    @staticmethod
    def _provider_method_for(task_type: TaskType) -> str:
        return {
            TaskType.FINANCIAL: "financial",
            TaskType.MARKET: "financial",
            TaskType.NEWS: "news",
            TaskType.SEC: "filings",
            TaskType.RISK: "local_evidence_analysis",
            TaskType.REPORT: "build_context",
            TaskType.EVALUATION: "EvaluationService (downstream handoff)",
        }.get(task_type, "unmapped")

    @staticmethod
    def build_context(state: ResearchState) -> str:
        sections = []
        if state.financial_data:
            sections.append(
                "Financial data:\n"
                + "\n".join(
                    f"- {key}: {value}"
                    for key, value in state.financial_data.items()
                )
            )
        max_context_chars = max(1000, int(os.getenv("RESEARCH_MAX_CONTEXT_CHARS", "6000")))
        context_length = sum(len(section) for section in sections)
        for record in state.evidence[:5]:
            section = f"Evidence: {record.citation()}\n{record.content[:500]}"
            if context_length + len(section) > max_context_chars:
                section = section[: max(0, max_context_chars - context_length)]
            if not section:
                break
            sections.append(section)
            context_length += len(section)
            if context_length >= max_context_chars:
                break
        risk_findings = state.metadata.get("risk_analysis", [])
        if risk_findings:
            sections.append(
                "Candidate risk statements (heuristic; verify against source):\n"
                + "\n".join(f"- {item['finding']}" for item in risk_findings[:10])
            )
        return "\n\n".join(sections)[:max_context_chars]

    @staticmethod
    def _deduplicate(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
        unique = {}
        for record in records:
            unique[record.document_id or record.url or record.content] = record
        return list(unique.values())
