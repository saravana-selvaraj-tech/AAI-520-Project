"""
Memory-aware research orchestration.

Coordinates Cohort 2 response generation, memory retrieval, and the
Cohort 3 evaluator-optimizer workflow.

Cohort 2 owns initial response generation through its router service.
Cohort 3 owns evaluation, reflection, optimization, and persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.graph import AgentGraph

from config import DEFAULT_MEMORY_MAX_RESULTS
from logging_config import get_logger
from evaluator_optimizer.memory.context import MemoryContext
from evaluator_optimizer.memory.memory_builder import MemoryBuilder
from evaluator_optimizer.memory.memory_service import MemoryService
from evaluator_optimizer.memory.repository import MemoryQuery
from evaluator_optimizer.optimization.optimization_controller import (
    OptimizationController,
)
from evaluator_optimizer.orchestration.evaluation_trace import (
    EvaluationTrace,
)
if TYPE_CHECKING:
    from routing.router import RouterService
    from tools.providers import ResearchProvider


logger = get_logger("orchestration.memory_aware")



class MemoryAwareResearchService:
    """Coordinate Cohort 2 generation with Cohort 3 optimization and memory."""

    def __init__(
        self,
        router_service: RouterService,
        memory_service: MemoryService,
        memory_builder: MemoryBuilder,
        optimization_controller: OptimizationController,
        data_provider: ResearchProvider | None = None,
        answer_generator=None,
    ) -> None:
        self._router_service = router_service
        self._memory_service = memory_service
        self._memory_builder = memory_builder
        self._optimization_controller = optimization_controller
        self._data_provider = data_provider
        self._answer_generator = answer_generator

    def run(
        self,
        exchange_symbol: str,
        query: str,
        context: str,
        topics: list[str] | None = None,
    ) -> tuple[EvaluationTrace, MemoryContext]:
        """Generate an initial response through the shared router and optimize it."""
        return self._run(
            exchange_symbol=exchange_symbol,
            query=query,
            context=context,
            topics=topics,
            initial_answer=None,
        )

    def run_question(
        self,
        question: str,
        exchange_symbol: str,
    ) -> tuple[EvaluationTrace, MemoryContext]:
        """Research a question using memory, then evaluate and improve it.

        The method owns the complete question-only workflow: it retrieves prior
        memory, creates an ``AgentGraph`` for Cohort 2 routing, runs the initial
        answer through that graph/router, delegates evaluation and reflection
        to Cohort 3, and persists the final answer for later runs.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if not exchange_symbol or not exchange_symbol.strip():
            raise ValueError("exchange_symbol must not be empty")
        if self._data_provider is None:
            raise ValueError(
                "data_provider is required for question-only execution"
            )

        question = question.strip()
        exchange_symbol = exchange_symbol.strip().upper()
        logger.info(
            "Question workflow started | symbol=%s",
            exchange_symbol,
        )

        # Create one graph/router and share its router with the optimizer so
        # any reflection-driven revision uses the same Cohort 2 generation path.
        agent_graph = AgentGraph(
            provider=self._data_provider,
            answer_generator=self._answer_generator,
        )
        router_service = agent_graph.get_router_service()
        self._router_service = router_service
        self._optimization_controller.set_router_service(router_service)

        memory_context = self.retrieve_memory(
            exchange_symbol=exchange_symbol,
            query=question,
            topics=[],
        )

        if memory_context.memories:
            initial_context = self._enrich_context_with_memory(
                context="",
                memory_context=memory_context,
            )
            initial_answer = router_service.route(
                query=question,
                context=initial_context,
            )
        else:
            research_state = agent_graph.run(
                query=question,
                ticker=exchange_symbol,
            )
            initial_answer = research_state.draft_report
            initial_context = research_state.metadata.get("context") or ""

        enriched_context = (
            initial_context
            if memory_context.memories
            else self._enrich_context_with_memory(
                context=initial_context,
                memory_context=memory_context,
            )
        )
        trace = self._optimization_controller.run(
            query=question,
            answer=initial_answer,
            context=enriched_context,
        )

        retrieved_ids = [
            memory.memory_id for memory in memory_context.memories
        ]
        trace.record_memory_retrieved(retrieved_ids)
        final_answer = trace.optimized_answer or trace.initial_answer

        memory = self._memory_builder.build(
            exchange_symbol=exchange_symbol,
            query=question,
            research_summary=final_answer,
            key_findings=[],
            claims=[],
            evidence=[],
            identified_gaps=memory_context.research_gaps,
            evaluation_results=trace.final_evaluation,
            topics=None,
        )
        self._memory_service.save(memory)
        trace.record_memory_applied(retrieved_ids)

        logger.info(
            "Question workflow completed | symbol=%s | retrieved=%d",
            exchange_symbol,
            len(retrieved_ids),
        )
        return trace, memory_context

    def run_with_initial_answer(
        self,
        exchange_symbol: str,
        query: str,
        context: str,
        initial_answer: str,
        topics: list[str] | None = None,
    ) -> tuple[EvaluationTrace, MemoryContext]:
        """Evaluate and optimize a report already generated by AgentGraph.

        This handoff method avoids regenerating the initial report. The same
        graph-owned RouterService remains wired into AnswerOptimizer for every
        suggestion-driven revision.
        """
        if not initial_answer or not initial_answer.strip():
            raise ValueError("initial_answer must not be empty")

        return self._run(
            exchange_symbol=exchange_symbol,
            query=query,
            context=context,
            topics=topics,
            initial_answer=initial_answer,
        )

    def _run(
        self,
        exchange_symbol: str,
        query: str,
        context: str,
        topics: list[str] | None,
        initial_answer: str | None,
    ) -> tuple[EvaluationTrace, MemoryContext]:
        """Run memory retrieval, evaluation/optimization, and persistence."""
        logger.info(
            "Memory-aware workflow started | symbol=%s",
            exchange_symbol,
        )

        memory_context = self.retrieve_memory(
            exchange_symbol=exchange_symbol,
            query=query,
            topics=topics or [],
        )

        # Make retrieved memory available to Cohort 3 evaluation/optimization.
        # Historical memory is labelled separately from the current evidence.
        enriched_context = self._enrich_context_with_memory(
            context=context,
            memory_context=memory_context,
        )

        if initial_answer is None:
            initial_response = self._router_service.route(
                query=query,
                context=enriched_context,
            )
            logger.info(
                "Initial response generated by Cohort 2 router | symbol=%s",
                exchange_symbol,
            )
        else:
            initial_response = initial_answer
            logger.info(
                "Using AgentGraph-generated report | symbol=%s",
                exchange_symbol,
            )

        trace = self._optimization_controller.run(
            query=query,
            answer=initial_response,
            context=enriched_context,
        )

        trace.record_memory_retrieved(
            [memory.memory_id for memory in memory_context.memories]
        )

        final_answer = trace.optimized_answer or trace.initial_answer

        memory = self._memory_builder.build(
            exchange_symbol=exchange_symbol,
            query=query,
            research_summary=final_answer,
            key_findings=[],
            claims=[],
            evidence=[],
            identified_gaps=memory_context.research_gaps,
            evaluation_results=trace.final_evaluation,
            topics=topics,
        )

        self._memory_service.save(memory)

        trace.record_memory_applied(
            [memory_item.memory_id for memory_item in memory_context.memories]
        )

        logger.info(
            "Memory-aware workflow completed | retrieved=%d | applied=%d",
            len(trace.memory_retrieved),
            len(trace.memory_applied),
        )

        return trace, memory_context

    @staticmethod
    def _enrich_context_with_memory(
        context: str,
        memory_context: MemoryContext,
    ) -> str:
        """Append retrieved summaries and known gaps to the runtime context.

        Memory is presented as prior research, not as independently verified
        current evidence. The evaluation and optimization workflow can then
        reuse relevant details while retaining the supplied current context.
        """
        sections = [context.strip()] if context.strip() else []

        for memory in memory_context.memories:
            if memory.research_summary.strip():
                sections.append(
                    "Prior research memory (verify against current evidence):\n"
                    + memory.research_summary.strip()
                )
            if memory.evidence:
                sections.append(
                    "Evidence recorded in prior memory:\n"
                    + "\n".join(f"- {item}" for item in memory.evidence)
                )

        if memory_context.research_gaps:
            sections.append(
                "Previously identified research gaps:\n"
                + "\n".join(
                    f"- {gap}" for gap in memory_context.research_gaps
                )
            )

        return "\n\n".join(sections)

    def retrieve_memory(
        self,
        exchange_symbol: str,
        query: str,
        topics: list[str],
    ) -> MemoryContext:
        """Retrieve relevant memories for a research request."""
        query_signature = self._memory_builder.create_query_signature(query)

        results = self._memory_service.search(
            MemoryQuery(
                exchange_symbol=exchange_symbol,
                topics=topics,
                query_signature=query_signature,
                query_text=None,
                max_results=DEFAULT_MEMORY_MAX_RESULTS,
            )
        )

        context = MemoryContext()

        for memory in results:
            context.add_memory(memory)

            for gap in memory.identified_gaps:
                context.add_research_gap(gap)

        return context
