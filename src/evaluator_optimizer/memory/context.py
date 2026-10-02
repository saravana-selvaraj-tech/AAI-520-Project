"""
MemoryContext.

MemoryContext is the runtime object passed between memory retrieval,
research, evaluation, reflection, and optimization.

It records:
- reusable memories,
- evidence reused from prior runs,
- evidence considered stale,
- research gaps discovered previously.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from evaluator_optimizer.memory.models import MemoryItem


class MemoryContext(BaseModel):
    """
    Runtime memory context for one research execution.
    """

    memories: list[MemoryItem] = Field(
        default_factory=list,
    )

    reused_evidence_ids: list[str] = Field(
        default_factory=list,
    )

    stale_evidence_ids: list[str] = Field(
        default_factory=list,
    )

    research_gaps: list[str] = Field(
        default_factory=list,
    )

    def add_memory(
        self,
        memory: MemoryItem,
    ) -> None:
        """Add a reusable memory to the context."""
        if not any(
            item.memory_id == memory.memory_id
            for item in self.memories
        ):
            self.memories.append(memory)

    def add_reused_evidence(
        self,
        evidence_id: str,
    ) -> None:
        """Record evidence reused by the current research."""
        if evidence_id not in self.reused_evidence_ids:
            self.reused_evidence_ids.append(evidence_id)

    def add_stale_evidence(
        self,
        evidence_id: str,
    ) -> None:
        """Record evidence identified as stale."""
        if evidence_id not in self.stale_evidence_ids:
            self.stale_evidence_ids.append(evidence_id)

    def add_research_gap(
        self,
        gap: str,
    ) -> None:
        """Record a research gap."""
        if gap not in self.research_gaps:
            self.research_gaps.append(gap)