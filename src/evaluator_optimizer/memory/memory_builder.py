"""
Memory builder.

Transforms a completed research execution into a reusable
MemoryItem.

The builder deliberately stores structured research knowledge rather
than simply saving the generated answer.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from uuid import uuid4

from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.memory.models import MemoryItem


class MemoryBuilder:
    """Build MemoryItem objects from completed research."""

    def build(
        self,
        exchange_symbol: str,
        query: str,
        research_summary: str,
        key_findings: list[str],
        claims: list[str],
        evidence: list[str],
        identified_gaps: list[str],
        evaluation_results: list[EvaluationResult],
        topics: list[str] | None = None,
    ) -> MemoryItem:
        """
        Create reusable memory from a completed research run.
        """
        now = datetime.now(timezone.utc)

        evaluation_summary = {
            result.metric.value: result.score
            for result in evaluation_results
        }

        return MemoryItem(
            memory_id=str(uuid4()),
            exchange_symbol=exchange_symbol.upper(),
            query=query,
            query_signature=self.create_query_signature(query),
            topics=topics or [],
            research_summary=research_summary,
            key_findings=key_findings,
            claims=claims,
            evidence=evidence,
            identified_gaps=identified_gaps,
            evaluation_summary=evaluation_summary,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def create_query_signature(query: str) -> str:
        """
        Generate a stable normalized signature for a research query.

        Minor whitespace and case differences should result in the
        same signature.
        """
        normalized = re.sub(
            r"\s+",
            " ",
            query.strip().lower(),
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()