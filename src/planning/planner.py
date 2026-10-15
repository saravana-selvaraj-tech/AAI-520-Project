"""Deterministic research planning with a stable Cohort 2 contract."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    FINANCIAL = "financial"
    MARKET = "market"
    NEWS = "news"
    SEC = "sec"
    RISK = "risk"
    REPORT = "report"
    EVALUATION = "evaluation"


class ResearchTask(BaseModel):
    """One executable unit in a research plan."""

    task_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    task_type: TaskType
    description: str
    priority: int = 1
    status: str = "pending"


class Planner:
    """Convert a user request into deterministic, executable tasks."""

    SYMBOLS: ClassVar[set[str]] = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "SNDK", "NTNX"}

    def plan(self, query: str, ticker: str | None = None) -> list[ResearchTask]:
        if not query.strip():
            raise ValueError("query must not be empty")

        normalized = query.casefold()
        tasks: list[ResearchTask] = []
        if any(word in normalized for word in ("price", "market", "stock", "return", "volatility")):
            tasks.append(ResearchTask(task_type=TaskType.MARKET, description="Analyze market performance."))
        if any(word in normalized for word in ("news", "headline", "sentiment", "event")):
            tasks.append(ResearchTask(task_type=TaskType.NEWS, description="Retrieve and analyze recent company news."))
        if any(word in normalized for word in ("sec", "filing", "10-k", "10-q", "regulatory")):
            tasks.append(ResearchTask(task_type=TaskType.SEC, description="Retrieve relevant regulatory filings."))
        if not tasks or any(word in normalized for word in ("financial", "earnings", "revenue", "profit", "eps")):
            tasks.insert(0, ResearchTask(task_type=TaskType.FINANCIAL, description="Retrieve and analyze financial performance."))
        if any(word in normalized for word in ("risk", "uncertainty", "downside")) or not any(task.task_type is TaskType.RISK for task in tasks):
            tasks.append(ResearchTask(task_type=TaskType.RISK, description="Identify material risks and uncertainties."))
        
        tasks.extend([
            ResearchTask(task_type=TaskType.REPORT, description="Synthesize an evidence-supported research report."),
            ResearchTask(task_type=TaskType.EVALUATION, description="Evaluate relevance, grounding, and completeness."),
        ])
        
        return tasks

    @classmethod
    def extract_ticker(cls, query: str) -> str | None:
        match = re.search(r"\b[A-Z]{2,5}\b", query)
        if match and match.group() in cls.SYMBOLS:
            return match.group()
        for name, symbol in {"apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
                    "sandisk": "SNDK", "nutanix": "NTNX",
                    "amazon": "AMZN", "alphabet": "GOOGL", "google": "GOOGL"}.items():
            if name in query.casefold():
                return symbol
        return None