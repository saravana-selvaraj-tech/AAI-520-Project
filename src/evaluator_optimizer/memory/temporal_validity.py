"""Temporal validity utilities for research memory.

This module identifies potentially stale evidence stored in memory.

Financial research is time-sensitive. A previously valid claim can become
outdated when a newer filing, earnings release, market update, or other
authoritative source becomes available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from evaluator_optimizer.config import DEFAULT_MEMORY_MAX_AGE_DAYS


@dataclass(frozen=True)
class TemporalValidityResult:
    """Result of evaluating the temporal validity of evidence."""

    evidence_id: str
    is_stale: bool
    evidence_date: datetime | None
    reference_date: datetime
    reason: str


class TemporalValidityChecker:
    """Determine whether stored evidence should be considered stale."""

    DATE_FIELDS = (
        "published_at",
        "publication_date",
        "source_date",
        "document_date",
        "date",
        "timestamp",
    )

    def __init__(self, max_age_days: int = DEFAULT_MEMORY_MAX_AGE_DAYS) -> None:
        """Initialize the temporal validity checker.

        Args:
            max_age_days: Maximum acceptable age for evidence when no
                newer evidence is available.

        Raises:
            ValueError: If max_age_days is not positive.
        """
        if max_age_days <= 0:
            raise ValueError("max_age_days must be greater than zero.")

        self.max_age_days = max_age_days

    def evaluate(
        self,
        evidence_id: str,
        evidence: Any,
        reference_date: datetime | None = None,
    ) -> TemporalValidityResult:
        """Evaluate whether evidence is stale.

        Args:
            evidence_id: Identifier for the evidence.
            evidence: Evidence object or dictionary.
            reference_date: Date against which freshness is evaluated.

        Returns:
            TemporalValidityResult describing evidence freshness.
        """
        reference = self._normalize_datetime(
            reference_date or datetime.now(timezone.utc)
        )

        evidence_date = self._extract_date(evidence)

        if evidence_date is None:
            return TemporalValidityResult(
                evidence_id=evidence_id,
                is_stale=False,
                evidence_date=None,
                reference_date=reference,
                reason="No evidence date was available.",
            )

        age_days = (reference - evidence_date).days

        if age_days > self.max_age_days:
            return TemporalValidityResult(
                evidence_id=evidence_id,
                is_stale=True,
                evidence_date=evidence_date,
                reference_date=reference,
                reason=(
                    f"Evidence is {age_days} days old, exceeding "
                    f"the {self.max_age_days}-day freshness window."
                ),
            )

        return TemporalValidityResult(
            evidence_id=evidence_id,
            is_stale=False,
            evidence_date=evidence_date,
            reference_date=reference,
            reason=f"Evidence is {age_days} days old.",
        )

    def _extract_date(self, evidence: Any) -> datetime | None:
        """Extract a date from common evidence representations."""
        if isinstance(evidence, dict):
            for field_name in self.DATE_FIELDS:
                if field_name in evidence:
                    parsed = self._parse_datetime(evidence[field_name])
                    if parsed is not None:
                        return parsed

        for field_name in self.DATE_FIELDS:
            if hasattr(evidence, field_name):
                parsed = self._parse_datetime(
                    getattr(evidence, field_name)
                )
                if parsed is not None:
                    return parsed

        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        """Parse a datetime from common representations."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return TemporalValidityChecker._normalize_datetime(value)

        if not isinstance(value, str):
            return None

        normalized = value.strip()

        if not normalized:
            return None

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        return TemporalValidityChecker._normalize_datetime(parsed)

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        """Normalize a datetime to timezone-aware UTC."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)