"""Tests for temporal evidence validity."""

from datetime import datetime, timezone

import pytest

from evaluator_optimizer.memory.temporal_validity import (
    TemporalValidityChecker,
)


REFERENCE_DATE = datetime(
    2026,
    9,
    29,
    tzinfo=timezone.utc,
)


def test_recent_evidence_is_not_stale() -> None:
    """Recent evidence should remain reusable."""
    checker = TemporalValidityChecker(max_age_days=365)

    evidence = {
        "evidence_id": "sec-2026-001",
        "publication_date": "2026-06-30T00:00:00+00:00",
    }

    result = checker.evaluate(
        evidence_id="sec-2026-001",
        evidence=evidence,
        reference_date=REFERENCE_DATE,
    )

    assert result.is_stale is False
    assert result.evidence_id == "sec-2026-001"


def test_old_evidence_is_stale() -> None:
    """Evidence older than the configured window should be stale."""
    checker = TemporalValidityChecker(max_age_days=365)

    evidence = {
        "evidence_id": "sec-2024-001",
        "publication_date": "2024-01-01T00:00:00+00:00",
    }

    result = checker.evaluate(
        evidence_id="sec-2024-001",
        evidence=evidence,
        reference_date=REFERENCE_DATE,
    )

    assert result.is_stale is True
    assert "exceeding" in result.reason


def test_missing_date_is_not_assumed_to_be_stale() -> None:
    """Undated evidence should not be falsely classified as stale."""
    checker = TemporalValidityChecker(max_age_days=365)

    evidence = {
        "evidence_id": "unknown-date",
        "content": "Historical research finding.",
    }

    result = checker.evaluate(
        evidence_id="unknown-date",
        evidence=evidence,
        reference_date=REFERENCE_DATE,
    )

    assert result.is_stale is False
    assert result.evidence_date is None


def test_invalid_max_age_is_rejected() -> None:
    """Freshness window must be positive."""
    with pytest.raises(ValueError):
        TemporalValidityChecker(max_age_days=0)