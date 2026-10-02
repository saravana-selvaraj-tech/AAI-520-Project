import pytest

from evaluator_optimizer.evaluation.severity import (
    IssueSeverity,
    determine_severity,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.00, IssueSeverity.NONE),
        (0.80, IssueSeverity.NONE),
        (0.79, IssueSeverity.LOW),
        (0.65, IssueSeverity.LOW),
        (0.64, IssueSeverity.MEDIUM),
        (0.50, IssueSeverity.MEDIUM),
        (0.49, IssueSeverity.HIGH),
        (0.30, IssueSeverity.HIGH),
        (0.29, IssueSeverity.CRITICAL),
        (0.00, IssueSeverity.CRITICAL),
    ],
)
def test_determine_severity(
    score: float,
    expected: IssueSeverity,
) -> None:
    assert determine_severity(score) == expected


@pytest.mark.parametrize(
    "score",
    [-0.01, 1.01, 2.0],
)
def test_determine_severity_rejects_invalid_score(
    score: float,
) -> None:
    with pytest.raises(ValueError):
        determine_severity(score)