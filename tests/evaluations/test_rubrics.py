from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.rubrics import EVALUATION_RUBRICS


def test_every_metric_has_a_rubric() -> None:
    for metric in EvaluationMetric:
        assert metric in EVALUATION_RUBRICS
        assert EVALUATION_RUBRICS[metric].strip()


def test_temporal_validity_rubric_contains_required_concepts() -> None:
    rubric = EVALUATION_RUBRICS[
        EvaluationMetric.TEMPORAL_VALIDITY
    ].lower()

    assert "reporting period" in rubric
    assert "stale" in rubric
    assert "historical" in rubric


def test_clarity_rubric_contains_required_concepts() -> None:
    rubric = EVALUATION_RUBRICS[
        EvaluationMetric.CLARITY
    ].lower()

    assert "directness" in rubric
    assert "ambiguity" in rubric
    assert "readability" in rubric