from evaluator_optimizer.evaluation.metrics import EvaluationMetric


def test_all_required_metrics_are_registered() -> None:
    expected = {
        "relevance",
        "groundedness",
        "accuracy",
        "temporal_validity",
        "clarity",
    }

    actual = {metric.value for metric in EvaluationMetric}

    assert actual == expected