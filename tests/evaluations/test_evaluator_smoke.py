import os

import pytest

from evaluator_optimizer.evaluation.evaluator import EvaluatorLLM
from evaluator_optimizer.evaluation.metrics import EvaluationMetric


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not configured",
)
def test_real_evaluator_llm() -> None:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROK_API_KEY"))
    #evaluator = EvaluatorLLM(model="meta-llama/llama-prompt-guard-2-22m", client=client)
    evaluator = EvaluatorLLM(model="openai/gpt-oss-120b", client=client)

    result = evaluator.evaluate(
        query="What was Apple's FY2025 revenue?",
        answer=(
            "Apple reported approximately $416.2 billion "
            "in FY2025 revenue."
        ),
        context=(
            "Apple FY2025 annual report states that net sales "
            "were approximately $416.2 billion."
        ),
        metric=EvaluationMetric.GROUNDEDNESS,
    )

    print(f"Result obtained: {result}")
    assert result.metric == EvaluationMetric.GROUNDEDNESS
    assert 0.0 <= result.score <= 1.0
    assert result.rationale