"""
LLM-backed evaluator.

EvaluatorLLM implements the generic LLM-as-a-Judge mechanism used
by all evaluation metrics.

Metric-specific classes such as ClarityEvaluator and
TemporalValidityEvaluator are intentionally thin adapters around
this component.

Architecture:

    User Query
        |
        v
    Generated Answer
        |
        v
    Retrieved Context
        |
        v
    EvaluatorLLM
        |
        +---- Relevance
        +---- Groundedness
        +---- Accuracy
        +---- Temporal Validity
        +---- Clarity
        |
        v
    EvaluationResult
        |
        v
    Reflection / Optimization
"""

from __future__ import annotations

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional provider dependency
    Groq = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None

from evaluator_optimizer.config import (
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_PASS_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.evaluation.rubrics import EVALUATION_RUBRICS
from evaluator_optimizer.logging_config import get_logger

logger = get_logger("evaluation.evaluator")


class EvaluatorLLM:
    """
    Generic LLM-as-a-Judge evaluator.

    The class is intentionally independent of any particular metric.
    The metric and its rubric are supplied at evaluation time.

    Args:
        model:
            OpenAI model used for evaluation.

        client:
            Optional OpenAI/Groq client. Dependency injection makes the
            evaluator easy to unit test.

        pass_threshold:
            Minimum score required for an evaluation to pass.
    """

    def __init__(
        self,
        model: str | None = None,
        client: object | None = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        if not MIN_SCORE <= pass_threshold <= MAX_SCORE:
            raise ValueError(
                "pass_threshold must be between MIN_SCORE and MAX_SCORE"
            )

        self._model = model or DEFAULT_EVALUATOR_MODEL

        if client is None:
            if OpenAI is None:
                raise ImportError(
                    "OpenAI SDK is required when no client is injected."
                )
            client = OpenAI()

        self._client = client

        self._pass_threshold = pass_threshold

    @property
    def model(self) -> str:
        """Return the configured evaluator model."""
        return self._model

    @property
    def pass_threshold(self) -> float:
        """Return the configured pass threshold."""
        return self._pass_threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        metric: EvaluationMetric,
    ) -> EvaluationResult:
        """
        Evaluate an answer for one metric.

        Args:
            query:
                Original user question.

            answer:
                Generated research answer.

            context:
                Evidence retrieved and supplied to the answer.

            metric:
                Evaluation dimension.

        Returns:
            Structured EvaluationResult.
        """
        if not query.strip():
            raise ValueError("query cannot be empty")

        if not answer.strip():
            raise ValueError("answer cannot be empty")

        if not context.strip():
            raise ValueError("context cannot be empty")

        logger.info(
            "Evaluation started | metric=%s | model=%s",
            metric.value,
            self._model,
        )

        rubric = EVALUATION_RUBRICS[metric]

        system_prompt = self._build_system_prompt(
            metric=metric,
            rubric=rubric,
        )

        user_prompt = self._build_user_prompt(
            query=query,
            answer=answer,
            context=context,
            metric=metric,
        )

        if Groq is not None and isinstance(self._client, Groq):
            from pydantic import TypeAdapter
            json_schema = TypeAdapter(EvaluationResult).json_schema()
            response = self._client.chat.completions.create(
                #model="llama-3.3-70b-versatile",  # or other Llama models on Groq
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "EvaluationResult",
                        #"strict": True,
                        "schema": json_schema,
                    }
                }
                #temperature=0.2,
                #max_tokens=256,
            )
            import json
            result_json = json.loads(response.choices[0].message.content)
            result = EvaluationResult(**result_json)
        #elif isinstance(self._client, OpenAI):
        else:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                text_format=EvaluationResult,
            )
            result = response.output_parsed

        #else:
        #    raise Exception(f"Unsupported LLM Client : {self._client}")

        if result is None:
            raise ValueError(
                "Evaluator LLM returned no structured evaluation result."
            )

        # The requested metric is authoritative. The model should
        # not be allowed to accidentally return a different metric.
        result.metric = metric

        # Pass/fail is controlled by the application threshold rather
        # than trusting the LLM's independently generated boolean.
        result.passed = result.score >= self._pass_threshold

        logger.info(
            "Evaluation completed | metric=%s | score=%.4f | passed=%s",
            metric.value,
            result.score,
            result.passed,
        )

        return result

    def evaluate_all(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:
        """
        Evaluate an answer against every configured metric.

        Returns:
            One EvaluationResult for each EvaluationMetric.
        """
        return [
            self.evaluate(
                query=query,
                answer=answer,
                context=context,
                metric=metric,
            )
            for metric in EvaluationMetric
        ]

    @staticmethod
    def _build_system_prompt(
        metric: EvaluationMetric,
        rubric: str,
    ) -> str:
        """Build the evaluator system prompt."""
        return f"""
You are an expert evaluator for an autonomous investment research
agent.

Your task is to evaluate a generated answer using ONLY the supplied
user question, generated answer, and retrieved context.

Evaluation metric:
{metric.value}

Evaluation rubric:
{rubric}

Evaluation rules:
1. Return a score between 0.0 and 1.0.
2. Use the supplied evidence when judging factual claims.
3. Do not invent missing evidence.
4. Provide a concise but specific rationale.
5. Identify concrete issues when quality is below the desired level.
6. Assign issue severity according to the seriousness of the issue.
7. Do not evaluate a different metric from the requested metric.
8. Do not provide investment advice.
9. Do not rewrite the answer. Only evaluate it.

The score represents the degree to which the answer satisfies the
requested evaluation metric.
""".strip()

    @staticmethod
    def _build_user_prompt(
        query: str,
        answer: str,
        context: str,
        metric: EvaluationMetric,
    ) -> str:
        """Build the evaluation input supplied to the LLM."""
        return f"""
Evaluate the following investment research response.

METRIC:
{metric.value}

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return a structured evaluation containing:
- metric
- score
- passed
- rationale
- issues

The score must be between 0.0 and 1.0.
""".strip()