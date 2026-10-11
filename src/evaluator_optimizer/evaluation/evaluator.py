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

import inspect
import json
import os
import re

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional provider dependency
    Groq = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional provider dependency
    OpenAI = None

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional provider dependency
    InferenceClient = None


def _import_huggingface_answer_generator():
    """Import the project's HuggingFaceAnswerGenerator from known locations.

    The source tree has used different package layouts across iterations, so
    resolve the existing project class lazily rather than importing it for
    OpenAI/Groq-only installations.
    """
    from importlib import import_module

    candidates = (
        ("evaluator_optimizer.generation.huggingface_answer_generator",
         "HuggingFaceAnswerGenerator"),
        ("evaluator_optimizer.generators.huggingface_answer_generator",
         "HuggingFaceAnswerGenerator"),
        ("evaluator_optimizer.answer_generators.huggingface_answer_generator",
         "HuggingFaceAnswerGenerator"),
        ("evaluator_optimizer.generation.answer_generator",
         "HuggingFaceAnswerGenerator"),
        ("answer_generator", "HuggingFaceAnswerGenerator"),
        ("huggingface_answer_generator", "HuggingFaceAnswerGenerator"),
    )
    errors = []
    for module_name, class_name in candidates:
        try:
            module = import_module(module_name)
            generator_class = getattr(module, class_name, None)
            if generator_class is not None:
                return generator_class
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "Could not import the project's HuggingFaceAnswerGenerator. "
        "Update _import_huggingface_answer_generator() with its actual module "
        "path, or inject an answer_generator into EvaluatorLLM."
    )

from config import (
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_PASS_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
)
from evaluator_optimizer.evaluation.metrics import EvaluationMetric
from evaluator_optimizer.evaluation.models import EvaluationResult
from evaluator_optimizer.evaluation.rubrics import EVALUATION_RUBRICS
from logging_config import get_logger

logger = get_logger("evaluation.evaluator")


class EvaluatorLLM:
    """
    Generic LLM-as-a-Judge evaluator.

    The class is intentionally independent of any particular metric.
    The metric and its rubric are supplied at evaluation time.

    Args:
        model:
            Provider-specific model identifier. Hugging Face uses the
            configured InferenceClient model.

        client:
            Optional provider client. Dependency injection supports
            deterministic unit tests without external API calls.

        provider:
            One of ``openai``, ``groq`` or ``huggingface``. When omitted,
            OpenAI is the default; an injected Groq client is detected for
            backwards compatibility.

        pass_threshold:
            Minimum score required for an evaluation to pass.
    """

    def __init__(
        self,
        model: str | None = None,
        client: object | None = None,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        provider: str | None = None,
        answer_generator: object | None = None,
    ) -> None:
        if not MIN_SCORE <= pass_threshold <= MAX_SCORE:
            raise ValueError(
                "pass_threshold must be between MIN_SCORE and MAX_SCORE"
            )

        supported_providers = {"openai", "groq", "huggingface", "hugging_face"}
        if provider is not None:
            provider = provider.strip().lower()
            if provider == "hugging_face":
                provider = "huggingface"
            if provider not in supported_providers - {"hugging_face"}:
                raise ValueError(
                    f"Unsupported evaluator provider '{provider}'. "
                    "Choose one of: openai, groq, hugging_face."
                )
        elif Groq is not None and client is not None and isinstance(client, Groq):
            # Preserve compatibility with existing callers that inject Groq.
            provider = "groq"
        else:
            provider = "openai"

        self._provider = provider
        default_model = (
            "Qwen/Qwen2.5-3B-Instruct"
            if provider == "huggingface"
            else DEFAULT_EVALUATOR_MODEL
        )
        self._model = model or default_model

        if provider == "huggingface":
            if answer_generator is None:
                generator_class = _import_huggingface_answer_generator()
                try:
                    parameters = inspect.signature(generator_class).parameters
                except (TypeError, ValueError):
                    parameters = {}

                # Reuse an injected client or construct InferenceClient when
                # the project generator supports client injection. Otherwise,
                # use its no-argument constructor; the generator may own its
                # InferenceClient configuration internally.
                client_parameter = (
                    "client" if "client" in parameters
                    else "inference_client" if "inference_client" in parameters
                    else None
                )
                if client_parameter is not None:
                    if client is None:
                        if InferenceClient is None:
                            raise ImportError(
                                "Hugging Face provider requires huggingface_hub. "
                                "Install it with `pip install huggingface_hub`."
                            )
                        client = InferenceClient(
                            model=self._model,
                            token=(
                                os.getenv("HF_TOKEN")
                                or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                            ),
                        )
                    answer_generator = generator_class(
                        **{client_parameter: client}
                    )
                else:
                    answer_generator = generator_class()
            self._answer_generator = answer_generator
            self._client = client
        elif client is None:
            self._answer_generator = answer_generator
            if provider == "groq":
                if Groq is None:
                    raise ImportError(
                        "Groq SDK is required for provider='groq'. "
                        "Install it with `pip install groq`."
                    )
                client = Groq()
            else:
                if OpenAI is None:
                    raise ImportError(
                        "OpenAI SDK is required for provider='openai'. "
                        "Install it with `pip install openai`."
                    )
                client = OpenAI()

        if provider != "huggingface":
            self._answer_generator = answer_generator
            self._client = client

        self._pass_threshold = pass_threshold

    @property
    def model(self) -> str:
        """Return the configured evaluator model identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Return the configured evaluation backend."""
        return self._provider

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

        if self._provider == "huggingface":
            result = self._evaluate_with_huggingface(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        elif self._provider == "groq":
            from pydantic import TypeAdapter

            json_schema = TypeAdapter(EvaluationResult).json_schema()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "EvaluationResult",
                        "schema": json_schema,
                    },
                },
            )
            result_json = json.loads(response.choices[0].message.content)
            result = EvaluationResult(**result_json)
        else:  # OpenAI Responses API
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=EvaluationResult,
            )
            result = response.output_parsed

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

    def _evaluate_with_huggingface(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> EvaluationResult:
        """Evaluate using the project's HuggingFaceAnswerGenerator.

        The generator is expected to expose ``generate(prompt)`` or
        ``generate_answer(prompt)``. Common string/dict response shapes are
        accepted to keep the adapter simple. The project's generator owns or
        receives its Hugging Face InferenceClient configuration.
        """
        prompt = (
            f"{system_prompt}\n\n{user_prompt}\n\n"
            "Return only one valid JSON object. Do not wrap it in Markdown."
        )
        generator = self._answer_generator
        generate = getattr(generator, "generate", None)
        if not callable(generate):
            generate = getattr(generator, "generate_answer", None)
        if not callable(generate):
            raise TypeError(
                "HuggingFaceAnswerGenerator must expose generate(prompt) "
                "or generate_answer(prompt)."
            )

        generated_response = generate(prompt)
        if isinstance(generated_response, str):
            generated = generated_response
        elif isinstance(generated_response, dict):
            generated = (
                generated_response.get("generated_text")
                or generated_response.get("content")
                or generated_response.get("text")
            )
        else:
            generated = getattr(generated_response, "content", None)
            if generated is None:
                generated = getattr(generated_response, "generated_text", None)

        if not isinstance(generated, str) or not generated.strip():
            raise ValueError(
                "HuggingFaceAnswerGenerator returned an empty or unsupported response."
            )

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", generated.strip())
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            payload = None
            for index, character in enumerate(cleaned):
                if character == "{":
                    try:
                        payload, _ = decoder.raw_decode(cleaned[index:])
                        break
                    except json.JSONDecodeError:
                        continue
            if payload is None:
                raise ValueError(
                    "Hugging Face evaluator response did not contain valid JSON."
                )

        return EvaluationResult(**payload)

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