"""Hugging Face answer generation for the research workflow."""

from __future__ import annotations

from typing import Any


class HuggingFaceAnswerGenerator:
    """Generate grounded answers through an injected Hugging Face client."""

    def __init__(
        self,
        client: Any,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
    ) -> None:
        self._client = client
        self.model = model

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return f"No retrieved evidence is available to answer: {query}"
        system = (
            "You are an investment research agent. Answer only from the supplied "
            "evidence. State when evidence is missing, preserve dates and numbers, "
            "and cite sources by their labels."
        )
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nEvidence:\n{context}",
                },
            ],
            model=self.model,
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""