"""Deterministic, inspectable prompt-chain stages for research evidence."""

from __future__ import annotations

import re

from agent.state import EvidenceRecord


def preprocess_text(text: str) -> str:
    """Normalize whitespace and remove empty fragments from source text."""
    return " ".join((text or "").split())


def classify_text(text: str) -> str:
    """Assign a transparent rule-based sentiment label."""
    lowered = (text or "").casefold()
    positive = sum(
        word in lowered
        for word in ("growth", "increase", "increased", "profit", "strong", "improved")
    )
    negative = sum(
        word in lowered
        for word in ("decline", "loss", "risk", "weak", "decrease", "decreased")
    )
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def extract_facts(text: str) -> list[str]:
    """Extract candidate evidence sentences for inspection, not verified claims."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text or "")
        if sentence.strip()
    ]
    keywords = (
        "revenue", "earnings", "profit", "loss", "growth", "decline",
        "risk", "debt", "cash flow", "margin", "guidance", "price",
        "return", "volatility", "increase", "decrease", "%",
    )
    matches = [
        sentence for sentence in sentences
        if any(keyword in sentence.casefold() for keyword in keywords)
        or re.search(r"\d", sentence)
    ]
    return matches or sentences[:1]


def summarize_text(text: str, max_length: int = 500) -> str:
    """Create a concise extractive summary from the leading source text."""
    normalized = preprocess_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def process_evidence(
    records: list[EvidenceRecord],
    *,
    max_records: int = 12,
    max_source_chars: int = 4_000,
    max_preview_chars: int = 1_200,
    max_facts_per_record: int = 5,
) -> list[dict[str, object]]:
    """Run explicit preprocess -> classify -> extract -> summarize stages.

    The stage outputs are retained in each record so a notebook can show the
    actual transformations. The sentiment classifier and fact extractor are
    deterministic heuristics, not financial validation or an LLM judgment.
    """
    # Keep the evidence objects as the canonical full-text store. This derived
    # view intentionally contains bounded previews instead of copying every
    # source document several times into notebook state.
    processed: list[dict[str, object]] = []
    for record in records[:max(0, max_records)]:
        source_text = (record.content or "")[:max_source_chars]
        cleaned_text = preprocess_text(source_text)
        sentiment = classify_text(cleaned_text)
        facts = [
            fact[:max_preview_chars]
            for fact in extract_facts(cleaned_text)[:max_facts_per_record]
        ]
        summary = summarize_text(cleaned_text, max_length=max_preview_chars)
        cleaned_preview = cleaned_text[:max_preview_chars]
        processed.append({
            "text": cleaned_preview,
            "preprocessed_text": cleaned_preview,
            "sentiment": sentiment,
            "classification": sentiment,
            "extracted_facts": facts,
            "summary": summary,
            "source": record.citation(),
            "ticker": record.ticker,
            "url": record.url,
            "stage_outputs": {
                "preprocess": cleaned_preview,
                "classify": sentiment,
                "extract": facts,
                "summarize": summary,
            },
        })
    return processed
