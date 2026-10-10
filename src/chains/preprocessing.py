"""Small deterministic prompt-chain stages with an LLM-ready seam."""

from __future__ import annotations

from agent.state import EvidenceRecord


def process_evidence(records: list[EvidenceRecord]) -> list[dict[str, object]]:
    """Clean, classify, extract, and summarize records deterministically."""
    processed = []
    for record in records:
        text = " ".join(record.content.split())
        processed.append({
            "text": text,
            "summary": text[:500],
            "sentiment": _sentiment(text),
            "source": record.citation(),
            "ticker": record.ticker,
            "url": record.url,
        })
    return processed


def _sentiment(text: str) -> str:
    lowered = text.casefold()
    positive = sum(word in lowered for word in ("growth", "increase", "profit", "strong"))
    negative = sum(word in lowered for word in ("decline", "loss", "risk", "weak"))
    return "positive" if positive > negative else "negative" if negative > positive else "neutral"