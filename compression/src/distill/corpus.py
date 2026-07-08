"""Pure filtering / weighting / merge helpers for the chat corpus."""
from __future__ import annotations

from src.distill.ko_text import english_prose_ratio


def conversation_is_clean(messages, threshold: float = 0.05) -> bool:
    asst = [m["content"] for m in messages if m["role"] == "assistant"]
    return all(english_prose_ratio(c) < threshold for c in asst)


def is_multiturn(messages) -> bool:
    return sum(1 for m in messages if m["role"] == "assistant") >= 2


def weighted_merge(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        for _ in range(s["repeat"]):
            out.extend(s["convos"])
    return out
