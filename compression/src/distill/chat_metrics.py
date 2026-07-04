"""Deterministic chat-quality metrics: english mixing, IDK calibration."""
from __future__ import annotations

from src.distill.ko_text import english_prose_ratio

# 거부/불확실 동사형만 — 바레 "모르"는 "모르핀","잘 모르시면" 같은 오탐 유발(codex 지적).
_REFUSE_MARKERS = ("모르겠", "모릅니다", "모른다", "확실하지 않", "알 수 없", "확인할 수 없", "정보가 없")


def english_mixing_rate(answers, threshold: float = 0.05) -> float:
    if not answers:
        return 0.0
    bad = sum(1 for a in answers if english_prose_ratio(a) >= threshold)
    return bad / len(answers)


def refused(answer: str) -> bool:
    return any(m in answer for m in _REFUSE_MARKERS)


def idk_calibration(records) -> dict:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    return {
        "answerable_answered": (sum(1 for r in ans if not r["refused"]) / len(ans)) if ans else 0.0,
        "unanswerable_refused": (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0,
    }
