"""Programmatic verifiers for instruction-following synthetic answers."""
from __future__ import annotations

import json
import re


def verify_list_count(answer: str, n: int) -> bool:
    """답이 '오직' 1..n 순번의 번호목록 n줄로만 이뤄져야 통과(앞뒤 산문·중복번호 불가)."""
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    if len(lines) != n:
        return False
    for idx, ln in enumerate(lines, 1):
        if not re.match(rf"^\s*{idx}\.\s+\S", ln):
            return False
    return True


def verify_json_keys(answer: str, keys: list[str]) -> bool:
    """답 전체가 JSON 객체여야 통과('JSON으로만'). 산문으로 감싼 JSON 은 불가."""
    try:
        obj = json.loads(answer.strip())
    except Exception:
        return False
    return isinstance(obj, dict) and sorted(obj.keys()) == sorted(keys)


def verify_forbidden(answer: str, words: list[str]) -> bool:
    return not any(w in answer for w in words)


def verify_ending(answer: str, ending: str) -> bool:
    return answer.strip().endswith(ending) and answer.strip().count(ending) >= 1 \
        and answer.strip()[-len(ending):] == ending
