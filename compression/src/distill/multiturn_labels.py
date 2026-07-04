"""Mark supervised token positions for every assistant turn (multi-turn)."""
from __future__ import annotations

from src.distill.chat_labels import _render_ids


def build_multiturn_labels(tokenizer, messages, max_length: int = 4096) -> dict:
    full = _render_ids(tokenizer, messages, add_generation_prompt=False)[:max_length]
    pos: list[int] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = _render_ids(tokenizer, messages[:i], add_generation_prompt=True)
        end = _render_ids(tokenizer, messages[: i + 1], add_generation_prompt=False)
        # 정렬 가드: prefix 가 full 의 실제 접두부가 아니면 fail closed(대화 통째 폐기).
        if full[: len(prefix)] != prefix[: min(len(prefix), len(full))]:
            return {"input_ids": full, "pos": [], "ok": False}
        # 이 assistant 턴이 truncation 으로 끝이 잘렸으면 부분 라벨 금지 → 턴 통째 제외.
        if len(end) > len(full):
            continue
        pos.extend(j for j in range(len(prefix), len(end)) if j >= 1)
    return {"input_ids": full, "pos": pos, "ok": True}
