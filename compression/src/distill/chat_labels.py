"""Render chat and mark assistant-only supervised positions."""
from __future__ import annotations


def _render_ids(tokenizer, messages, add_generation_prompt: bool) -> list:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # transformers 5.x: tokenize=True는 list[int]가 아니라 BatchEncoding을 반환.
    if not isinstance(out, (list, tuple)):
        out = out["input_ids"]
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):   # 배치 차원 제거
        out = list(out[0])
    return out


def build_assistant_labels(tokenizer, messages, max_length: int = 2048) -> dict:
    full = _render_ids(tokenizer, messages, add_generation_prompt=False)
    prefix = _render_ids(tokenizer, messages[:-1], add_generation_prompt=True)
    full = full[:max_length]
    start = min(len(prefix), len(full))
    pos = [j for j in range(start, len(full)) if j >= 1]
    return {"input_ids": full, "pos": pos}
