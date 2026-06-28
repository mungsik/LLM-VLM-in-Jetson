"""Render chat and mark assistant-only supervised positions."""
from __future__ import annotations


def build_assistant_labels(tokenizer, messages, max_length: int = 2048) -> dict:
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full = list(full)[:max_length]
    start = min(len(prefix), len(full))
    pos = [j for j in range(start, len(full)) if j >= 1]
    return {"input_ids": full, "pos": pos}
