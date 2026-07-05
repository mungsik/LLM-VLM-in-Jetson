"""HF instruction/chat 데이터셋을 정규 messages 포맷으로 변환하는 어댑터."""
from __future__ import annotations

_VALID_ROLES = {"user", "assistant", "system"}


def alpaca_to_messages(
    ex: dict,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> list[dict] | None:
    """Alpaca형 {instruction, input, output} → 단일턴 messages."""
    instr = (ex.get(instruction_field) or "").strip()
    inp = (ex.get(input_field) or "").strip()
    out = (ex.get(output_field) or "").strip()
    if not instr or not out:
        return None
    user = f"{instr}\n\n{inp}" if inp else instr
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def messages_to_canonical(ex: dict, messages_field: str = "messages") -> list[dict] | None:
    """이미 messages형인 데이터에서 role/content만 추출(부가필드 제거)."""
    raw = ex.get(messages_field) or []
    msgs = []
    for m in raw:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in _VALID_ROLES and content:
            msgs.append({"role": role, "content": content})
    has_user = any(m["role"] == "user" for m in msgs)
    has_asst = any(m["role"] == "assistant" for m in msgs)
    if not (has_user and has_asst):
        return None
    return msgs
