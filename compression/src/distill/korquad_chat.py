"""Parse heegyu/korquad-chat-v1 <sys>/<usr>/<bot> text into messages."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"<(sys|usr|bot)>(.*?)(?=<(?:sys|usr|bot)>|$)", re.S)
_ROLE = {"usr": "user", "bot": "assistant"}


def parse_korquad_chat(text: str) -> dict:
    system = ""
    messages = []
    for tag, body in _TOKEN.findall(text):
        body = body.strip()
        if tag == "sys":
            system = body
        else:
            messages.append({"role": _ROLE[tag], "content": body})
    return {"system": system, "messages": messages}
