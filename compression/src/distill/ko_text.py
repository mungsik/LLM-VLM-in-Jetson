"""Estimate English-prose contamination via runs of consecutive English words.

영어 산문 = 한글이 섞이지 않은 '순수 라틴 단어'가 3개 이상 연속된 구간.
이렇게 하면 'Python', 'CPU', `append` 같은 단발 기술용어/한글붙은 영어는 통과시키되,
"I started by rephrasing the sentence" 같은 영어 문장(대소문자 무관)은 잡는다.
"""
from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_URL = re.compile(r"https?://\S+")
# 순수 라틴 단어: 라틴 글자로 시작, 라틴/문장부호만, 라틴 글자 2개 이상(고유 'I','a' 제외).
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’.,!?;:()\"-]*$")
_MIN_RUN = 3


def _is_latin_word(w: str) -> bool:
    return bool(_LATIN_WORD.match(w)) and len(re.findall(r"[A-Za-z]", w)) >= 2


def english_prose_ratio(text: str) -> float:
    if not text:
        return 0.0
    t = _CODE_BLOCK.sub(" ", text)
    t = _INLINE_CODE.sub(" ", t)
    t = _URL.sub(" ", t)
    nonspace = len(re.findall(r"\S", text)) or 1

    prose_letters = 0
    run: list[str] = []

    def flush() -> None:
        nonlocal prose_letters
        if len(run) >= _MIN_RUN:
            prose_letters += sum(len(re.findall(r"[A-Za-z]", w)) for w in run)

    for tok in t.split():
        if _is_latin_word(tok):
            run.append(tok)
        else:
            flush()
            run = []
    flush()
    return prose_letters / nonspace
