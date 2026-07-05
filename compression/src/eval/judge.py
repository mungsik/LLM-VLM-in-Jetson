"""GPT 페어와이즈 대화품질 judge (LogicKor/MT-Bench식, 위치편향 보정)."""
from __future__ import annotations

import re

_VERDICT = re.compile(r"\[\[([ABC])\]\]")

_PROMPT = """당신은 한국어 대화 품질을 평가하는 심사위원입니다.
아래 [질문]에 대한 두 답변 [답변 A]와 [답변 B]를 비교하세요.
평가 기준: 한국어의 자연스러움(번역투 감점), 질문 의도 충족, 정확성, 도움이 되는 정도.
간단한 근거를 쓴 뒤, 마지막 줄에 반드시 판정을 다음 중 하나로 출력하세요:
[[A]] (A가 더 나음) / [[B]] (B가 더 나음) / [[C]] (무승부).

[질문]
{q}

[답변 A]
{a}

[답변 B]
{b}
"""


def build_pairwise_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return _PROMPT.format(q=question, a=answer_a, b=answer_b)


def parse_pairwise_verdict(text: str) -> str:
    matches = _VERDICT.findall(text or "")
    if not matches:
        return "tie"
    last = matches[-1]
    return "tie" if last == "C" else last


def aggregate_two_orders(v_ab: str, v_ba: str) -> str:
    """v_ab: our=A,base=B 판정. v_ba: base=A,our=B 판정(순서뒤집음).
    our 우세 신호 = 1차 'A' + 2차 'B'. base 우세 = 1차 'B' + 2차 'A'."""
    our_first = v_ab == "A"
    our_second = v_ba == "B"
    base_first = v_ab == "B"
    base_second = v_ba == "A"
    if our_first and our_second:
        return "win"
    if base_first and base_second:
        return "loss"
    return "tie"


def judge_pairwise(client, question, our_answer, base_answer, model: str = "gpt-4o") -> str:
    """두 순서로 GPT 호출 후 합산. client = openai.OpenAI(). (통합, 스모크 검증)"""
    def _ask(a, b):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_pairwise_prompt(question, a, b)}],
            temperature=0,
        )
        return parse_pairwise_verdict(resp.choices[0].message.content)

    v_ab = _ask(our_answer, base_answer)   # A=our
    v_ba = _ask(base_answer, our_answer)   # A=base
    return aggregate_two_orders(v_ab, v_ba)
