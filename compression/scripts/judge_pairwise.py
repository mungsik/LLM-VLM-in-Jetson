"""Pairwise LLM-judge: compare two assistants' answers (position-bias handled by caller)."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

JUDGE = (
    "두 답변 중 한국어로 더 자연스럽고 정확하며 지시를 잘 따른 쪽을 고르세요.\n"
    "질문: {q}\n\n[A]\n{a}\n\n[B]\n{b}\n\n"
    "더 나은 쪽을 [[A]] 또는 [[B]] 로, 비등하면 [[C]] 로만 표기하세요."
)


def build_judge_prompt(question, ans_a, ans_b) -> str:
    return JUDGE.format(q=question, a=ans_a, b=ans_b)


def parse_verdict(text: str) -> str:
    # 형식 못 지킨 출력은 "invalid"로 분리(codex 지적) — tie 로 묻으면 심판 실패가 가려짐.
    m = re.search(r"\[\[([ABC])\]\]", text)
    if not m:
        return "invalid"
    return {"A": "A", "B": "B", "C": "tie"}[m.group(1)]


def judge(url, model, question, ans_a, ans_b) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": build_judge_prompt(question, ans_a, ans_b)}],
                       "temperature": 0, "max_tokens": 8}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return parse_verdict(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="jsonl: {question, ans_a, ans_b}")
    ap.add_argument("--url", default="http://localhost:8002/v1/chat/completions")
    ap.add_argument("--model", default="judge")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pairs)]
    wins = {"A": 0, "B": 0, "tie": 0, "invalid": 0}
    _swap = {"A": "B", "B": "A", "tie": "tie", "invalid": "invalid"}
    for r in rows:
        # 위치편향 완화: A/B 스왑 2회 후 합산. invalid 는 별도 집계(tie 로 안 묻음).
        v1 = judge(args.url, args.model, r["question"], r["ans_a"], r["ans_b"])
        v2 = judge(args.url, args.model, r["question"], r["ans_b"], r["ans_a"])
        for v in (v1, _swap[v2]):
            wins[v] += 1
    print(f"A={wins['A']} B={wins['B']} tie={wins['tie']} invalid={wins['invalid']}")


if __name__ == "__main__":
    main()
