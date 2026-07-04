"""LLM-judge naturalness scoring for sampled conversations (translationese detection)."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

JUDGE_PROMPT = (
    "다음 한국어 답변이 얼마나 자연스러운 원어민 한국어인지 1~5로 평가하세요"
    "(1=어색한 번역투, 5=완전 자연스러움). 숫자만 답하세요.\n\n답변:\n{ans}"
)


def build_prompt(answer: str) -> str:
    return JUDGE_PROMPT.format(ans=answer)


def parse_score(text: str) -> int | None:
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None


def judge(url: str, model: str, answer: str) -> int | None:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": build_prompt(answer)}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return parse_score(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--model", default="judge")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.corpus)][: args.sample]
    scores = []
    for r in rows:
        asst = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        if not asst:
            continue
        # 첫 턴 + 마지막 턴 둘 다 채점(후반 번역투 턴이 안 보이는 문제 방지).
        for ans in {asst[0], asst[-1]}:
            s = judge(args.url, args.model, ans)
            if s is not None:
                scores.append(s)
    if scores:
        print(f"n={len(scores)} mean_naturalness={sum(scores)/len(scores):.2f} "
              f"low(<=2)={sum(1 for s in scores if s<=2)/len(scores)*100:.1f}%")


if __name__ == "__main__":
    main()
