"""Synthesize instruction/format-constrained pairs; keep only verifier-passing answers.

답 생성은 템플릿(결정적)으로 — 형식 제약은 템플릿으로 정확히 만족 가능. (LLM 불필요)
다양성: 주제 풀 + 제약 풀을 곱집합으로 확장.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.ifeval_verify import verify_list_count, verify_json_keys, verify_forbidden, verify_ending

TOPICS = [
    "제주 여행 준비물", "노트북 구매 점검사항", "재택근무 장점", "건강한 아침 루틴",
    "초보자를 위한 등산 팁", "전기차 고려사항", "독서 습관 만들기", "예산 절약 방법",
]

def gen_list(topic, n):
    # '목록으로만' 제약 → 앞뒤 산문 없이 번호목록만.
    items = [f"{i+1}. {topic} 항목{i+1}" for i in range(n)]
    return "\n".join(items), ("list_count", n)

def gen_json(topic):
    a = json.dumps({"summary": f"{topic} 요약", "risks": f"{topic} 위험"}, ensure_ascii=False)
    return a, ("json_keys", ["summary", "risks"])

def gen_forbidden(topic):
    return f"{topic}은 신중히 고르면 좋은 선택이 됩니다.", ("forbidden", ["최고", "완벽", "무조건"])

def gen_ending(topic):
    return f"{topic}에 대해 말하자면, 결국 균형이 핵심이다.", ("ending", "균형이 핵심이다.")

GENS = {"list_count": gen_list, "json_keys": gen_json, "forbidden": gen_forbidden, "ending": gen_ending}

def prompt_for(kind, topic, n):
    return {
        "list_count": f"{topic}을(를) 정확히 {n}개의 번호 목록으로만 답하세요.",
        "json_keys": f"{topic}을(를) JSON 객체로만 답하세요. 키는 summary, risks 두 개.",
        "forbidden": f"금지어 '최고','완벽','무조건'을 쓰지 말고 {topic}을(를) 한 문장으로 설명하세요.",
        "ending": f"{topic}에 대해 답하되 마지막 문장은 반드시 '균형이 핵심이다.'로 끝내세요.",
    }[kind]

def verify(kind, ans, arg):
    if kind == "list_count": return verify_list_count(ans, arg)
    if kind == "json_keys": return verify_json_keys(ans, arg)
    if kind == "forbidden": return verify_forbidden(ans, arg)
    if kind == "ending": return verify_ending(ans, arg)
    return False

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/synth_instructions.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--holdout", default="ending", help="평가용으로 빼는 제약유형")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kinds = [k for k in GENS if k != args.holdout]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with out.open("w", encoding="utf-8") as f:
        while kept < args.n:
            kind = rng.choice(kinds); topic = rng.choice(TOPICS); n = rng.choice([3, 4, 5])
            if kind == "list_count":
                ans, (vk, varg) = gen_list(topic, n)
            else:
                ans, (vk, varg) = GENS[kind](topic)
            if not verify(vk, ans, varg):
                dropped += 1; continue
            row = {"messages": [{"role": "user", "content": prompt_for(kind, topic, n)},
                                {"role": "assistant", "content": ans}],
                   "meta": {"constraint": kind}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); kept += 1
    print(f"saved {out} kept={kept} dropped={dropped} (holdout={args.holdout})")


if __name__ == "__main__":
    main()
