"""Synthesize Korean uncertainty/refusal data with answerable counterparts.

규모 작게(수천). 비율은 build_chat_corpus 에서 상한 적용.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

UNANSWERABLE = [
    "2027년 노벨물리학상 수상자는 누구인가요?",
    "강원도 양구군에 있는 '청록별빛마을'의 정확한 주소를 알려주세요.",
    "내년 로또 1등 당첨 번호를 알려주세요.",
    "제 옆자리 동료의 이름이 뭔가요?",
]
IDK_ANSWERS = [
    "그건 아직 일어나지 않은 일이라 확실하지 않습니다. 발표 후에 확인하실 수 있어요.",
    "정확히 확인할 수 없는 정보예요. 잘못된 정보를 드리지 않기 위해 모른다고 말씀드릴게요.",
]
# (지문, 지문에 답이 있는 질문, 지문에 없는 질문, 답)
GROUNDED = [
    ("세종특별자치시는 2012년 7월 1일 출범했다.",
     "세종특별자치시는 언제 출범했나요?", "세종시의 초대 시장은 누구인가요?",
     "세종특별자치시는 2012년 7월 1일에 출범했습니다."),
]
CLARIFY = [
    ("그거 어떻게 해?", "어떤 것을 말씀하시는지 조금 더 알려주시겠어요? 맥락을 알면 정확히 도와드릴 수 있어요."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/synth_idk.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        r = rng.random()
        if r < 0.4:
            q = rng.choice(UNANSWERABLE); a = rng.choice(IDK_ANSWERS)
            rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "meta": {"kind": "idk_unanswerable"}})
        elif r < 0.8:
            sys, qa, qn, ans = rng.choice(GROUNDED)
            rows.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": qa}, {"role": "assistant", "content": ans}], "meta": {"kind": "answerable_pair"}})
            rows.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": qn}, {"role": "assistant", "content": "그 내용은 지문에 없어 알 수 없습니다."}], "meta": {"kind": "idk_grounded"}})
        else:
            q, a = rng.choice(CLARIFY)
            rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "meta": {"kind": "clarify"}})
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
