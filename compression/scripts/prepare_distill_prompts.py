"""Build a Korean sequence-KD prompt pool for the distillation pilot.

The output is JSONL with:
  {"id": "...", "source": "...", "prompt": "..."}

Datasets are intentionally sampled, not exhausted. Gated or unavailable datasets
are skipped so the same script can run in both smoke and full pilot modes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_MIX = [
    ("MarkrAI/KoCommercial-Dataset", 15_000),
    ("MarkrAI/KOpen-HQ-Hermes-2.5-60K", 10_000),
    ("nlpai-lab/kullm-v2", 10_000),
    ("squarelike/OpenOrca-gugugo-ko", 5_000),
]


def clean(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).replace("\r", "\n").split())


def first_role_text(items: Any, role_names: set[str]) -> str:
    if not isinstance(items, list):
        return ""
    for item in items:
        if not isinstance(item, dict):
            continue
        role = clean(item.get("role") or item.get("from") or item.get("speaker")).lower()
        if role not in role_names:
            continue
        text = clean(item.get("content") or item.get("value") or item.get("text"))
        if text:
            return text
    return ""


def extract_prompt(row: dict[str, Any]) -> str:
    instruction = clean(row.get("instruction"))
    inp = clean(row.get("input"))
    if instruction:
        return f"{instruction}\n\n{inp}" if inp else instruction

    for key in ("question", "prompt", "query"):
        value = clean(row.get(key))
        if value:
            return value

    text = first_role_text(row.get("messages"), {"user", "human"})
    if text:
        return text
    text = first_role_text(row.get("conversations"), {"user", "human"})
    if text:
        return text

    return ""


def iter_dataset_prompts(name: str, n: int, seed: int):
    try:
        ds = load_dataset(name, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=min(max(n * 4, 1_000), 50_000))
    except Exception as exc:
        print(f"[skip] {name}: {type(exc).__name__}: {exc}")
        return

    seen = 0
    try:
        for row in ds:
            prompt = extract_prompt(row)
            if len(prompt) < 12 or len(prompt) > 4_000:
                continue
            yield {"source": name, "prompt": prompt}
            seen += 1
            if seen >= n:
                break
    except Exception as exc:
        print(f"[partial-skip] {name}: {type(exc).__name__}: {exc}")
    print(f"[dataset] {name}: {seen}/{n}")


def synthetic_ifeval_prompts(n: int, seed: int):
    rng = random.Random(seed)
    topics = [
        "Jetson에서 한국어 음성 비서를 운영하는 방법",
        "초등학생에게 설명하는 양자화의 장단점",
        "한국어 문서 요약 시스템의 실패 원인",
        "소형 언어모델을 제품에 넣을 때의 체크리스트",
        "뉴스 기사에서 근거와 추론을 분리하는 방법",
        "고객 상담 챗봇의 안전한 답변 정책",
    ]
    formats = [
        "정확히 {k}개의 번호 목록으로 답하세요. 각 항목은 18자 이상 45자 이하로 작성하세요.",
        "JSON 객체로만 답하세요. 키는 summary, risks, next_steps 세 개만 사용하세요.",
        "두 문단으로 답하세요. 첫 문단은 장점, 둘째 문단은 한계만 다루세요.",
        "표 형식으로 답하세요. 열 이름은 항목, 이유, 우선순위입니다.",
        "금지어 '최고', '완벽', '무조건'을 쓰지 말고 답하세요.",
        "마지막 문장은 반드시 '검증 후 적용한다.'로 끝내세요.",
    ]
    for i in range(n):
        topic = rng.choice(topics)
        fmt = rng.choice(formats).format(k=rng.choice([3, 4, 5]))
        prompt = f"{topic}에 대해 한국어로 답하세요.\n\n제약: {fmt}"
        yield {"source": "synthetic-ko-ifeval-style", "prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/pilot_prompts.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", choices=["smoke", "pilot"], default="pilot")
    args = ap.parse_args()

    if args.scale == "smoke":
        mix = [(name, max(25, n // 500)) for name, n in DEFAULT_MIX]
        synthetic_n = 100
    else:
        mix = DEFAULT_MIX
        synthetic_n = 5_000

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for name, n in mix:
            for item in iter_dataset_prompts(name, n, args.seed + count):
                item["id"] = f"distill-{count:08d}"
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        for item in synthetic_ifeval_prompts(synthetic_n, args.seed):
            item["id"] = f"distill-{count:08d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"saved {out} rows={count}")


if __name__ == "__main__":
    main()
