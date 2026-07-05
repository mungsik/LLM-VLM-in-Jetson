"""SFT 대화데이터 빌드 CLI: HF 소스(streaming) → 어댑터 → 정제 → 가중 병합 → JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
from datasets import load_dataset

# 'src' 패키지 import 용 루트 경로 추가 (기존 스크립트와 동일)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distill.corpus import conversation_is_clean
from src.distill.sft_sources import alpaca_to_messages, messages_to_canonical

ADAPTERS = {"alpaca": alpaca_to_messages, "messages": messages_to_canonical}


def _iter_clean_convos(source: dict, max_rows: int | None, threshold: float):
    ds = load_dataset(
        source["path"], source.get("name"), split=source.get("split", "train"), streaming=True
    )
    adapt = ADAPTERS[source["type"]]
    fields = source.get("fields", {})
    for i, ex in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        msgs = adapt(ex, **fields)
        if msgs and conversation_is_clean(msgs, threshold=threshold):
            yield msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_rows = cfg.get("max_rows_per_source")
    threshold = cfg.get("clean_threshold", 0.05)

    total = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for s in cfg["sources"]:
            # repeat 위해 소스 1개분을 리스트로 유지(SFT 규모 수십만 대화 = 텍스트 수 GB,
            # 기존 MultiturnSFTDataset도 eager 적재 — 동일 기조).
            convos = list(_iter_clean_convos(s, max_rows, threshold))
            for _ in range(s.get("repeat", 1)):
                for msgs in convos:
                    w.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    total += 1
            print(f"{s['path']}: {len(convos)} clean convos x{s.get('repeat', 1)}")
    print(f"total lines -> {args.out}: {total}")


if __name__ == "__main__":
    main()
