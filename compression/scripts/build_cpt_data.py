"""CPT 데이터 빌드 CLI: HF 소스(streaming) 로드 → build_cpt_dataset → 디스크 저장."""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

# 'src' 패키지를 import 할 수 있도록 compression/ 루트를 경로에 추가 (기존 스크립트와 동일)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cpt_corpus import build_cpt_dataset


def _load_texts(sources: list[dict], max_docs: int | None) -> list[str]:
    """streaming으로 소스별 text 필드 수집 (대용량 코퍼스 대응)."""
    texts: list[str] = []
    for s in sources:
        ds = load_dataset(
            s["path"], s.get("name"), split=s.get("split", "train"), streaming=True
        )
        field = s["text_field"]
        for i, ex in enumerate(ds):
            if max_docs is not None and i >= max_docs:
                break
            val = ex.get(field)
            if val:
                texts.append(val)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt_data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    max_docs = cfg.get("max_docs_per_source")
    primary = _load_texts(cfg["primary_sources"], max_docs)
    replay = _load_texts(cfg.get("replay_sources", []), max_docs)

    ds = build_cpt_dataset(
        primary_texts=primary,
        replay_texts=replay,
        tokenizer=tok,
        block_size=cfg["block_size"],
        replay_ratio=cfg["replay_ratio"],
        seed=cfg["seed"],
    )
    ds.save_to_disk(args.out)
    print(f"saved {len(ds)} blocks (block_size={cfg['block_size']}) -> {args.out}")


if __name__ == "__main__":
    main()
