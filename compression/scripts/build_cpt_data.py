"""CPT 데이터 빌드 CLI: HF 소스(streaming) → build_cpt_dataset_streaming → 디스크 저장.

primary는 전량 materialize 없이 streaming으로 소비하고, Dataset.from_generator가
Arrow에 점진 기록하므로 대용량(280GB급) 코퍼스도 피크 메모리 O(block)로 처리한다.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from transformers import AutoTokenizer

# 'src' 패키지를 import 할 수 있도록 compression/ 루트를 경로에 추가 (기존 스크립트와 동일)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cpt_corpus import build_cpt_dataset_streaming


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt_data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    ds = build_cpt_dataset_streaming(
        primary_specs=cfg["primary_sources"],
        replay_specs=cfg.get("replay_sources", []),
        tokenizer=tok,
        block_size=cfg["block_size"],
        replay_ratio=cfg["replay_ratio"],
        seed=cfg["seed"],
        max_docs=cfg.get("max_docs_per_source"),
    )
    ds.save_to_disk(args.out)
    print(f"saved {len(ds)} blocks (block_size={cfg['block_size']}) -> {args.out}")


if __name__ == "__main__":
    main()
