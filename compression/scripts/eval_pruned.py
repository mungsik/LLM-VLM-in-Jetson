"""pruned 모델 1개 평가 워커: 파라미터수 + KMMLU → stdout JSON.
서브프로세스로 실행돼 종료 시 GPU 메모리 완전 해제(스윕 OOM 방지)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.eval_kmmlu import run_kmmlu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--kmmlu-limit", type=int, default=None)
    args = ap.parse_args()

    # 파라미터수는 CPU 로드로 카운트(GPU 안 씀)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto")
    n_params = sum(p.numel() for p in m.parameters())
    del m

    kmmlu = run_kmmlu(args.model_dir, limit=args.kmmlu_limit)
    print("RESULT " + json.dumps({"n_params": n_params, "kmmlu": kmmlu}))


if __name__ == "__main__":
    main()
