"""KMMLU 평가를 문항별 예측까지 저장 (질문/정답/모델답).

lm-eval를 log_samples=True로 돌려 각 문항의
(질문, 보기 A~D, 정답, 모델 예측, 정오)를 JSONL로 덤프한다.

사용법:
    .venv/bin/python scripts/eval_samples.py --limit 20 --out samples.jsonl \
        --models original=microsoft/phi-4 \
                 depth=artifacts/phi4-pruned-depth \
                 act=artifacts/phi4-pruned-act \
                 mag=artifacts/phi4-pruned-smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from lm_eval import simple_evaluate

CHOICES = ["A", "B", "C", "D"]


def pred_letter(sample: dict) -> str:
    """filtered_resps의 선택지별 loglik에서 argmax → 예측 letter."""
    resps = sample.get("filtered_resps") or sample.get("resps")
    try:
        logliks = [r[0] if isinstance(r, (list, tuple)) else float(r) for r in resps]
        return CHOICES[int(max(range(len(logliks)), key=lambda i: logliks[i]))]
    except Exception:
        return "?"


def gold_letter(sample: dict) -> str:
    t = sample.get("target")
    if isinstance(t, int) and 0 <= t < len(CHOICES):
        return CHOICES[t]
    # doc의 answer(1~4) 폴백
    ans = (sample.get("doc") or {}).get("answer")
    if isinstance(ans, int) and 1 <= ans <= 4:
        return CHOICES[ans - 1]
    return str(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fout = open(args.out, "w", encoding="utf-8")
    summary = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"[eval] {name} ({path}) ...", flush=True)
        res = simple_evaluate(
            model="hf",
            model_args=f"pretrained={path},trust_remote_code=True",
            tasks=["kmmlu"],
            limit=args.limit,
            device=args.device,
            batch_size="auto",
            log_samples=True,
        )
        acc = float(res["results"]["kmmlu"]["acc,none"])
        summary.append((name, acc))
        n = 0
        for task_name, samples in res["samples"].items():
            subject = task_name.replace("kmmlu_", "")
            for s in samples:
                doc = s.get("doc", {})
                g, p = gold_letter(s), pred_letter(s)
                fout.write(json.dumps({
                    "model": name,
                    "subject": subject,
                    "question": doc.get("question", ""),
                    "A": doc.get("A", ""), "B": doc.get("B", ""),
                    "C": doc.get("C", ""), "D": doc.get("D", ""),
                    "gold": g, "pred": p, "correct": g == p,
                }, ensure_ascii=False) + "\n")
                n += 1
        print(f"  -> {name}: acc={acc:.4f}, {n} samples logged", flush=True)
    fout.close()

    print("\n=== summary ===")
    for name, acc in summary:
        print(f"{name:12s} {acc*100:.2f}%")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
