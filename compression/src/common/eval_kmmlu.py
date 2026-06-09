"""KMMLU(한국어 MMLU) 평가 래퍼 — lm-eval-harness 기반."""
from __future__ import annotations

from lm_eval import simple_evaluate


def extract_accuracy(results: dict, task: str = "kmmlu") -> float:
    """lm-eval 결과 dict에서 정확도(acc)를 추출."""
    return float(results["results"][task]["acc,none"])


def run_kmmlu(model_path: str, limit: int | None = None, device: str = "cuda") -> float:
    """주어진 모델 경로/이름에 대해 KMMLU 정확도를 측정해 반환."""
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=["kmmlu"],
        limit=limit,
        device=device,
    )
    return extract_accuracy(results, task="kmmlu")
