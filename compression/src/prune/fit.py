"""Jetson 8GB fit 추정 + 스윕 결과에서 최적 프루닝률 선택 (순수 로직)."""
from __future__ import annotations

_BITS = {"Q4_K_M": 4.5, "Q3_K_M": 3.9, "Q2_K": 3.35}


def estimate_gguf_gb(n_params: int, quant: str = "Q4_K_M") -> float:
    if quant not in _BITS:
        raise ValueError(f"unknown quant: {quant}")
    return n_params * _BITS[quant] / 8 / 1e9


def fits_jetson(size_gb: float, budget_gb: float = 4.5) -> bool:
    return size_gb <= budget_gb


def pick_best_ratio(results: list[dict], budget_gb: float = 4.5, quant: str = "Q4_K_M") -> dict | None:
    fitting = [
        r for r in results
        if fits_jetson(estimate_gguf_gb(r["n_params"], quant), budget_gb)
    ]
    if not fitting:
        return None
    return max(fitting, key=lambda r: r["kmmlu"])
