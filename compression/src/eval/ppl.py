"""Perplexity 평가 — 집계식(순수) + 모델 기반 계산(통합)."""
from __future__ import annotations

import math


def perplexity_from_nll(total_nll: float, n_tokens: int) -> float:
    if n_tokens <= 0:
        return float("inf")
    return math.exp(total_nll / n_tokens)


def compute_ppl(model, tokenizer, texts: list[str], max_length: int = 2048) -> float:
    """causal LM NLL 누적 → perplexity. 실모델 필요(스모크 검증)."""
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = ids["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # HF loss = 평균 NLL(shift 반영). 토큰수 = n-1.
        n = input_ids.size(1) - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    return perplexity_from_nll(total_nll, total_tokens)
