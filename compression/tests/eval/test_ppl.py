import math

from src.eval.ppl import perplexity_from_nll


def test_perplexity_uniform():
    # 평균 NLL = 1.0 → ppl = e
    assert abs(perplexity_from_nll(10.0, 10) - math.e) < 1e-9


def test_perplexity_zero_nll_is_one():
    assert perplexity_from_nll(0.0, 5) == 1.0


def test_perplexity_no_tokens_is_inf():
    assert perplexity_from_nll(0.0, 0) == float("inf")
