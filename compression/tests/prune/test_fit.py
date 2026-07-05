from src.prune.fit import estimate_gguf_gb, fits_jetson, pick_best_ratio


def test_estimate_size_q4():
    # 7B @ Q4_K_M(4.5 bit) = 7e9*4.5/8/1e9 ≈ 3.94 GB
    assert abs(estimate_gguf_gb(7_000_000_000, "Q4_K_M") - 3.9375) < 1e-3


def test_fits_budget():
    assert fits_jetson(4.4, 4.5) is True
    assert fits_jetson(5.0, 4.5) is False


def test_pick_best_prefers_quality_among_fitting():
    results = [
        {"ratio": 0.25, "n_params": 10_600_000_000, "kmmlu": 0.40},  # ~5.96GB Q4 → fit X
        {"ratio": 0.35, "n_params": 8_000_000_000, "kmmlu": 0.37},   # =4.5GB(경계 포함) → fit O
        {"ratio": 0.45, "n_params": 7_000_000_000, "kmmlu": 0.34},   # ~3.94GB → fit O
    ]
    best = pick_best_ratio(results, budget_gb=4.5, quant="Q4_K_M")
    # fit 되는 {0.35, 0.45} 중 kmmlu 최고(0.37) → 0.35. (0.25는 fit X로 제외)
    assert best["ratio"] == 0.35


def test_pick_none_when_nothing_fits():
    results = [{"ratio": 0.1, "n_params": 14_700_000_000, "kmmlu": 0.41}]
    assert pick_best_ratio(results, budget_gb=4.5) is None
