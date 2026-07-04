from src.data.replay import interleave


def test_hits_target_ratio_and_consumes_all_primary():
    primary = list(range(1000))          # < 10000
    replay = list(range(10000, 20000))   # >= 10000
    mixed = interleave(primary, replay, replay_ratio=0.2, seed=0)
    n_replay = sum(1 for x in mixed if x >= 10000)
    assert abs(n_replay / len(mixed) - 0.2) < 0.02
    assert sum(1 for x in mixed if x < 1000) == 1000  # 모든 primary 유지


def test_zero_ratio_returns_primary_unchanged():
    primary = [1, 2, 3]
    assert interleave(primary, [9, 9], replay_ratio=0.0) == [1, 2, 3]


def test_deterministic_with_seed():
    p, r = list(range(50)), list(range(100, 200))
    assert interleave(p, r, 0.3, seed=7) == interleave(p, r, 0.3, seed=7)
