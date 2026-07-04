"""primary 스트림에 replay를 목표 비율로 결정적 인터리브 (망각 방지용)."""
from __future__ import annotations

import random


def interleave(primary: list, replay: list, replay_ratio: float, seed: int = 0) -> list:
    if replay_ratio <= 0 or not replay:
        return list(primary)
    if not 0 < replay_ratio < 1:
        raise ValueError("replay_ratio must be in (0, 1)")
    rng = random.Random(seed)
    pool = list(replay)
    rng.shuffle(pool)
    factor = replay_ratio / (1 - replay_ratio)  # primary 1개당 삽입할 replay 기대수
    out: list = []
    n_primary = 0
    n_replay = 0
    ri = 0
    for item in primary:
        out.append(item)
        n_primary += 1
        while n_replay < factor * n_primary:
            out.append(pool[ri % len(pool)])
            ri += 1
            n_replay += 1
    return out
