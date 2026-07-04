"""primary 스트림에 replay를 목표 비율로 결정적 인터리브 (망각 방지용)."""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence


def interleave_stream(
    primary: Iterable, replay_pool: Sequence, replay_ratio: float, seed: int = 0
) -> Iterator:
    """primary(스트리밍 iterable)에 replay_pool(유한 시퀀스, 순환)을 목표 비율로 삽입.
    primary 전부 소비, 메모리 = O(replay_pool). primary는 280GB급이어도 됨."""
    if replay_ratio <= 0 or not replay_pool:
        yield from primary
        return
    if not 0 < replay_ratio < 1:
        raise ValueError("replay_ratio must be in (0, 1)")
    rng = random.Random(seed)
    pool = list(replay_pool)
    rng.shuffle(pool)
    factor = replay_ratio / (1 - replay_ratio)  # primary 1개당 삽입할 replay 기대수
    n_primary = 0
    n_replay = 0
    ri = 0
    for item in primary:
        yield item
        n_primary += 1
        while n_replay < factor * n_primary:
            yield pool[ri % len(pool)]
            ri += 1
            n_replay += 1


def interleave(primary: list, replay: list, replay_ratio: float, seed: int = 0) -> list:
    """리스트 버전(소량/테스트용). 스트리밍 구현을 그대로 소진."""
    return list(interleave_stream(primary, replay, replay_ratio, seed))
