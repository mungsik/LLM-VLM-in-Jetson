"""고정 길이 블록 패킹 — CPT용 (문서 사이 EOS 삽입, 잔여 조각 폐기)."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def pack_token_stream(
    token_lists: Iterable[list[int]], block_size: int, eos_id: int
) -> Iterator[list[int]]:
    """문서 토큰들을 스트리밍 소비해 완전한 block_size 블록을 순차 yield.
    메모리 = O(block_size + 한 문서). 대용량 코퍼스(280GB급)용."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    buffer: list[int] = []
    for doc in token_lists:
        buffer.extend(doc)
        buffer.append(eos_id)
        while len(buffer) >= block_size:
            yield buffer[:block_size]
            del buffer[:block_size]


def pack_token_lists(
    token_lists: list[list[int]], block_size: int, eos_id: int
) -> list[list[int]]:
    """리스트 버전(소량/테스트용). 스트리밍 구현을 그대로 소진."""
    return list(pack_token_stream(token_lists, block_size, eos_id))
