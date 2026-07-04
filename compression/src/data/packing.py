"""고정 길이 블록 패킹 — CPT용 (문서 사이 EOS 삽입, 잔여 조각 폐기)."""
from __future__ import annotations


def pack_token_lists(
    token_lists: list[list[int]], block_size: int, eos_id: int
) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    buffer: list[int] = []
    for doc in token_lists:
        buffer.extend(doc)
        buffer.append(eos_id)
    n_full = len(buffer) // block_size
    return [buffer[i * block_size : (i + 1) * block_size] for i in range(n_full)]
