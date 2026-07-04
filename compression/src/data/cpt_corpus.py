"""CPT 코퍼스 빌더: 텍스트 → 토큰화 → 패킹 → 영어 replay 혼합 → datasets.Dataset."""
from __future__ import annotations

from datasets import Dataset

from src.data.packing import pack_token_lists
from src.data.replay import interleave


def tokenize_texts(texts: list[str], tokenizer) -> list[list[int]]:
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def build_cpt_dataset(
    primary_texts: list[str],
    replay_texts: list[str],
    tokenizer,
    block_size: int,
    replay_ratio: float,
    seed: int = 0,
) -> Dataset:
    eos = tokenizer.eos_token_id
    primary_blocks = pack_token_lists(tokenize_texts(primary_texts, tokenizer), block_size, eos)
    replay_blocks = (
        pack_token_lists(tokenize_texts(replay_texts, tokenizer), block_size, eos)
        if replay_texts and replay_ratio > 0
        else []
    )
    blocks = (
        interleave(primary_blocks, replay_blocks, replay_ratio, seed)
        if replay_blocks
        else primary_blocks
    )
    return Dataset.from_dict({"input_ids": blocks, "labels": [list(b) for b in blocks]})
