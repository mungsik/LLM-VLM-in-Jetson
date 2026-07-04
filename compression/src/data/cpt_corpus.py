"""CPT 코퍼스 빌더: 텍스트 → 토큰화 → 패킹 → 영어 replay 혼합 → datasets.Dataset.

- build_cpt_dataset: 리스트 인메모리 버전(소량/단위테스트용).
- build_cpt_dataset_streaming: 대용량(280GB급) 스트리밍 버전. HF 소스를 streaming으로
  읽어 Dataset.from_generator로 Arrow에 점진 기록 → 피크 메모리 O(block).
  (from_generator는 gen_kwargs를 fingerprint 위해 pickle하므로, 제너레이터는 모듈레벨
   함수 + picklable한 소스 스펙으로 구성한다. 라이브 제너레이터를 넘기면 pickle 실패.)
"""
from __future__ import annotations

from datasets import Dataset, load_dataset

from src.data.packing import pack_token_lists, pack_token_stream
from src.data.replay import interleave, interleave_stream


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
    """리스트 버전(소량/테스트용, 전량 인메모리)."""
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


def _stream_texts(source_specs: list[dict], max_docs: int | None):
    """소스 스펙(picklable) → text 필드를 streaming yield. 전량 materialize 안 함."""
    for s in source_specs:
        ds = load_dataset(
            s["path"], s.get("name"), split=s.get("split", "train"), streaming=True
        )
        field = s["text_field"]
        for i, ex in enumerate(ds):
            if max_docs is not None and i >= max_docs:
                break
            val = ex.get(field)
            if val:
                yield val


def _cpt_example_gen(primary_specs, replay_blocks, tokenizer, block_size, replay_ratio, seed, max_docs):
    """모듈레벨 제너레이터(from_generator용). primary는 스트리밍, replay는 유한 풀."""
    eos = tokenizer.eos_token_id
    primary_tok = (
        tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in _stream_texts(primary_specs, max_docs)
    )
    primary_blocks = pack_token_stream(primary_tok, block_size, eos)
    stream = (
        interleave_stream(primary_blocks, replay_blocks, replay_ratio, seed)
        if replay_blocks
        else primary_blocks
    )
    for block in stream:
        yield {"input_ids": block, "labels": list(block)}


def build_cpt_dataset_streaming(
    primary_specs: list[dict],
    replay_specs: list[dict],
    tokenizer,
    block_size: int,
    replay_ratio: float,
    seed: int = 0,
    max_docs: int | None = None,
) -> Dataset:
    """대용량 스트리밍 빌더. primary_specs/replay_specs = [{path,name,split,text_field}, ...]."""
    eos = tokenizer.eos_token_id
    replay_blocks = (
        list(pack_token_stream(
            (tokenizer(t, add_special_tokens=False)["input_ids"]
             for t in _stream_texts(replay_specs, max_docs)),
            block_size, eos,
        ))
        if replay_specs and replay_ratio > 0
        else []
    )
    return Dataset.from_generator(
        _cpt_example_gen,
        gen_kwargs=dict(
            primary_specs=primary_specs,
            replay_blocks=replay_blocks,
            tokenizer=tokenizer,
            block_size=block_size,
            replay_ratio=replay_ratio,
            seed=seed,
            max_docs=max_docs,
        ),
    )
