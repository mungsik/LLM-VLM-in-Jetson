from src.data.cpt_corpus import build_cpt_dataset, tokenize_texts


class FakeTokenizer:
    """공백 분할 → 정수 id. eos_token_id 제공. add_special_tokens 무시."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [(abs(hash(tok)) % 1000) + 1 for tok in text.split()]
        return {"input_ids": ids}


def test_tokenize_texts_shape():
    tok = FakeTokenizer()
    out = tokenize_texts(["a b c", "d e"], tok)
    assert [len(x) for x in out] == [3, 2]


def test_build_dataset_blocks_and_labels():
    tok = FakeTokenizer()
    primary = ["w " * 10 for _ in range(20)]   # 문서당 10토큰
    ds = build_cpt_dataset(
        primary_texts=primary, replay_texts=[], tokenizer=tok,
        block_size=8, replay_ratio=0.0, seed=0,
    )
    assert set(ds.column_names) == {"input_ids", "labels"}
    assert all(len(r) == 8 for r in ds["input_ids"])
    assert ds["labels"][0] == ds["input_ids"][0]   # CPT: labels == input_ids


def test_build_dataset_mixes_replay():
    tok = FakeTokenizer()
    primary = ["ko " * 8 for _ in range(50)]
    replay = ["en " * 8 for _ in range(50)]
    ds = build_cpt_dataset(primary, replay, tok, block_size=8, replay_ratio=0.2, seed=0)
    # replay 섞였으니 블록 수가 primary-only보다 많아야 함
    ds_only = build_cpt_dataset(primary, [], tok, block_size=8, replay_ratio=0.0)
    assert len(ds) > len(ds_only)
