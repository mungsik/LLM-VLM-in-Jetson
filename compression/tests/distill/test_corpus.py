from src.distill.corpus import conversation_is_clean, is_multiturn, weighted_merge


def _conv(*pairs):
    out = []
    for u, a in pairs:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def test_clean_passes_dirty_fails():
    clean = _conv(("질문", "한국어로만 답합니다."))
    dirty = _conv(("질문", "I started by rephrasing the sentence to make it clearer."))
    assert conversation_is_clean(clean) is True
    assert conversation_is_clean(dirty) is False


def test_is_multiturn():
    assert is_multiturn(_conv(("a", "b"))) is False           # assistant 1개
    assert is_multiturn(_conv(("a", "b"), ("c", "d"))) is True  # assistant 2개


def test_weighted_merge_upsamples():
    native = _conv(("원", "어민"))
    trans = _conv(("번", "역"))
    merged = weighted_merge([
        {"name": "native", "convos": [native], "repeat": 3},
        {"name": "trans", "convos": [trans], "repeat": 1},
    ])
    assert len(merged) == 4          # 3 + 1
    assert merged.count(native) == 3
