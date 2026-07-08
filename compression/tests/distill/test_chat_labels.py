from src.distill.chat_labels import build_assistant_labels


class FakeTok:
    """role 토큰을 정수로 흉내내는 최소 토크나이저."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]  # BOS
        for m in messages:
            head = 10 if m["role"] == "user" else 20
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]  # assistant 헤더만 (응답 없음)
        return ids if tokenize else "".join(map(str, ids))


def test_assistant_span_only():
    tok = FakeTok()
    messages = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
    ]
    out = build_assistant_labels(tok, messages, max_length=2048)
    full = tok.apply_chat_template(messages, add_generation_prompt=False)
    prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
    assert out["input_ids"] == full
    # pos 는 prefix 길이 이후의 모든 인덱스
    assert out["pos"] == list(range(len(prefix), len(full)))
    assert 0 not in out["pos"]


def test_truncation_respected():
    tok = FakeTok()
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
    ]
    out = build_assistant_labels(tok, messages, max_length=32)
    assert len(out["input_ids"]) <= 32
    assert all(p < len(out["input_ids"]) for p in out["pos"])
