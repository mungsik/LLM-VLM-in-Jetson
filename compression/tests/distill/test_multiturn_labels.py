from src.distill.multiturn_labels import build_multiturn_labels


class FakeTok:
    """role 헤더 토큰을 정수로 흉내내는 최소 토크나이저. user=10, assistant=20, system=30."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]  # BOS
        for m in messages:
            head = {"user": 10, "assistant": 20, "system": 30}[m["role"]]
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]  # assistant 헤더만
        return ids


def test_two_assistant_turns_both_supervised():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
        {"role": "user", "content": "cd"},
        {"role": "assistant", "content": "z"},
    ]
    out = build_multiturn_labels(tok, msgs, max_length=4096)
    assert out["ok"] is True
    full = tok.apply_chat_template(msgs)
    p1 = tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
    e1 = tok.apply_chat_template(msgs[:2])
    p2 = tok.apply_chat_template(msgs[:3], add_generation_prompt=True)
    e2 = tok.apply_chat_template(msgs[:4])
    expected = [j for j in range(len(p1), len(e1))] + [j for j in range(len(p2), len(e2))]
    assert out["pos"] == expected
    assert all(full[p] != 10 for p in out["pos"])


def test_system_turn_and_user_ending_handled():
    tok = FakeTok()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},   # user 로 끝남(미완성)
    ]
    out = build_multiturn_labels(tok, msgs, max_length=4096)
    p1 = tok.apply_chat_template(msgs[:2], add_generation_prompt=True)
    e1 = tok.apply_chat_template(msgs[:3])
    assert out["pos"] == [j for j in range(len(p1), len(e1))]


def test_mid_turn_truncation_drops_partial_turn():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "xxxxxxxx"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "yyyyyyyy"},
    ]
    p1 = tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
    e1 = tok.apply_chat_template(msgs[:2])
    p2 = tok.apply_chat_template(msgs[:3], add_generation_prompt=True)
    # 2번째 assistant 턴 중간에서 자름 → 그 턴은 통째 제외, 1번째 턴만 라벨.
    out = build_multiturn_labels(tok, msgs, max_length=len(p2) + 3)
    assert out["pos"] == [j for j in range(len(p1), len(e1))]
    assert all(p < len(p2) for p in out["pos"])   # 2번째 턴 영역 토큰 전혀 없음


class BadTok:
    """add_generation_prompt 에 full 에 없는 토큰을 붙여 prefix 불일치를 유발."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]
        for m in messages:
            ids += [99] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [7, 7, 7]
        return ids


def test_prefix_mismatch_fails_closed():
    out = build_multiturn_labels(
        BadTok(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert out["ok"] is False and out["pos"] == []
