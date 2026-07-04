from src.distill.korquad_chat import parse_korquad_chat


def test_parses_sys_usr_bot():
    text = "<sys>바그너는 1839년 파우스트를 읽었다.\n<usr>왜 끌렸나요?\n<bot>심경에 공감해서요.\n<usr>그래서요?\n<bot>교향곡을 구상했어요."
    out = parse_korquad_chat(text)
    assert out["system"].startswith("바그너는")
    assert out["messages"] == [
        {"role": "user", "content": "왜 끌렸나요?"},
        {"role": "assistant", "content": "심경에 공감해서요."},
        {"role": "user", "content": "그래서요?"},
        {"role": "assistant", "content": "교향곡을 구상했어요."},
    ]


def test_missing_sys_returns_empty_system():
    text = "<usr>안녕\n<bot>네 안녕하세요"
    out = parse_korquad_chat(text)
    assert out["system"] == ""
    assert out["messages"][0] == {"role": "user", "content": "안녕"}
