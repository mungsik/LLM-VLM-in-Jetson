from src.distill.sft_sources import korquad_to_messages


def test_korquad_parses_multiturn_with_system():
    ex = {"text": "<sys>문서 내용입니다.<usr>질문1<bot>답변1<usr>질문2<bot>답변2"}
    assert korquad_to_messages(ex) == [
        {"role": "system", "content": "문서 내용입니다."},
        {"role": "user", "content": "질문1"},
        {"role": "assistant", "content": "답변1"},
        {"role": "user", "content": "질문2"},
        {"role": "assistant", "content": "답변2"},
    ]


def test_korquad_needs_user_and_assistant():
    # 문서(sys)만 있고 대화 없음 → None
    assert korquad_to_messages({"text": "<sys>문서만 있음"}) is None
    assert korquad_to_messages({"text": ""}) is None


def test_korquad_custom_field():
    ex = {"conv": "<usr>안녕<bot>안녕하세요"}
    assert korquad_to_messages(ex, text_field="conv") == [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요"},
    ]
