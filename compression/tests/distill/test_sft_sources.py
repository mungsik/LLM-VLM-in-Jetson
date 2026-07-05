from src.distill.sft_sources import alpaca_to_messages, messages_to_canonical


def test_alpaca_basic():
    ex = {"instruction": "질문", "input": "", "output": "답"}
    assert alpaca_to_messages(ex) == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답"},
    ]


def test_alpaca_appends_input():
    ex = {"instruction": "요약해", "input": "본문", "output": "요약"}
    assert alpaca_to_messages(ex)[0]["content"] == "요약해\n\n본문"


def test_alpaca_drops_when_missing():
    assert alpaca_to_messages({"instruction": "", "input": "", "output": "답"}) is None
    assert alpaca_to_messages({"instruction": "q", "input": "", "output": ""}) is None


def test_alpaca_custom_fields():
    ex = {"q": "안녕", "a": "응"}
    out = alpaca_to_messages(ex, instruction_field="q", input_field="none", output_field="a")
    assert out == [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "응"}]


def test_messages_extracts_role_content_dropping_extras():
    ex = {"messages": [
        {"role": "user", "content": "안녕", "content_en": "hi"},
        {"role": "assistant", "content": "응", "content_en": "yes"},
    ]}
    assert messages_to_canonical(ex) == [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "응"},
    ]


def test_messages_requires_user_and_assistant():
    assert messages_to_canonical({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert messages_to_canonical({"messages": []}) is None


def test_messages_skips_blank_and_invalid_roles():
    ex = {"messages": [
        {"role": "user", "content": "  "},          # blank → skip
        {"role": "user", "content": "질문"},
        {"role": "tool", "content": "x"},            # invalid role → skip
        {"role": "assistant", "content": "답"},
    ]}
    assert messages_to_canonical(ex) == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답"},
    ]
