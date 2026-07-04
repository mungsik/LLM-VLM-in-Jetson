from src.distill.ifeval_verify import (
    verify_list_count, verify_json_keys, verify_forbidden, verify_ending,
)


def test_list_count():
    good = "1. 가\n2. 나\n3. 다"
    assert verify_list_count(good, 3) is True
    assert verify_list_count(good, 5) is False
    # 중복 번호 거부
    assert verify_list_count("1. 가\n1. 나\n1. 다", 3) is False
    # 앞뒤 산문 섞이면 거부('목록으로만')
    assert verify_list_count("다음은 목록입니다.\n1. 가\n2. 나\n3. 다", 3) is False


def test_json_keys():
    assert verify_json_keys('{"summary": "x", "risks": "y"}', ["summary", "risks"]) is True
    assert verify_json_keys('{"summary": "x"}', ["summary", "risks"]) is False
    assert verify_json_keys("not json", ["a"]) is False
    # 산문으로 감싼 JSON 거부('JSON으로만')
    assert verify_json_keys('답: {"summary":"x","risks":"y"} 입니다', ["summary", "risks"]) is False


def test_forbidden():
    assert verify_forbidden("좋은 방법입니다.", ["최고", "완벽"]) is True
    assert verify_forbidden("이게 최고입니다.", ["최고", "완벽"]) is False


def test_ending():
    assert verify_ending("결론은 균형이 핵심이다.", "균형이 핵심이다.") is True
    assert verify_ending("균형이 핵심이다. 추가로...", "균형이 핵심이다.") is False
