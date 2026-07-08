from scripts.judge_pairwise import build_judge_prompt, parse_verdict


def test_parse_verdict():
    assert parse_verdict("최종: [[A]]") == "A"
    assert parse_verdict("판정 [[B]] 입니다") == "B"
    assert parse_verdict("[[C]]") == "tie"
    # 형식 못 지킨 출력은 invalid(별도 집계), tie 아님
    assert parse_verdict("모호") == "invalid"
    assert parse_verdict("nonsense without brackets") == "invalid"


def test_build_judge_prompt_contains_both():
    p = build_judge_prompt("질문?", "답변가", "답변나")
    assert "답변가" in p and "답변나" in p and "질문?" in p
