from scripts.score_naturalness import build_prompt, parse_score


def test_parse_score():
    assert parse_score("4") == 4
    assert parse_score("점수: 2점") == 2
    assert parse_score("모르겠음") is None


def test_build_prompt_includes_answer():
    assert "안녕하세요" in build_prompt("안녕하세요")
