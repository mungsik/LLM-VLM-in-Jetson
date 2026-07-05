from src.eval.judge import (
    aggregate_two_orders,
    build_pairwise_prompt,
    parse_pairwise_verdict,
)


def test_prompt_contains_question_and_both_answers():
    p = build_pairwise_prompt("질문?", "답A", "답B")
    assert "질문?" in p and "답A" in p and "답B" in p
    assert "[[A]]" in p and "[[B]]" in p and "[[C]]" in p  # 출력 형식 명시


def test_parse_last_verdict():
    assert parse_pairwise_verdict("설명...\n최종 판정: [[A]]") == "A"
    assert parse_pairwise_verdict("[[A]] 였다가 정정 [[B]]") == "B"   # 마지막 우선
    assert parse_pairwise_verdict("[[C]]") == "tie"
    assert parse_pairwise_verdict("판정 불가") == "tie"                # 없으면 tie


def test_aggregate_consistent_win():
    # 1차: A=our 우세(A). 2차(순서뒤집음): B=our 우세(B). → 일관 win
    assert aggregate_two_orders("A", "B") == "win"


def test_aggregate_consistent_loss():
    # 1차: B=base 우세(B). 2차: A=base 우세(A). → 일관 loss
    assert aggregate_two_orders("B", "A") == "loss"


def test_aggregate_inconsistent_is_tie():
    assert aggregate_two_orders("A", "A") == "tie"      # 위치편향(항상 A) → tie
    assert aggregate_two_orders("tie", "tie") == "tie"  # 실제 파이프라인이 주는 값(C→"tie")
    # 한쪽만 결정적이고 다른쪽 tie → 보수적으로 tie
    assert aggregate_two_orders("A", "tie") == "tie"
    assert aggregate_two_orders("tie", "B") == "tie"
    assert aggregate_two_orders("B", "tie") == "tie"
