from src.distill.ko_text import english_prose_ratio


def test_pure_korean_is_zero():
    assert english_prose_ratio("리스트는 수정 가능한 자료구조입니다.") == 0.0


def test_allowed_tokens_not_counted():
    # 코드/약어/URL/단발 영어단어 는 영어산문으로 안 침
    t = "파이썬에서 `append`를 쓰거나 CPU 정보를 https://x.io 에서 봅니다."
    assert english_prose_ratio(t) < 0.05


def test_korean_attached_english_allowed():
    # 'Python을','CPU와' 처럼 한글 붙은 영어, 단발 영어는 통과(정상 한국어)
    assert english_prose_ratio("파이썬에서 Python을 쓰고 CPU와 GPU를 확인하세요.") < 0.05


def test_lowercase_english_sentence_is_high():
    t = "리스트는 mutable. I started by rephrasing the first sentence to make it clearer."
    assert english_prose_ratio(t) > 0.3


def test_titlecase_english_sentence_is_high():
    # codex 가 지적한 구멍: Title Case 영어 문장도 잡아야 함
    t = "This Is A Full English Sentence With Every Word Capitalized."
    assert english_prose_ratio(t) > 0.5


def test_allcaps_english_sentence_is_high():
    assert english_prose_ratio("THIS IS A FULL ENGLISH SENTENCE HERE.") > 0.5


def test_empty_is_zero():
    assert english_prose_ratio("") == 0.0
