from src.distill.chat_metrics import english_mixing_rate, idk_calibration, refused


def test_english_mixing_rate():
    answers = ["한국어로만 답합니다.", "I started by rephrasing the whole sentence now."]
    assert english_mixing_rate(answers) == 0.5


def test_refused():
    assert refused("정확히 알 수 없습니다.") is True
    assert refused("잘 모르겠습니다.") is True
    assert refused("답은 700원입니다.") is False
    # 오탐 방지(codex 지적): 약물명·정중표현은 거부가 아님
    assert refused("모르핀은 진통제입니다.") is False
    assert refused("잘 모르시면 도와드릴게요.") is False


def test_refused_catches_synth_idk_answers():
    # synth_idk.py 가 실제로 생성하는 거부 답변을 refused() 가 전부 잡아야 함(codex 지적).
    synth_refusals = [
        "그건 아직 일어나지 않은 일이라 확실하지 않습니다. 발표 후에 확인하실 수 있어요.",
        "정확히 확인할 수 없는 정보예요. 잘못된 정보를 드리지 않기 위해 모른다고 말씀드릴게요.",
        "그 내용은 지문에 없어 알 수 없습니다.",
    ]
    assert all(refused(a) for a in synth_refusals)


def test_idk_calibration():
    recs = [
        {"answerable": True, "refused": False},
        {"answerable": True, "refused": True},
        {"answerable": False, "refused": True},
        {"answerable": False, "refused": False},
    ]
    out = idk_calibration(recs)
    assert out["answerable_answered"] == 0.5
    assert out["unanswerable_refused"] == 0.5
