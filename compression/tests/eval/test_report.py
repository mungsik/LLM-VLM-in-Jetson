from src.eval.report import build_stage_report, format_report_md


def test_flags_english_mixing_high():
    r = build_stage_report("sft", {"english_mixing_rate": 0.2})
    assert "english_mixing_high" in r["flags"]


def test_flags_english_ppl_regression():
    r = build_stage_report("cpt", {"ppl_en": 30.0, "baseline_ppl_en": 20.0})
    assert "english_ppl_regressed" in r["flags"]   # 30 > 20*1.2


def test_no_flags_when_healthy():
    r = build_stage_report("cpt", {
        "english_mixing_rate": 0.01,
        "ppl_ko": 8.0, "baseline_ppl_ko": 12.0,     # 개선
        "ppl_en": 21.0, "baseline_ppl_en": 20.0,    # 1.2배 이내
    })
    assert r["flags"] == []


def test_format_md_contains_stage_and_metric():
    md = format_report_md(build_stage_report("prune", {"kmmlu": 0.38}))
    assert "prune" in md and "kmmlu" in md
