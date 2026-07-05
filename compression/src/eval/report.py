"""단계별 평가 리포트 조립 + 경고 플래그."""
from __future__ import annotations


def build_stage_report(stage: str, metrics: dict) -> dict:
    flags: list[str] = []
    m = metrics

    if m.get("english_mixing_rate", 0.0) > 0.10:
        flags.append("english_mixing_high")

    if "ppl_ko" in m and "baseline_ppl_ko" in m and m["ppl_ko"] > m["baseline_ppl_ko"]:
        flags.append("korean_ppl_regressed")

    if "ppl_en" in m and "baseline_ppl_en" in m and m["ppl_en"] > m["baseline_ppl_en"] * 1.2:
        flags.append("english_ppl_regressed")

    return {"stage": stage, "metrics": m, "flags": flags}


def format_report_md(report: dict) -> str:
    lines = [f"## 평가 리포트 — {report['stage']}", "", "| 지표 | 값 |", "|---|---|"]
    for k, v in report["metrics"].items():
        lines.append(f"| {k} | {v} |")
    if report["flags"]:
        lines += ["", "**⚠️ 경고:** " + ", ".join(report["flags"])]
    return "\n".join(lines)
