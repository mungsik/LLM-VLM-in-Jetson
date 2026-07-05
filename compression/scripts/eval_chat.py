"""대화 평가 CLI: our vs base 페어와이즈(GPT judge) + 영어섞임/자연스러움 + 리포트."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distill.chat_metrics import english_mixing_rate
from src.eval.judge import judge_pairwise
from src.eval.report import build_stage_report, format_report_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True,
                    help='JSONL: {"question","our","base"} 라인들')
    ap.add_argument("--stage", default="sft")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rows = []
    with open(args.answers, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wins = losses = ties = 0
    for r in rows:
        v = judge_pairwise(client, r["question"], r["our"], r["base"], model=args.model)
        wins += v == "win"
        losses += v == "loss"
        ties += v == "tie"

    our_answers = [r["our"] for r in rows]
    metrics = {
        "n": len(rows),
        "win": wins, "loss": losses, "tie": ties,
        "win_rate": wins / len(rows) if rows else 0.0,
        "english_mixing_rate": english_mixing_rate(our_answers),
    }
    report = build_stage_report(args.stage, metrics)
    print(format_report_md(report))


if __name__ == "__main__":
    main()
