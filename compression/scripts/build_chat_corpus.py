"""Build the unified multi-turn chat corpus from filtered + synthesized sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.corpus import conversation_is_clean, is_multiturn, weighted_merge
from src.distill.korquad_chat import parse_korquad_chat


def load_smol(limit, threshold):
    ds = load_dataset("lemon-mint/smol-koreantalk", split="train", streaming=True)
    keep = []
    for row in ds:
        msgs = [{"role": m.get("role"), "content": m.get("content", "")} for m in row.get("messages", [])]
        msgs = [m for m in msgs if m["role"] in ("user", "assistant", "system")]
        if is_multiturn(msgs) and conversation_is_clean(msgs, threshold):
            keep.append(msgs)
        if limit and len(keep) >= limit:
            break
    return keep


def load_korquad():
    ds = load_dataset("heegyu/korquad-chat-v1", split="train")
    out = []
    for row in ds:
        p = parse_korquad_chat(row["text"])
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + p["messages"]
        if is_multiturn(p["messages"]):
            out.append(msgs)
    return out


def load_jsonl_rows(path, required=True):
    if not path or not Path(path).exists():
        if required:
            print(f"[warn] 합성 파일 없음: {path} — 먼저 synth_*.py 실행 필요(해당 구성 누락됨)", file=sys.stderr)
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


REFUSAL_KINDS = {"idk_unanswerable", "idk_grounded"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/chat_corpus.jsonl")
    ap.add_argument("--smol-limit", type=int, default=80000)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--korquad-repeat", type=int, default=5)   # 원어민 업샘플링
    ap.add_argument("--idk", default="data/distill/synth_idk.jsonl")
    ap.add_argument("--instructions", default="data/distill/synth_instructions.jsonl")
    ap.add_argument("--idk-max-frac", type=float, default=0.10)
    args = ap.parse_args()

    if not (0.0 <= args.idk_max_frac < 1.0):
        raise SystemExit(f"--idk-max-frac 는 [0,1) 범위여야 합니다: {args.idk_max_frac}")

    smol = load_smol(args.smol_limit, args.threshold)
    korquad = load_korquad()
    instr = [r["messages"] for r in load_jsonl_rows(args.instructions)]
    idk_rows = load_jsonl_rows(args.idk)
    # 거부(모른다)만 캡 대상. answerable_pair·clarify 는 일반 데이터라 캡 없이 base 로.
    refusals = [r["messages"] for r in idk_rows if r.get("meta", {}).get("kind") in REFUSAL_KINDS]
    idk_other = [r["messages"] for r in idk_rows if r.get("meta", {}).get("kind") not in REFUSAL_KINDS]

    base = weighted_merge([
        {"name": "smol", "convos": smol, "repeat": 1},
        {"name": "korquad", "convos": korquad, "repeat": args.korquad_repeat},
        {"name": "instructions", "convos": instr, "repeat": 1},
        {"name": "idk_other", "convos": idk_other, "repeat": 1},
    ])
    # 모른다(거부) 비율 상한: 최종 코퍼스의 idk_max_frac 이하
    cap = int(len(base) * args.idk_max_frac / (1 - args.idk_max_frac))
    refusals = refusals[:cap]
    allc = base + refusals

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for msgs in allc:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"saved {out} total={len(allc)} | smol={len(smol)} korquad={len(korquad)}x{args.korquad_repeat} "
          f"instr={len(instr)} idk_other={len(idk_other)} refusals={len(refusals)} (cap {cap})")
    print(f"번역:원어민 ≈ {len(smol)} : {len(korquad)*args.korquad_repeat}")


if __name__ == "__main__":
    main()
