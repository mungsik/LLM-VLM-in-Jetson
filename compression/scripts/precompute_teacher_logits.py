"""Precompute teacher top-k logits over teacher responses (offline KD targets).

각 응답을 학습과 동일한 chat_template로 렌더링 → teacher forward →
assistant 토큰 j 마다 teacher_logits[j-1].topk(k) 저장 (shift 규약).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.chat_labels import build_assistant_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="data/distill/scaleup_teacher.jsonl")
    ap.add_argument("--out-dir", default="data/distill/teacher_kd")
    ap.add_argument("--teacher", default="microsoft/phi-4")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = (out_dir / "manifest.jsonl").open("w", encoding="utf-8")

    done = 0
    skipped = 0
    with open(args.responses, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            lab = build_assistant_labels(tok, row["messages"], args.max_length)
            input_ids = torch.tensor(lab["input_ids"], dtype=torch.long)
            pos = [p for p in lab["pos"] if p - 1 >= 0]
            if not pos:
                # prefix 정렬 실패(prefix_ok=False) 또는 빈 assistant 구간 → skip
                skipped += 1
                continue
            with torch.inference_mode():
                logits = model(input_ids.unsqueeze(0).to(model.device)).logits[0]  # [L, V]
            pred_pos = torch.tensor([p - 1 for p in pos], device=logits.device)
            sel = logits.index_select(0, pred_pos)                  # [P, V]
            vals, idx = sel.topk(args.k, dim=-1)                    # [P, k]

            ex_id = row["id"]
            fpath = out_dir / f"{ex_id}.safetensors"
            save_file(
                {
                    "input_ids": input_ids.to(torch.int32),
                    "pos": torch.tensor(pos, dtype=torch.int32),
                    "topk_idx": idx.cpu().to(torch.int32),
                    "topk_logit": vals.cpu().to(torch.float16),
                },
                str(fpath),
            )
            manifest.write(json.dumps({"id": ex_id, "file": str(fpath), "n_pos": len(pos)}) + "\n")
            done += 1
            if done % 200 == 0:
                manifest.flush()
                print(f"precomputed {done}", flush=True)
    manifest.close()
    print(f"done rows={done} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
