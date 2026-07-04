"""Generate teacher responses for sequence-level distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/distill/pilot_prompts.jsonl")
    ap.add_argument("--out", default="data/distill/pilot_teacher.jsonl")
    ap.add_argument("--teacher", default="microsoft/phi-4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out.open("w", encoding="utf-8") as f:
        for row in load_prompts(Path(args.prompts), args.limit):
            messages = [{"role": "user", "content": row["prompt"]}]
            prompt_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(prompt_text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=0.9,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            new_tokens = gen[0, inputs["input_ids"].shape[-1] :]
            answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
            out_row = {
                "id": row["id"],
                "source": row["source"],
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": answer},
                ],
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"generated {done}", flush=True)
    print(f"saved {out} rows={done}")


if __name__ == "__main__":
    main()
