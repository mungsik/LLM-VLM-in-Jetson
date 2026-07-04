"""Generate teacher responses for sequence-level distillation (BATCHED)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path: Path, limit: int | None):
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/distill/pilot_prompts.jsonl")
    ap.add_argument("--out", default="data/distill/teacher_batched.jsonl")
    ap.add_argument("--teacher", default="microsoft/phi-4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    ).eval()

    rows = load_prompts(Path(args.prompts), args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompt_texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in batch
            ]
            inputs = tok(prompt_texts, return_tensors="pt", padding=True).to(model.device)
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
            in_len = inputs["input_ids"].shape[1]  # left-padded => same for all rows
            new_tokens = gen[:, in_len:]
            answers = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for r, ans in zip(batch, answers):
                out_row = {
                    "id": r["id"], "source": r["source"],
                    "messages": [
                        {"role": "user", "content": r["prompt"]},
                        {"role": "assistant", "content": ans.strip()},
                    ],
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            done += len(batch)
            rate = done / (time.time() - t0)
            print(f"generated {done}/{len(rows)}  ({rate:.2f}/s)", flush=True)
    print(f"saved {out} rows={done} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
