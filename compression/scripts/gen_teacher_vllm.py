"""Generate teacher responses for sequence-level distillation (vLLM, high-throughput)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


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
    ap.add_argument("--out", default="data/distill/teacher_vllm.jsonl")
    ap.add_argument("--teacher", default="microsoft/phi-4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = load_prompts(Path(args.prompts), args.limit)
    convos = [[{"role": "user", "content": r["prompt"]}] for r in rows]

    llm = LLM(
        model=args.teacher,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    outs = llm.chat(convos, sp)
    dt = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            ans = o.outputs[0].text.strip()
            out_row = {
                "id": r["id"], "source": r["source"],
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": ans},
                ],
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    print(f"saved {out} rows={len(rows)} in {dt:.0f}s ({len(rows)/dt:.2f}/s)", flush=True)


if __name__ == "__main__":
    main()
