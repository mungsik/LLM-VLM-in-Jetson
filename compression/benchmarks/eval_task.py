"""임의 lm-eval 태스크를 HF/GGUF 모델로 평가하고 집계 지표를 jsonl로 저장.

HRM8K(생성, exact_match) 등 객관식 외 태스크용. gguf 버전감지 패치 포함.

사용법:
    .venv/bin/python scripts/eval_task.py --repo unsloth/Phi-4-mini-instruct-GGUF \
        --gguf-file Phi-4-mini-instruct-Q4_K_M.gguf --tokenizer unsloth/Phi-4-mini-instruct \
        --name phi4mini-Q4_K_M --tasks hrm8k_ksm --limit 50 --out hrm8k.jsonl
"""
from __future__ import annotations

import argparse
import json

import transformers.utils.import_utils as _iu  # noqa: E402
_orig = _iu._is_package_available
def _patched(p, return_version=False):
    if p == "gguf":
        import importlib.metadata as m, importlib.util as u
        if u.find_spec("gguf") is not None:
            try: v = m.version("gguf")
            except Exception: v = "0.0.0"
            return (True, v) if return_version else True
    return _orig(p, return_version=return_version)
_iu._is_package_available = _patched

from lm_eval import simple_evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gguf-file", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", required=True, help="콤마구분")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ma = f"pretrained={args.repo},trust_remote_code=True,dtype=float16"
    if args.gguf_file:
        ma += f",gguf_file={args.gguf_file}"
    if args.tokenizer:
        ma += f",tokenizer={args.tokenizer}"

    tasks = args.tasks.split(",")
    print(f"[task] {args.name}: {tasks} limit={args.limit}", flush=True)
    res = simple_evaluate(model="hf", model_args=ma, tasks=tasks,
                          limit=args.limit, device=args.device, batch_size="auto")

    with open(args.out, "a", encoding="utf-8") as f:
        for task, metrics in res["results"].items():
            clean = {k: v for k, v in metrics.items() if not k.startswith("alias")}
            print(f"  {args.name} / {task}: {clean}", flush=True)
            f.write(json.dumps({"name": args.name, "task": task, "metrics": clean},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
