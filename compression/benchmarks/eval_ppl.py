"""한국어 perplexity 측정 (HF 또는 GGUF gguf_file 로더).

고정 한국어 코퍼스(kowiki 일부)를 seq_len 윈도우로 잘라 평균 NLL→PPL.
모든 모델에 동일 코퍼스·동일 토크나이저(같은 모델군)면 비교 가능.
GGUF는 transformers gguf_file 로더 사용(양자화 손실 반영).

사용법:
    .venv/bin/python scripts/eval_ppl.py --repo unsloth/Phi-4-mini-instruct-GGUF \
        --gguf-file Phi-4-mini-instruct-Q4_K_M.gguf --name phi4mini-Q4_K_M \
        --tokenizer unsloth/Phi-4-mini-instruct --out ppl.jsonl
"""
from __future__ import annotations

import argparse
import json

# gguf 버전감지 패치 (eval_gguf.py와 동일 사유)
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

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_korean_corpus(n_docs: int) -> str:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    parts = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        parts.append(row["text"])
    return "\n\n".join(parts)


@torch.no_grad()
def compute_ppl(model, tok, text: str, seq_len: int, stride: int, device: str) -> float:
    enc = tok(text, return_tensors="pt").input_ids.to(device)
    n = enc.size(1)
    nlls, count = [], 0
    prev = 0
    for begin in range(0, n, stride):
        end = min(begin + seq_len, n)
        trg_len = end - prev
        ids = enc[:, begin:end]
        tgt = ids.clone()
        tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        # out.loss = 평균 NLL(유효 토큰). 토큰수로 가중합산.
        valid = trg_len
        nlls.append(out.loss.float() * valid)
        count += valid
        prev = end
        if end == n:
            break
    return float(torch.exp(torch.stack(nlls).sum() / count))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gguf-file", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok_src = args.tokenizer or args.repo
    print(f"[ppl] {args.name}: load tokenizer {tok_src}", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_src)

    # Phi 계열은 bf16 네이티브. fp16은 긴 시퀀스 teacher-forcing에서 오버플로→NaN.
    kw = {"dtype": torch.bfloat16}
    if args.gguf_file:
        kw["gguf_file"] = args.gguf_file
    print(f"[ppl] load model {args.repo} {args.gguf_file or ''}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.repo, **kw).to(args.device).eval()

    text = load_korean_corpus(args.n_docs)
    ppl = compute_ppl(model, tok, text, args.seq_len, args.stride, args.device)
    print(f"\n=== {args.name}: PPL={ppl:.4f} (n_docs={args.n_docs}, seq={args.seq_len}) ===", flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": args.name, "ppl": ppl,
                            "n_docs": args.n_docs, "seq_len": args.seq_len}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
