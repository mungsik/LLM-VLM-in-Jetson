"""GGUF 모델을 transformers gguf_file 로더로 KMMLU 평가 + 문항별 예측 저장.

transformers는 GGUF를 디퀀타이즈해 HF 모델로 로드 → 기존 hf 백엔드와 동일 경로,
양자화 손실은 반영됨. eval_samples.py와 같은 JSONL 포맷으로 덤프.

사용법:
    .venv/bin/python scripts/eval_gguf.py \
        --repo unsloth/Phi-4-mini-instruct-GGUF \
        --gguf-file Phi-4-mini-instruct-Q4_K_M.gguf \
        --name phi4mini-q4 --limit 20 --out gguf_samples.jsonl
"""
from __future__ import annotations

import argparse
import json

# --- gguf 버전 감지 패치 ---------------------------------------------------
# gguf 0.19.0 휠은 packages_distributions 매핑에 top-level("gguf")을 등록하지
# 않아 transformers가 버전을 'N/A'로 읽고 version.parse('N/A')에서 크래시한다.
# is_gguf_available()는 매번 _is_package_available()를 호출하므로 그 함수만
# 감싸 gguf일 때 실제 메타데이터 버전을 돌려주면 모든 호출 지점이 정상화된다.
import transformers.utils.import_utils as _iu  # noqa: E402

_orig_is_pkg = _iu._is_package_available


def _patched_is_pkg(pkg_name, return_version=False):
    if pkg_name == "gguf":
        import importlib.metadata as _m
        import importlib.util as _u
        if _u.find_spec("gguf") is not None:
            try:
                ver = _m.version("gguf")
            except Exception:
                ver = "0.0.0"
            return (True, ver) if return_version else True
    return _orig_is_pkg(pkg_name, return_version=return_version)


_iu._is_package_available = _patched_is_pkg
# --------------------------------------------------------------------------

from lm_eval import simple_evaluate  # noqa: E402

CHOICES = ["A", "B", "C", "D"]


def pred_letter(sample):
    resps = sample.get("filtered_resps") or sample.get("resps")
    try:
        ll = [r[0] if isinstance(r, (list, tuple)) else float(r) for r in resps]
        return CHOICES[int(max(range(len(ll)), key=lambda i: ll[i]))]
    except Exception:
        return "?"


def gold_letter(sample):
    t = sample.get("target")
    if isinstance(t, int) and 0 <= t < 4:
        return CHOICES[t]
    ans = (sample.get("doc") or {}).get("answer")
    if isinstance(ans, int) and 1 <= ans <= 4:
        return CHOICES[ans - 1]
    return str(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF GGUF repo id")
    ap.add_argument("--gguf-file", required=True, help="repo 내 .gguf 파일명")
    ap.add_argument("--name", required=True, help="결과 라벨")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--tokenizer",
        default="unsloth/Phi-4-mini-instruct",
        help="토크나이저 레포(비-GGUF). transformers의 GGUF→tokenizer 변환 버그 우회용.",
    )
    args = ap.parse_args()

    # 토크나이저는 비-GGUF 원본에서 로드(GGUF tokenizer 변환이 BPE 오류로 실패함).
    model_args = (
        f"pretrained={args.repo},"
        f"gguf_file={args.gguf_file},"
        f"tokenizer={args.tokenizer},"
        f"trust_remote_code=True"
    )
    print(f"[eval-gguf] {args.name}: {args.repo}::{args.gguf_file}", flush=True)
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["kmmlu"],
        limit=args.limit,
        device=args.device,
        batch_size="auto",
        log_samples=True,
    )
    acc = float(res["results"]["kmmlu"]["acc,none"])
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for task_name, samples in res["samples"].items():
            subject = task_name.replace("kmmlu_", "")
            for s in samples:
                doc = s.get("doc", {})
                g, p = gold_letter(s), pred_letter(s)
                f.write(json.dumps({
                    "model": args.name, "subject": subject,
                    "question": doc.get("question", ""),
                    "A": doc.get("A", ""), "B": doc.get("B", ""),
                    "C": doc.get("C", ""), "D": doc.get("D", ""),
                    "gold": g, "pred": p, "correct": g == p,
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"\n=== {args.name}: acc={acc*100:.2f}%, {n} samples → {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
