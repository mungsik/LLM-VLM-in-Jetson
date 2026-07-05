"""프루닝률 스윕: ratio별 depth 프루닝 → KMMLU/파라미터수 → fit 비교 → 추천.

모델 로딩(프루닝/평가)은 전부 서브프로세스로 격리 → 각 ratio 종료 시 GPU 메모리
완전 해제(메인 프로세스는 모델을 들지 않음, 스윕 OOM 방지)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prune.fit import estimate_gguf_gb, pick_best_ratio

HERE = os.path.dirname(os.path.abspath(__file__))


def _eval_pruned(model_dir: str, kmmlu_limit: int | None) -> dict:
    """서브프로세스로 파라미터수+KMMLU 측정 → dict."""
    cmd = [sys.executable, os.path.join(HERE, "eval_pruned.py"), "--model-dir", model_dir]
    if kmmlu_limit is not None:
        cmd += ["--kmmlu-limit", str(kmmlu_limit)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"eval_pruned no RESULT line for {model_dir}:\n{out.stdout[-500:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/prune_phi4.yaml")
    ap.add_argument("--ratios", default="0.25,0.35,0.45")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--kmmlu-limit", type=int, default=None)
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()

    with open(args.base_config, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    os.makedirs(args.out_root, exist_ok=True)
    results = []
    for ratio in [float(r) for r in args.ratios.split(",")]:
        out_dir = os.path.join(args.out_root, f"ratio_{ratio}")
        cfg = copy.deepcopy(base)
        cfg["prune"]["ratio"] = ratio
        cfg["output"]["dir"] = out_dir
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
            yaml.safe_dump(cfg, tf, allow_unicode=True)
            tmp_cfg = tf.name

        # 1) 프루닝(서브프로세스)
        subprocess.run(
            [sys.executable, os.path.join(HERE, "run_depth_prune.py"),
             "--config", tmp_cfg, "--out", out_dir],
            check=True,
        )
        # 2) 평가(서브프로세스, 종료 시 GPU 해제)
        ev = _eval_pruned(out_dir, args.kmmlu_limit)
        size_gb = estimate_gguf_gb(ev["n_params"], args.quant)
        results.append({"ratio": ratio, "n_params": ev["n_params"], "kmmlu": ev["kmmlu"],
                        "est_gguf_gb": round(size_gb, 2), "out": out_dir})
        print(f"[sweep] ratio={ratio} params={ev['n_params']/1e9:.2f}B "
              f"kmmlu={ev['kmmlu']:.4f} ~{size_gb:.2f}GB", flush=True)

    best = pick_best_ratio(results, quant=args.quant)
    report = {"quant": args.quant, "budget_gb": 4.5, "results": results, "recommended": best}
    with open(os.path.join(args.out_root, "sweep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[sweep] recommended:", best, flush=True)


if __name__ == "__main__":
    main()
