# 프루닝 스윕 + CPT 학습 Implementation Plan (Plan 4/5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phi-4-14B를 여러 프루닝률로 감량(스윕)해 8GB fit·품질을 견주어 최적 ratio를 고르고, 그 pruned 모델에 Plan 1 CPT 데이터로 full-FT 한국어 CPT(프루닝 복구 겸함)를 돌려, 각 단계를 Plan 3 하니스로 평가한다.

**Architecture:** 8GB fit 판정·최적 ratio 선택은 순수 로직(TDD). 프루닝 스윕은 기존 `run_depth_prune.py`를 ratio별로 호출하고 KMMLU/PPL로 평가하는 오케스트레이터(통합, 스모크). CPT는 Unsloth full-FT 학습 스크립트(통합, tiny 스모크). 실제 대형 학습은 별도 실행(수시간·GPU 비용).

**Tech Stack:** Python 3.10, 기존 `src/prune`·`src/common`, Unsloth + TRL, transformers, `src/eval`(Plan 3), lm-eval, pytest, uv.

## Global Constraints

- 하드웨어 타겟 = **Jetson Orin Nano 8GB 고정** → GGUF 모델 파일 ≲ **4.5GB**. 프루닝률·양자화는 이 fit을 맞추는 커플링 레버.
- **프루닝률은 스윕으로 결정**(고정 아님). 후보 25/35/45%.
- **CPT는 full-FT**(QLoRA는 새 언어 주입에 약함), Blackwell 96GB 단일. LR 낮게 + 영어 replay(데이터에 이미 포함) + 1 epoch → 망각 방지.
- CPT 순서 = **프루닝 → CPT**(복구+한국어 겸함).
- 작업 위치 = `compression/`. 커밋은 VM에서만. 광범위 `git add -A` 금지.
- **`uv sync` 금지**(vllm/deps prune) → 추가 dep은 `uv pip install`.
- 실대형 학습 전 반드시 **tiny 스모크**로 스크립트 검증(비용 낭비 방지).

## 재사용 (그대로)

- `scripts/run_depth_prune.py` — `--config <yaml> --out <dir>` depth 프루닝.
- `src/common/model_loader.py`, `src/prune/*` — 프루닝 파이프라인.
- `src/common/eval_kmmlu.py` `run_kmmlu`, `src/eval/ppl.py` `compute_ppl`(Plan 3).
- `configs/prune_phi4.yaml` — 프루닝 베이스 설정(ratio만 스윕).

## File Structure

- `compression/src/prune/fit.py` — 8GB fit 추정 + 최적 ratio 선택 (순수, TDD)
- `compression/tests/prune/test_fit.py`
- `compression/scripts/sweep_prune.py` — ratio 스윕 오케스트레이터 (통합)
- `compression/scripts/train_cpt.py` — Unsloth full-FT CPT (통합)
- `compression/configs/cpt_train.yaml` — CPT 학습 설정

---

### Task 1: fit 추정 + 최적 ratio 선택 (`fit.py`)

**Files:**
- Create: `compression/src/prune/fit.py`
- Test: `compression/tests/prune/test_fit.py`

**Interfaces:**
- Produces:
  - `estimate_gguf_gb(n_params: int, quant: str = "Q4_K_M") -> float` — 파라미터수×bits/8/1e9. `_BITS = {"Q4_K_M":4.5,"Q3_K_M":3.9,"Q2_K":3.35}`.
  - `fits_jetson(size_gb: float, budget_gb: float = 4.5) -> bool`.
  - `pick_best_ratio(results: list[dict], budget_gb=4.5, quant="Q4_K_M") -> dict | None` — `results`=[{ratio,n_params,kmmlu,...}]. fit 되는 후보 중 kmmlu 최고 선택. 없으면 None.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/prune/test_fit.py
from src.prune.fit import estimate_gguf_gb, fits_jetson, pick_best_ratio


def test_estimate_size_q4():
    # 7B @ Q4_K_M(4.5 bit) = 7e9*4.5/8/1e9 ≈ 3.94 GB
    assert abs(estimate_gguf_gb(7_000_000_000, "Q4_K_M") - 3.9375) < 1e-3


def test_fits_budget():
    assert fits_jetson(4.4, 4.5) is True
    assert fits_jetson(5.0, 4.5) is False


def test_pick_best_prefers_quality_among_fitting():
    results = [
        {"ratio": 0.25, "n_params": 10_600_000_000, "kmmlu": 0.40},  # ~5.96GB Q4 → fit X
        {"ratio": 0.35, "n_params": 8_000_000_000, "kmmlu": 0.37},   # ~4.5GB → 경계
        {"ratio": 0.45, "n_params": 7_000_000_000, "kmmlu": 0.34},   # ~3.94GB → fit O
    ]
    best = pick_best_ratio(results, budget_gb=4.5, quant="Q4_K_M")
    assert best["ratio"] == 0.45   # fit 되는 것 중 kmmlu 최고(0.35는 4.5 초과)


def test_pick_none_when_nothing_fits():
    results = [{"ratio": 0.1, "n_params": 14_700_000_000, "kmmlu": 0.41}]
    assert pick_best_ratio(results, budget_gb=4.5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/prune/test_fit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prune.fit'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/prune/fit.py
"""Jetson 8GB fit 추정 + 스윕 결과에서 최적 프루닝률 선택 (순수 로직)."""
from __future__ import annotations

_BITS = {"Q4_K_M": 4.5, "Q3_K_M": 3.9, "Q2_K": 3.35}


def estimate_gguf_gb(n_params: int, quant: str = "Q4_K_M") -> float:
    if quant not in _BITS:
        raise ValueError(f"unknown quant: {quant}")
    return n_params * _BITS[quant] / 8 / 1e9


def fits_jetson(size_gb: float, budget_gb: float = 4.5) -> bool:
    return size_gb <= budget_gb


def pick_best_ratio(results: list[dict], budget_gb: float = 4.5, quant: str = "Q4_K_M") -> dict | None:
    fitting = [
        r for r in results
        if fits_jetson(estimate_gguf_gb(r["n_params"], quant), budget_gb)
    ]
    if not fitting:
        return None
    return max(fitting, key=lambda r: r["kmmlu"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/prune/test_fit.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/prune/fit.py tests/prune/test_fit.py
git commit -m "feat(prune): 8GB fit 추정 + 최적 프루닝률 선택 로직"
```

---

### Task 2: 프루닝 스윕 오케스트레이터 (`sweep_prune.py`)

**Files:**
- Create: `compression/scripts/sweep_prune.py`

**Interfaces:**
- Consumes: `run_depth_prune.py`(subprocess), `run_kmmlu`(eval_kmmlu), `estimate_gguf_gb`/`pick_best_ratio`(Task 1).
- Produces: ratio별 pruned 모델(`--out-root/<ratio>/`) + 비교 리포트(JSON+마크다운) + 추천 ratio.

> 통합 태스크. 실모델 필요 → **tiny 스모크**로 스크립트 배선만 검증(작은 모델·1 ratio·KMMLU limit).

- [ ] **Step 1: 스크립트 작성**

```python
# compression/scripts/sweep_prune.py
"""프루닝률 스윕: ratio별 depth 프루닝 → KMMLU/파라미터수 → fit 비교 → 추천."""
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

from src.common.eval_kmmlu import run_kmmlu
from src.prune.fit import estimate_gguf_gb, pick_best_ratio

HERE = os.path.dirname(os.path.abspath(__file__))


def _count_params(model_dir: str) -> int:
    from transformers import AutoConfig, AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


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

    results = []
    for ratio in [float(r) for r in args.ratios.split(",")]:
        out_dir = os.path.join(args.out_root, f"ratio_{ratio}")
        cfg = copy.deepcopy(base)
        cfg["prune"]["ratio"] = ratio
        cfg["output"]["dir"] = out_dir
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
            yaml.safe_dump(cfg, tf, allow_unicode=True)
            tmp_cfg = tf.name

        subprocess.run(
            [sys.executable, os.path.join(HERE, "run_depth_prune.py"),
             "--config", tmp_cfg, "--out", out_dir],
            check=True,
        )
        n_params = _count_params(out_dir)
        kmmlu = run_kmmlu(out_dir, limit=args.kmmlu_limit)
        size_gb = estimate_gguf_gb(n_params, args.quant)
        results.append({"ratio": ratio, "n_params": n_params, "kmmlu": kmmlu,
                        "est_gguf_gb": round(size_gb, 2), "out": out_dir})
        print(f"[sweep] ratio={ratio} params={n_params/1e9:.2f}B "
              f"kmmlu={kmmlu:.4f} ~{size_gb:.2f}GB", flush=True)

    best = pick_best_ratio(results, quant=args.quant)
    report = {"quant": args.quant, "budget_gb": 4.5, "results": results, "recommended": best}
    with open(os.path.join(args.out_root, "sweep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[sweep] recommended:", best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: tiny 스모크 (VM, 작은 모델·1 ratio)**

작은 공개 모델로 배선만 확인(KMMLU는 limit로 축소):
```bash
uv run python scripts/sweep_prune.py \
  --base-config configs/prune_phi4.yaml \
  --ratios 0.25 --out-root /tmp/sweep_smoke --kmmlu-limit 20
```
(스모크 땐 `configs/prune_phi4.yaml`의 `model.name`을 작은 모델로 임시 교체 권장 — 예: `hf-internal-testing/tiny-random-LlamaForCausalLM` 계열은 KMMLU 무의미하므로, 실검증은 Phi-4-mini(microsoft/Phi-4-mini-instruct)로 1 ratio만.)
Expected: `ratio_0.25/` 생성, `sweep_report.json`에 results+recommended. 에러 없이 완주. 확인 후 `/tmp/sweep_smoke` 삭제, config model.name 원복.

- [ ] **Step 3: Commit** (VM에서)

```bash
git add scripts/sweep_prune.py
git commit -m "feat(prune): 프루닝률 스윕 오케스트레이터 (ratio별 프루닝+KMMLU+fit 비교)"
```

---

### Task 3: CPT full-FT 학습 스크립트 (`train_cpt.py`)

**Files:**
- Create: `compression/scripts/train_cpt.py`, `compression/configs/cpt_train.yaml`

**Interfaces:**
- Consumes: Plan 1 CPT 데이터셋(`load_from_disk`, `input_ids`/`labels`), pruned 모델 경로.
- Produces: CPT 완료 모델(저장) + 학습 로그.

> 통합 태스크. Unsloth 필요(`uv pip install unsloth`). 실대형 학습은 별도. **tiny 스모크**(작은 모델·few steps·`max_steps`)로 배선 검증.

- [ ] **Step 1: Unsloth 설치 확인 (VM)**

Run: `uv run python -c "import unsloth; print('unsloth ok')"`
없으면: `uv pip install unsloth` (uv sync 금지).

- [ ] **Step 2: 설정 파일 작성**

```yaml
# compression/configs/cpt_train.yaml
model_path: "artifacts/phi4-pruned/ratio_0.45"   # 스윕 추천 ratio 산출물
data_path: "artifacts/cpt_data"                   # Plan 1 build_cpt_data 산출물
output_dir: "artifacts/phi4-cpt"
max_seq_length: 4096
learning_rate: 3.0e-5
embedding_learning_rate: 5.0e-6      # 임베딩은 10배 작게(Unsloth CPT 팁)
num_train_epochs: 1
warmup_ratio: 0.05
lr_scheduler_type: "cosine"
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
logging_steps: 20
save_steps: 500
max_steps: -1                        # 스모크 땐 20 등으로 축소
```

- [ ] **Step 3: 학습 스크립트 작성**

```python
# compression/scripts/train_cpt.py
"""pruned Phi-4에 한국어 CPT(full-FT, Unsloth). Plan 1 데이터 소비, 프루닝 복구 겸함."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt_train.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from datasets import load_from_disk
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_path"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,             # auto(bf16)
        load_in_4bit=False,     # full-FT
        full_finetuning=True,
    )

    ds = load_from_disk(cfg["data_path"])   # input_ids/labels 이미 패킹됨

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=UnslothTrainingArguments(
            output_dir=cfg["output_dir"],
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=float(cfg["learning_rate"]),
            embedding_learning_rate=float(cfg["embedding_learning_rate"]),
            num_train_epochs=cfg["num_train_epochs"],
            max_steps=cfg["max_steps"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"[cpt] saved -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: tiny 스모크 (VM, few steps)**

`cpt_train.yaml`에서 `model_path`를 작은 모델(예: Phi-4-mini), `data_path`를 Plan 1 소량 산출물, `max_steps: 20`로 임시 설정 후:
```bash
uv run python scripts/train_cpt.py --config configs/cpt_train.yaml
```
Expected: 20 step 학습 진행 로그(loss 감소 추세) + `[cpt] saved` + output_dir 생성. 에러 없이 완주. 확인 후 산출물 삭제, config 원복.

- [ ] **Step 5: Commit** (VM에서)

```bash
git add scripts/train_cpt.py configs/cpt_train.yaml
git commit -m "feat(cpt): Unsloth full-FT CPT 학습 스크립트 + 설정"
```

---

## 본 실행 (계획 완료 후, 별도 GPU 세션 — 수시간·비용)

계획/스모크 검증 후 실제 파이프라인:
1. **CPT 데이터 빌드**(Plan 1): `uv run python scripts/build_cpt_data.py --config configs/cpt_data.yaml --out artifacts/cpt_data`
2. **프루닝 스윕**: `uv run python scripts/sweep_prune.py --base-config configs/prune_phi4.yaml --ratios 0.25,0.35,0.45 --out-root artifacts/prune_sweep` → `sweep_report.json`의 recommended 확인
3. **CPT 학습**: `cpt_train.yaml`의 `model_path`를 추천 ratio로 → `uv run python scripts/train_cpt.py`
4. **평가**(Plan 3): CPT 전/후 한국어 PPL·영어 회귀·Ko-IFEval 대조 → `build_stage_report`

각 단계 후 Blackwell **stop**(과금). 총 GPU 시간 = 프루닝 스윕(~1-1.5h) + CPT(데이터량 의존, 수시간).

## Self-Review

**1. Spec coverage:** 스펙 ①프루닝(스윕, fit 레버) ②CPT(full-FT, 복구+한국어, LR/embedding_lr/1epoch) 커버. 평가는 Plan 3 하니스 재사용. ✅
**2. Placeholder scan:** 순수 로직(fit/선택) 실제 코드+테스트. 학습/스윕은 통합이라 tiny 스모크로 검증(실대형은 별도 실행). Unsloth 하이퍼파라미터는 스펙 기반 초안값(실험 튜닝 대상 명시). ✅
**3. Type consistency:** `estimate_gguf_gb`/`pick_best_ratio`(Task1) → sweep_prune 소비. sweep_report.results 스키마 = pick_best_ratio 입력과 일치. CPT 데이터(input_ids/labels) = Plan 1 산출물과 일치. ✅

## 다음 플랜 (예정)
- **Plan 5**: SFT LoRA(Plan 2 데이터) + GGUF 양자화(llama.cpp) + Jetson Orin Nano 배포·실측(메모리 fit·tok/s).
