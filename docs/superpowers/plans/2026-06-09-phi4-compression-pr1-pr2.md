# Phi-4 경량화 PR1+PR2 (Scaffold + Minitron 구조적 프루닝) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phi-4 경량화 파이프라인의 시작점 — 공통 모듈(모델 로딩·파라미터 통계·KMMLU 평가 래퍼)과 Minitron식 활성값 기반 구조적 width 프루닝을 구현하고 PR로 공유한다.

**Architecture:** `compression/` 패키지에 `common/`(공통 유틸)과 `prune/`(프루닝)을 분리. 프루닝은 torch-pruning의 DependencyGraph 결합구조 그룹핑 위에 활성값 기반 중요도를 얹는다(Minitron 방식). 단위 테스트는 로컬 Mac에서 tiny GQA Llama 프록시 모델로 검증하고, 실제 Phi-4(14B) 실행은 GPU 서버에서 integration 마커로 분리한다.

**Tech Stack:** Python 3.10, PyTorch(CPU 로컬/CUDA 서버), HuggingFace transformers·datasets, torch-pruning, lm-eval-harness, pyyaml, pytest

---

## File Structure

PR1 (scaffold + common + eval 골격):
- `compression/requirements.txt` — 의존성
- `compression/pytest.ini` — 테스트 마커(`integration`, `gpu`) 등록
- `compression/README.md` — 개요·셋업
- `compression/configs/prune_phi4.yaml` — 프루닝 설정
- `compression/src/__init__.py`, `compression/src/common/__init__.py`, `compression/src/prune/__init__.py`
- `compression/src/common/param_stats.py` — 파라미터 카운트·메모리 추정
- `compression/src/common/model_loader.py` — HF causal LM 로딩
- `compression/src/common/eval_kmmlu.py` — lm-eval KMMLU 래퍼
- `compression/tests/conftest.py` — tiny GQA Llama 프록시 fixture
- `compression/tests/common/test_param_stats.py`, `test_model_loader.py`, `test_eval_kmmlu.py`

PR2 (Minitron 구조적 프루닝):
- `compression/src/prune/calibration.py` — 보정 데이터 토크나이즈/로더
- `compression/src/prune/importance.py` — 활성값 기반 중요도 수집
- `compression/src/prune/structured_prune.py` — torch-pruning 기반 width 프루닝(GQA 존중)
- `compression/src/prune/report.py` — before/after 리포트
- `compression/scripts/run_prune.py` — CLI 엔트리포인트
- `compression/tests/prune/test_calibration.py`, `test_importance.py`, `test_structured_prune.py`, `test_report.py`

모든 경로는 레포 루트(`Pseudo-Lab/LLM-VLM-in-Jetson`) 기준. 테스트 실행 인터프리터는 `compression/.venv/bin/python`.

---

## Task 0: 프로젝트 scaffold & 환경

**Files:**
- Create: `compression/requirements.txt`
- Create: `compression/pytest.ini`
- Create: `compression/README.md`
- Create: `compression/src/__init__.py`, `compression/src/common/__init__.py`, `compression/src/prune/__init__.py`
- Create: `compression/configs/prune_phi4.yaml`
- Create: `compression/tests/__init__.py`, `compression/tests/common/__init__.py`, `compression/tests/prune/__init__.py`

- [ ] **Step 1: requirements.txt 작성**

`compression/requirements.txt`:
```
torch>=2.2
transformers>=4.45
datasets>=2.19
accelerate>=0.30
torch-pruning>=1.4.1
lm-eval>=0.4.3
pyyaml>=6.0
pytest>=8.0
```

- [ ] **Step 2: venv 생성 및 설치**

Run:
```bash
cd compression && python3.10 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
```
Expected: 설치 성공 (torch CPU 휠 포함). 로컬 Mac은 CPU torch로 충분.

- [ ] **Step 3: pytest.ini 작성 (마커 등록)**

`compression/pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    integration: 실제 대형 모델/데이터셋 필요 (로컬 기본 스킵)
    gpu: CUDA 필요
addopts = -m "not integration and not gpu"
```

- [ ] **Step 4: 패키지 디렉토리 + __init__ 생성**

Run:
```bash
mkdir -p src/common src/prune tests/common tests/prune scripts configs
touch src/__init__.py src/common/__init__.py src/prune/__init__.py tests/__init__.py tests/common/__init__.py tests/prune/__init__.py
```

- [ ] **Step 5: 프루닝 설정 yaml 작성**

`compression/configs/prune_phi4.yaml`:
```yaml
model:
  name: microsoft/phi-4
  dtype: bfloat16
prune:
  width_ratio: 0.30          # 14.7B -> ~10B 목표(30~35% 감축)
  importance: activation     # activation | magnitude
calibration:
  datasets:
    - MarkrAI/KoCommercial-Dataset
    - beomi/KoAlpaca-RealQA
    - beomi/kowikitext-qa-ref-detail-preview
  seq_len: 1024
  n_samples: 256
  seed: 42
output:
  dir: artifacts/phi4-pruned
```

- [ ] **Step 6: README 스켈레톤 작성**

`compression/README.md`:
```markdown
# Phi-4 경량화 (Jetson Orin Nano 8GB)

Minitron 구조적 프루닝 → distillation → GGUF 양자화 파이프라인.
설계 문서: `docs/superpowers/specs/2026-06-08-phi4-compression-jetson-design.md`

## 셋업
    cd compression && python3.10 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## 테스트
    .venv/bin/python -m pytest            # 단위(로컬, tiny 모델)
    .venv/bin/python -m pytest -m integration   # 실모델(GPU 서버)

## 프루닝 실행 (GPU 서버)
    .venv/bin/python scripts/run_prune.py --config configs/prune_phi4.yaml
```

- [ ] **Step 7: Commit**

```bash
cd .. && git add compression && git commit -m "chore: scaffold compression package (deps, configs, dirs)"
```

---

## Task 1: tiny 프록시 모델 fixture (conftest)

**Files:**
- Create: `compression/tests/conftest.py`

tiny GQA Llama는 Phi-4의 구조적 특징(decoder-only, GQA, MLP gate/up/down)을 작게 재현하여 네트워크 없이 프루닝 로직을 검증한다.

- [ ] **Step 1: conftest fixture 작성**

`compression/tests/conftest.py`:
```python
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_model():
    """Phi-4 구조(GQA, MLP gate/up/down)를 축소 재현한 결정론적 프록시 모델."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    return model


@pytest.fixture
def example_inputs():
    """프루닝 DependencyGraph 빌드/forward용 입력."""
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 16))
```

- [ ] **Step 2: fixture 동작 확인 테스트 작성**

`compression/tests/common/test_param_stats.py` (초기 스모크):
```python
def test_tiny_model_forward(tiny_model, example_inputs):
    out = tiny_model(example_inputs)
    assert out.logits.shape == (2, 16, 128)
```

- [ ] **Step 3: 실행하여 PASS 확인**

Run: `.venv/bin/python -m pytest tests/common/test_param_stats.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add compression/tests && git commit -m "test: add tiny GQA Llama proxy fixture"
```

---

## Task 2: common/param_stats.py — 파라미터 통계·메모리 추정

**Files:**
- Create: `compression/src/common/param_stats.py`
- Test: `compression/tests/common/test_param_stats.py`

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/common/test_param_stats.py`에 추가:
```python
from src.common.param_stats import count_parameters, estimate_memory_gb


def test_count_parameters(tiny_model):
    n = count_parameters(tiny_model)
    assert isinstance(n, int) and n > 0


def test_estimate_memory_gb():
    # 14.7B 파라미터를 4비트로 → 약 7.35GB
    gb = estimate_memory_gb(14_700_000_000, bits=4)
    assert 7.0 < gb < 7.7
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/common/test_param_stats.py -v`
Expected: FAIL (`ModuleNotFoundError: src.common.param_stats`)

- [ ] **Step 3: 구현**

`compression/src/common/param_stats.py`:
```python
"""모델 파라미터 카운트 및 메모리 추정 유틸."""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """학습 가능 여부와 무관하게 전체 파라미터 수를 반환."""
    return sum(p.numel() for p in model.parameters())


def estimate_memory_gb(num_params: int, bits: int) -> float:
    """주어진 비트수로 양자화했을 때 가중치 메모리(GB, 10^9 기준)."""
    return num_params * (bits / 8) / 1e9
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/common/test_param_stats.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/common/param_stats.py compression/tests/common/test_param_stats.py
git commit -m "feat: param counting and memory estimation utils"
```

---

## Task 3: common/model_loader.py — HF causal LM 로딩

**Files:**
- Create: `compression/src/common/model_loader.py`
- Test: `compression/tests/common/test_model_loader.py`

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/common/test_model_loader.py`:
```python
import pytest
from src.common.model_loader import resolve_dtype


def test_resolve_dtype():
    import torch
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("float32") == torch.float32


def test_resolve_dtype_invalid():
    with pytest.raises(ValueError):
        resolve_dtype("int4")


@pytest.mark.integration
def test_load_real_model():
    # GPU 서버에서만: 실제 Phi-4 로딩 확인
    from src.common.model_loader import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer("microsoft/phi-4", dtype="bfloat16", device="cuda")
    assert model is not None and tok is not None
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/common/test_model_loader.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 구현**

`compression/src/common/model_loader.py`:
```python
"""HuggingFace causal LM + 토크나이저 로딩."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"지원하지 않는 dtype: {name} (가능: {list(_DTYPES)})")
    return _DTYPES[name]


def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
    """HF 모델·토크나이저를 로딩하여 (model, tokenizer) 반환."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=True,
    ).to(device).eval()
    return model, tokenizer
```

- [ ] **Step 4: PASS 확인 (단위만)**

Run: `.venv/bin/python -m pytest tests/common/test_model_loader.py -v`
Expected: PASS (integration 테스트는 자동 스킵, 단위 2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/common/model_loader.py compression/tests/common/test_model_loader.py
git commit -m "feat: HF causal LM loader with dtype resolution"
```

---

## Task 4: common/eval_kmmlu.py — KMMLU 평가 래퍼

**Files:**
- Create: `compression/src/common/eval_kmmlu.py`
- Test: `compression/tests/common/test_eval_kmmlu.py`

lm-eval-harness의 `kmmlu` 태스크를 호출하고 정확도를 추출한다. 실제 평가는 무겁고 모델이 필요하므로, 단위 테스트는 `lm_eval.simple_evaluate`를 monkeypatch하여 결과 파싱만 검증한다.

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/common/test_eval_kmmlu.py`:
```python
from src.common import eval_kmmlu


def test_extract_accuracy_parses_kmmlu_results():
    fake_results = {"results": {"kmmlu": {"acc,none": 0.4123, "acc_stderr,none": 0.01}}}
    acc = eval_kmmlu.extract_accuracy(fake_results, task="kmmlu")
    assert abs(acc - 0.4123) < 1e-6


def test_run_kmmlu_uses_simple_evaluate(monkeypatch):
    captured = {}

    def fake_simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {"kmmlu": {"acc,none": 0.5}}}

    monkeypatch.setattr(eval_kmmlu, "simple_evaluate", fake_simple_evaluate)
    acc = eval_kmmlu.run_kmmlu("dummy/path", limit=8, device="cpu")
    assert acc == 0.5
    assert captured["tasks"] == ["kmmlu"]
    assert captured["limit"] == 8
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/common/test_eval_kmmlu.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`compression/src/common/eval_kmmlu.py`:
```python
"""KMMLU(한국어 MMLU) 평가 래퍼 — lm-eval-harness 기반."""
from __future__ import annotations

from lm_eval import simple_evaluate


def extract_accuracy(results: dict, task: str = "kmmlu") -> float:
    """lm-eval 결과 dict에서 정확도(acc)를 추출."""
    return float(results["results"][task]["acc,none"])


def run_kmmlu(model_path: str, limit: int | None = None, device: str = "cuda") -> float:
    """주어진 모델 경로/이름에 대해 KMMLU 정확도를 측정해 반환."""
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=["kmmlu"],
        limit=limit,
        device=device,
    )
    return extract_accuracy(results, task="kmmlu")
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/common/test_eval_kmmlu.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/common/eval_kmmlu.py compression/tests/common/test_eval_kmmlu.py
git commit -m "feat: KMMLU evaluation wrapper over lm-eval-harness"
```

**→ 여기까지가 PR1 (scaffold + common + eval 골격). PR 생성 가능 지점.**

---

## Task 5: prune/calibration.py — 보정 데이터 토크나이즈

**Files:**
- Create: `compression/src/prune/calibration.py`
- Test: `compression/tests/prune/test_calibration.py`

활성값 중요도 계산에 쓸 보정 배치를 만든다. 네트워크 의존을 피하려 토크나이즈 로직(`tokenize_texts`)과 데이터셋 로딩(`load_korean_texts`, integration)을 분리한다.

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/prune/test_calibration.py`:
```python
import torch
from transformers import AutoTokenizer
from src.prune.calibration import tokenize_texts


def test_tokenize_texts_shape():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    texts = ["안녕하세요 반갑습니다", "오늘 날씨가 좋네요", "경량화 테스트 문장"]
    batch = tokenize_texts(texts, tok, seq_len=8)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[1] == 8
    assert batch.shape[0] == 3
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`compression/src/prune/calibration.py`:
```python
"""프루닝 보정(calibration) 데이터 준비."""
from __future__ import annotations

import torch


def tokenize_texts(texts: list[str], tokenizer, seq_len: int) -> torch.Tensor:
    """텍스트 리스트를 고정 길이 input_ids 배치로 토크나이즈."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return enc["input_ids"]


def load_korean_texts(dataset_names: list[str], n_samples: int, seed: int = 42) -> list[str]:
    """한국어 보정 코퍼스에서 텍스트 샘플을 모은다 (GPU 서버/네트워크 필요).

    KoCommercial / KoAlpaca-RealQA / kowikitext-qa의 instruction·output 필드를
    하나의 텍스트로 합쳐 반환한다.
    """
    from datasets import load_dataset

    texts: list[str] = []
    per = max(1, n_samples // len(dataset_names))
    for name in dataset_names:
        ds = load_dataset(name, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= per:
                break
            texts.append(_row_to_text(row))
    return texts[:n_samples]


def _row_to_text(row: dict) -> str:
    """데이터셋 행에서 학습용 텍스트를 추출 (필드명 차이 흡수)."""
    for key in ("text", "instruction", "question", "input"):
        if key in row and row[key]:
            extra = row.get("output") or row.get("answer") or ""
            return f"{row[key]}\n{extra}".strip()
    return str(next(iter(row.values())))
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add compression/src/prune/calibration.py compression/tests/prune/test_calibration.py
git commit -m "feat: calibration data tokenization for pruning"
```

---

## Task 6: prune/importance.py — 활성값 기반 중요도 수집

**Files:**
- Create: `compression/src/prune/importance.py`
- Test: `compression/tests/prune/test_importance.py`

Minitron 방식: 보정 배치를 forward하며 각 Linear 출력 채널의 활성값 제곱합을 누적 → 채널별 중요도(L2). gradient 불필요.

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/prune/test_importance.py`:
```python
import torch
from src.prune.importance import collect_activation_importance


def test_collect_activation_importance_per_channel(tiny_model, example_inputs):
    # 각 MLP up_proj 출력 채널(intermediate_size=128)에 대한 중요도 점수
    target = "model.layers.0.mlp.up_proj"
    scores = collect_activation_importance(tiny_model, [example_inputs], [target])
    assert target in scores
    vec = scores[target]
    assert vec.shape == (128,)
    assert torch.all(vec >= 0)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_importance.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`compression/src/prune/importance.py`:
```python
"""활성값 기반 구조적 프루닝 중요도(Minitron 방식)."""
from __future__ import annotations

import torch
import torch.nn as nn


def collect_activation_importance(
    model: nn.Module,
    calib_batches: list[torch.Tensor],
    target_module_names: list[str],
) -> dict[str, torch.Tensor]:
    """대상 모듈 출력 채널별 활성값 L2 중요도를 보정 배치로 누적해 반환.

    importance[ch] = sqrt(mean_over_tokens(activation[..., ch] ** 2))
    """
    name_to_module = dict(model.named_modules())
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            act = out.detach().float()
            flat = act.reshape(-1, act.shape[-1])  # (tokens, channels)
            sq = (flat ** 2).sum(dim=0)
            sums[name] = sums.get(name, torch.zeros_like(sq)) + sq
            counts[name] = counts.get(name, 0) + flat.shape[0]
        return hook

    for name in target_module_names:
        handles.append(name_to_module[name].register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in calib_batches:
            model(batch)

    for h in handles:
        h.remove()

    return {name: torch.sqrt(sums[name] / counts[name]) for name in target_module_names}
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_importance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add compression/src/prune/importance.py compression/tests/prune/test_importance.py
git commit -m "feat: activation-based importance scoring (Minitron-style)"
```

---

## Task 7: prune/structured_prune.py — torch-pruning 기반 width 프루닝

**Files:**
- Create: `compression/src/prune/structured_prune.py`
- Test: `compression/tests/prune/test_structured_prune.py`

torch-pruning의 `MetaPruner`로 결합구조를 일관 프루닝한다. `num_heads`를 넘겨 GQA attention을 헤드 단위로 처리하고, `lm_head`·embedding은 `ignored_layers`로 보호한다. 중요도는 활성값(없으면 magnitude)으로.

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/prune/test_structured_prune.py`:
```python
import torch
from src.common.param_stats import count_parameters
from src.prune.structured_prune import prune_width


def test_prune_width_reduces_params_and_runs(tiny_model, example_inputs):
    before = count_parameters(tiny_model)
    pruned, info = prune_width(tiny_model, example_inputs, ratio=0.25)
    after = count_parameters(pruned)

    # 파라미터가 실제로 감소
    assert after < before
    assert info["params_before"] == before
    assert info["params_after"] == after
    # forward가 여전히 동작하고 vocab 차원 유지
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_structured_prune.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`compression/src/prune/structured_prune.py`:
```python
"""torch-pruning 기반 구조적 width 프루닝 (GQA 존중)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch_pruning as tp

from src.common.param_stats import count_parameters


class _PrecomputedImportance(tp.importance.Importance):
    """활성값 등으로 미리 계산한 채널 점수를 torch-pruning에 공급."""

    def __init__(self, scores_by_module: dict[nn.Module, torch.Tensor]):
        self._scores = scores_by_module

    def __call__(self, group, **kwargs):
        for dep, idxs in group:
            layer = dep.target.module
            if layer in self._scores:
                return self._scores[layer][idxs]
        # fallback: L2 magnitude
        return tp.importance.MagnitudeImportance(p=2)(group, **kwargs)


def _attention_head_map(model: nn.Module) -> dict[nn.Module, int]:
    """attention q/k/v proj 모듈 → head 수 매핑 (GQA 처리용)."""
    head_map: dict[nn.Module, int] = {}
    cfg = model.config
    for layer in model.model.layers:
        attn = layer.self_attn
        head_map[attn.q_proj] = cfg.num_attention_heads
        head_map[attn.k_proj] = cfg.num_key_value_heads
        head_map[attn.v_proj] = cfg.num_key_value_heads
    return head_map


def prune_width(
    model: nn.Module,
    example_inputs: torch.Tensor,
    ratio: float,
    importance_scores: dict[nn.Module, torch.Tensor] | None = None,
):
    """폭(width) 구조적 프루닝을 수행하고 (pruned_model, info) 반환."""
    params_before = count_parameters(model)

    importance = (
        _PrecomputedImportance(importance_scores)
        if importance_scores
        else tp.importance.MagnitudeImportance(p=2)
    )
    ignored = [model.lm_head, model.model.embed_tokens]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=ignored,
        num_heads=_attention_head_map(model),
        prune_num_heads=True,
        prune_head_dims=False,
        global_pruning=False,
    )
    pruner.step()

    params_after = count_parameters(model)
    info = {
        "params_before": params_before,
        "params_after": params_after,
        "ratio_actual": 1.0 - params_after / params_before,
        "ratio_target": ratio,
    }
    return model, info
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_structured_prune.py -v`
Expected: PASS

> torch-pruning의 attention 처리는 모델 구조에 민감하다. tiny Llama에서 `prune_num_heads`/`num_heads` 인자가 버전에 따라 다르면, 설치된 torch-pruning 버전의 LLM 예제(`examples/transformers`)를 참조해 인자명을 맞춘다. 핵심 검증은 "파라미터 감소 + forward 정상"이다.

- [ ] **Step 5: Commit**

```bash
git add compression/src/prune/structured_prune.py compression/tests/prune/test_structured_prune.py
git commit -m "feat: structured width pruning via torch-pruning (GQA-aware)"
```

---

## Task 8: prune/report.py — before/after 리포트

**Files:**
- Create: `compression/src/prune/report.py`
- Test: `compression/tests/prune/test_report.py`

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/prune/test_report.py`:
```python
from src.prune.report import format_prune_report


def test_format_prune_report_contains_key_numbers():
    info = {"params_before": 1000, "params_after": 700, "ratio_actual": 0.30, "ratio_target": 0.30}
    text = format_prune_report(info, bits=4)
    assert "1,000" in text
    assert "700" in text
    assert "30" in text          # 감축률 %
    assert "GB" in text          # 메모리 추정 포함
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`compression/src/prune/report.py`:
```python
"""프루닝 before/after 리포트 포매팅."""
from __future__ import annotations

from src.common.param_stats import estimate_memory_gb


def format_prune_report(info: dict, bits: int = 4) -> str:
    """프루닝 결과 info dict를 사람이 읽는 표로 포매팅."""
    before = info["params_before"]
    after = info["params_after"]
    mem_before = estimate_memory_gb(before, bits)
    mem_after = estimate_memory_gb(after, bits)
    pct = info["ratio_actual"] * 100
    return (
        "=== Pruning Report ===\n"
        f"params: {before:,} -> {after:,} ({pct:.1f}% 감축, 목표 {info['ratio_target']*100:.0f}%)\n"
        f"메모리({bits}bit 추정): {mem_before:.2f}GB -> {mem_after:.2f}GB"
    )
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add compression/src/prune/report.py compression/tests/prune/test_report.py
git commit -m "feat: pruning before/after report formatting"
```

---

## Task 9: scripts/run_prune.py — CLI 엔트리포인트

**Files:**
- Create: `compression/scripts/run_prune.py`
- Test: `compression/tests/prune/test_run_prune.py`

config를 읽어 모델 로딩 → 보정 → 중요도 → 프루닝 → 저장 → 리포트를 엮는다. 실모델 실행은 GPU 서버(integration). 단위 테스트는 config 파싱·타깃 모듈 선택 로직만 검증한다.

- [ ] **Step 1: 실패 테스트 작성**

`compression/tests/prune/test_run_prune.py`:
```python
from src.prune.run_helpers import select_target_modules


def test_select_target_modules_picks_mlp_linears(tiny_model):
    names = select_target_modules(tiny_model)
    assert "model.layers.0.mlp.up_proj" in names
    assert "model.layers.0.mlp.gate_proj" in names
    # embedding/lm_head은 제외
    assert all("embed_tokens" not in n and "lm_head" not in n for n in names)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_run_prune.py -v`
Expected: FAIL

- [ ] **Step 3: 헬퍼 구현**

`compression/src/prune/run_helpers.py`:
```python
"""run_prune CLI용 헬퍼 (단위 테스트 가능한 순수 로직)."""
from __future__ import annotations

import torch.nn as nn


def select_target_modules(model: nn.Module) -> list[str]:
    """활성값 중요도를 수집할 대상 Linear(주로 MLP) 이름 목록."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".mlp." in name:
            targets.append(name)
    return targets
```

- [ ] **Step 4: PASS 확인**

Run: `.venv/bin/python -m pytest tests/prune/test_run_prune.py -v`
Expected: PASS

- [ ] **Step 5: CLI 엔트리포인트 작성 (integration)**

`compression/scripts/run_prune.py`:
```python
"""Phi-4 구조적 프루닝 실행 (GPU 서버)."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.model_loader import load_model_and_tokenizer  # noqa: E402
from src.prune.calibration import tokenize_texts, load_korean_texts  # noqa: E402
from src.prune.importance import collect_activation_importance  # noqa: E402
from src.prune.run_helpers import select_target_modules  # noqa: E402
from src.prune.structured_prune import prune_width  # noqa: E402
from src.prune.report import format_prune_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model, tok = load_model_and_tokenizer(
        cfg["model"]["name"], dtype=cfg["model"]["dtype"], device="cuda"
    )

    texts = load_korean_texts(
        cfg["calibration"]["datasets"],
        n_samples=cfg["calibration"]["n_samples"],
        seed=cfg["calibration"]["seed"],
    )
    batch = tokenize_texts(texts, tok, seq_len=cfg["calibration"]["seq_len"]).to("cuda")

    target_names = select_target_modules(model)
    scores_by_name = collect_activation_importance(model, [batch], target_names)
    name_to_module = dict(model.named_modules())
    scores_by_module = {name_to_module[n]: s for n, s in scores_by_name.items()}

    example_inputs = batch[:1]
    model, info = prune_width(
        model, example_inputs, ratio=cfg["prune"]["width_ratio"],
        importance_scores=scores_by_module,
    )

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(format_prune_report(info, bits=4))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 전체 단위 테스트 통과 확인**

Run: `.venv/bin/python -m pytest -v`
Expected: 모든 단위 테스트 PASS, integration/gpu 마커는 스킵

- [ ] **Step 7: Commit**

```bash
git add compression/src/prune/run_helpers.py compression/scripts/run_prune.py compression/tests/prune/test_run_prune.py
git commit -m "feat: run_prune CLI entrypoint and target selection"
```

**→ 여기까지가 PR2 (Minitron 구조적 프루닝 구현 시작). PR 생성 지점.**

---

## (GPU 서버) 실모델 검증 — integration

로컬 단위 테스트 통과 후, GPU 서버에서 실제 Phi-4로 한 번 돌려 스모크 확인한다 (1차 발표 범위의 "동작 증명").

- [ ] Phi-4 로딩 integration 테스트: `.venv/bin/python -m pytest -m integration tests/common/test_model_loader.py -v`
- [ ] 프루닝 실행: `.venv/bin/python scripts/run_prune.py --config configs/prune_phi4.yaml`
  - Expected: `=== Pruning Report ===` 출력, params 약 30% 감축, `artifacts/phi4-pruned/`에 저장
- [ ] (선택) 프루닝 전/후 KMMLU 비교: `run_kmmlu`로 baseline 대비 하락폭 기록 (distillation 회복 전이라 하락은 정상)

---

## Self-Review 결과

- **Spec 커버리지(1차 범위)**: §3[1] 프루닝(Minitron 활성값 width) → Task 6·7, §4 평가(KMMLU) → Task 4, §5 레포 구조/PR1·PR2 → Task 0~9, common 추상화(리스크 대비) → Task 2·3. distill/quant/E2E는 의도적으로 후속 plan.
- **Placeholder 스캔**: 모든 코드 step에 실제 코드 포함, 미완 표현 없음.
- **타입 일관성**: `prune_width`가 반환하는 `info` 키(`params_before/after`, `ratio_actual/target`)를 Task 7·8·9에서 동일하게 사용. `collect_activation_importance`는 이름→텐서 dict, `prune_width`는 모듈→텐서 dict를 받으므로 run_prune에서 변환 단계 명시.
- **알려진 위험**: torch-pruning의 attention head 프루닝 인자는 라이브러리 버전에 민감 → Task 7 Step 4 주석에 대응법 명시.
