# distill full-FT + logit-KD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** depth-pruned Phi-4 student를 teacher의 top-k logit 분포로 full-FT distill(logit-level KD)해서 단일턴 한국어 채팅 체감품질을 올린다.

**Architecture:** teacher(Phi-4)를 한 번만 돌려 응답 토큰별 top-k logit을 디스크에 사전계산한 뒤, teacher-free로 student를 full-FT한다. 손실 = `α·T²·KL(teacher_topk ‖ student) + (1−α)·CE`. 학습 데이터는 기존 25k 고정.

**Tech Stack:** PyTorch, transformers>=5.0, safetensors, bitsandbytes(paged_adamw_8bit), pytest. 실행은 GCP `phi4-blackwell`(RTX PRO 6000 96GB) VM.

## Global Constraints

- **공유 vocab:** student(`phi4-pruned-depth-masked`)와 teacher(`microsoft/phi-4`)는 vocab 동일(100,352). top-k 인덱스는 두 모델에서 직접 비교 가능 — 이 전제가 깨지면 KD 전체가 무효.
- **shift 규약(가장 흔한 버그):** causal LM에서 `logits[i]`는 "토큰 i 이후(=i+1)"를 예측. label 토큰 인덱스 `j`(assistant 토큰)를 supervise할 때 예측 위치는 `i = j-1`. teacher top-k도 `teacher_logits[j-1]`에서 뽑고, student도 `student_logits[j-1]`에서 gather한다. **precompute와 train이 동일 규약을 쓴다.**
- **assistant-only supervision:** 기존 `train_lora_sft.py`는 전체 시퀀스(user 프롬프트 포함)를 supervise했으나, 본 KD는 **assistant 응답 토큰만** supervise(label=-100 마스킹). 의도된 개선.
- **데이터 고정:** 입력은 VM의 `data/distill/pilot_teacher.jsonl`(24,958쌍). 신규 데이터 생성 금지(그건 Approach B).
- **하이퍼파라미터 1차값:** `T=2.0, α=0.9, k=64, lr=1e-5, epochs=2, optim=paged_adamw_8bit`.
- **venv:** 학습/테스트는 `.venv`(py3.11). vLLM 서빙은 `.venv-vllm`.
- **git:** 이 repo는 코드만 동기화(`data/`·`artifacts/`는 gitignore, VM-only). 광범위 `git add -A` 금지 — 변경 파일만 명시 add.

**파일 구조 (생성/수정)**
- Create `compression/src/distill/__init__.py` — 패키지 마커
- Create `compression/src/distill/kd_loss.py` — 순수 손실 함수(KD+CE). 핵심 유닛.
- Create `compression/src/distill/kd_data.py` — top-k 사전계산본 로딩 Dataset + 패딩 Collator
- Create `compression/src/distill/chat_labels.py` — chat_template 렌더링 + assistant-only label 마스크
- Create `compression/scripts/precompute_teacher_logits.py` — teacher forward → top-k 저장 (오케스트레이션)
- Create `compression/scripts/train_distill_kd.py` — full-FT KDTrainer (오케스트레이션)
- Create `compression/tests/distill/__init__.py`, `test_kd_loss.py`, `test_kd_data.py`, `test_chat_labels.py`
- Create `docs/results/probes/chat_probes_ko.jsonl` — 고정 프로브 10~15개
- Modify `compression/requirements.txt` — `bitsandbytes` 추가
- Modify `docs/HANDOFF.md` 또는 신규 HANDOFF — 런북(맨 끝 Task)

---

### Task 1: KD 손실 함수 (순수 유닛, TDD)

**Files:**
- Create: `compression/src/distill/__init__.py`
- Create: `compression/src/distill/kd_loss.py`
- Test: `compression/tests/distill/__init__.py`, `compression/tests/distill/test_kd_loss.py`

**Interfaces:**
- Produces:
  ```python
  def kd_topk_loss(
      student_logits,   # Tensor [B, L, V]
      pos,              # Tensor [B, P] long — label 토큰 인덱스 j (>=1)
      pos_mask,         # Tensor [B, P] — 1.0 valid / 0.0 padding
      topk_idx,         # Tensor [B, P, k] long — teacher top-k 토큰 id
      topk_logit,       # Tensor [B, P, k] float — teacher top-k logit (소프트닝 전)
      hard_labels,      # Tensor [B, P] long — pos 위치의 정답 토큰 id (= input_ids[pos])
      temperature=2.0,
      alpha=0.9,
  ) -> dict   # {"loss", "loss_kd", "loss_ce"}  (loss는 backward 가능 스칼라)
  ```
  핵심 규약: 예측 위치 = `pos-1`. student pred = `student_logits.gather(1, (pos-1))`. teacher 확률 = `softmax(topk_logit/T)` (top-k 위에서 이미 정규화). KD = `T² · Σ_k p_t·(log p_t − logsoftmax(student)_at_idx)`, padding 위치 제외 평균. CE = pred에 대한 `cross_entropy(hard_labels)`, padding 제외.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_kd_loss.py
import torch
from src.distill.kd_loss import kd_topk_loss


def _toy():
    # B=1, L=3, V=5, P=1, k=2. label 위치 j=2 → 예측 위치 i=1.
    student_logits = torch.zeros(1, 3, 5)
    pos = torch.tensor([[2]])
    pos_mask = torch.tensor([[1.0]])
    topk_idx = torch.tensor([[[3, 1]]])        # teacher top-k 토큰 id
    topk_logit = torch.tensor([[[2.0, 0.0]]])  # teacher가 3을 선호
    hard_labels = torch.tensor([[3]])
    return student_logits, pos, pos_mask, topk_idx, topk_logit, hard_labels


def test_returns_finite_scalar_with_grad():
    s, pos, m, ti, tl, hl = _toy()
    s.requires_grad_(True)
    out = kd_topk_loss(s, pos, m, ti, tl, hl, temperature=2.0, alpha=0.9)
    assert set(out) == {"loss", "loss_kd", "loss_ce"}
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_perfect_match_gives_near_zero_kd():
    # student이 teacher top-k 분포를 그대로 재현하면 KD ≈ 0
    s, pos, m, ti, tl, hl = _toy()
    # 예측 위치 i=1 의 logit을 teacher와 일치시킴 (idx 3→2.0, idx 1→0.0, 나머지 매우 작게)
    s[0, 1, :] = -30.0
    s[0, 1, 3] = 2.0
    s[0, 1, 1] = 0.0
    out = kd_topk_loss(s, pos, m, ti, tl, hl, temperature=2.0, alpha=1.0)
    assert out["loss_kd"].item() < 1e-3


def test_padding_position_is_ignored():
    s, pos, m, ti, tl, hl = _toy()
    # pos를 2개로 늘리고 두 번째는 padding(mask=0) — 결과가 P=1 단독과 동일해야
    pos2 = torch.tensor([[2, 0]])
    m2 = torch.tensor([[1.0, 0.0]])
    ti2 = torch.tensor([[[3, 1], [0, 0]]])
    tl2 = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    hl2 = torch.tensor([[3, -100]])
    a = kd_topk_loss(s, pos, m, ti, tl, hl)["loss"]
    b = kd_topk_loss(s, pos2, m2, ti2, tl2, hl2)["loss"]
    assert torch.allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/pytest tests/distill/test_kd_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/__init__.py
```
(빈 파일)

```python
# compression/src/distill/kd_loss.py
"""Top-k logit-level KD loss (+ CE) for sequence-level distillation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_topk_loss(
    student_logits: torch.Tensor,   # [B, L, V]
    pos: torch.Tensor,              # [B, P] long, label 토큰 인덱스 j (>=1)
    pos_mask: torch.Tensor,         # [B, P], 1 valid / 0 padding
    topk_idx: torch.Tensor,         # [B, P, k] long
    topk_logit: torch.Tensor,       # [B, P, k] float (소프트닝 전 teacher logit)
    hard_labels: torch.Tensor,      # [B, P] long (= input_ids[pos]); padding은 -100 가능
    temperature: float = 2.0,
    alpha: float = 0.9,
) -> dict:
    B, P, k = topk_idx.shape
    V = student_logits.size(-1)
    pred_pos = (pos - 1).clamp(min=0)                       # [B, P] 예측 위치 i=j-1
    idx = pred_pos.unsqueeze(-1).expand(B, P, V)            # [B, P, V]
    pred_logits = student_logits.gather(1, idx)            # [B, P, V]

    # --- KD: top-k 위에서 KL ---
    t_prob = F.softmax(topk_logit / temperature, dim=-1)    # [B, P, k] (top-k 정규화)
    s_logprob_full = F.log_softmax(pred_logits / temperature, dim=-1)  # [B, P, V]
    s_logprob_topk = s_logprob_full.gather(2, topk_idx)     # [B, P, k]
    kl = (t_prob * (torch.log(t_prob + 1e-9) - s_logprob_topk)).sum(-1)  # [B, P]
    mask = pos_mask.float()
    denom = mask.sum().clamp(min=1.0)
    loss_kd = (temperature ** 2) * (kl * mask).sum() / denom

    # --- CE: 정답 토큰 hard target ---
    ce_tok = F.cross_entropy(
        pred_logits.reshape(B * P, V),
        hard_labels.reshape(B * P),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, P)
    loss_ce = (ce_tok * mask).sum() / denom

    loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
    return {"loss": loss, "loss_kd": loss_kd.detach(), "loss_ce": loss_ce.detach()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/pytest tests/distill/test_kd_loss.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/__init__.py compression/src/distill/kd_loss.py \
        compression/tests/distill/__init__.py compression/tests/distill/test_kd_loss.py
git commit -m "feat(distill): top-k logit-level KD loss"
```

---

### Task 2: chat_template assistant-only 라벨 (TDD)

**Files:**
- Create: `compression/src/distill/chat_labels.py`
- Test: `compression/tests/distill/test_chat_labels.py`

**Interfaces:**
- Produces:
  ```python
  def build_assistant_labels(tokenizer, messages, max_length=2048) -> dict
  # 반환: {"input_ids": list[int], "pos": list[int]}
  #   input_ids = apply_chat_template(messages, tokenize, add_generation_prompt=False)
  #   pos       = assistant 응답 토큰 인덱스들 (prompt 접두부 길이 이후, j>=1)
  ```
  prompt 접두부 길이 = `apply_chat_template(messages[:-1], add_generation_prompt=True)` 토큰 수. 그 이후~끝이 assistant 구간. `j=0`은 절대 포함 안 됨(접두부가 항상 존재).

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_chat_labels.py
from src.distill.chat_labels import build_assistant_labels


class FakeTok:
    """role 토큰을 정수로 흉내내는 최소 토크나이저."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]  # BOS
        for m in messages:
            head = 10 if m["role"] == "user" else 20
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]  # assistant 헤더만 (응답 없음)
        return ids if tokenize else "".join(map(str, ids))


def test_assistant_span_only():
    tok = FakeTok()
    messages = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
    ]
    out = build_assistant_labels(tok, messages, max_length=2048)
    full = tok.apply_chat_template(messages, add_generation_prompt=False)
    prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
    assert out["input_ids"] == full
    # pos 는 prefix 길이 이후의 모든 인덱스
    assert out["pos"] == list(range(len(prefix), len(full)))
    assert 0 not in out["pos"]


def test_truncation_respected():
    tok = FakeTok()
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
    ]
    out = build_assistant_labels(tok, messages, max_length=32)
    assert len(out["input_ids"]) <= 32
    assert all(p < len(out["input_ids"]) for p in out["pos"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/pytest tests/distill/test_chat_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.chat_labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/chat_labels.py
"""Render chat and mark assistant-only supervised positions."""
from __future__ import annotations


def build_assistant_labels(tokenizer, messages, max_length: int = 2048) -> dict:
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full = list(full)[:max_length]
    start = min(len(prefix), len(full))
    pos = [j for j in range(start, len(full)) if j >= 1]
    return {"input_ids": full, "pos": pos}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/pytest tests/distill/test_chat_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/chat_labels.py compression/tests/distill/test_chat_labels.py
git commit -m "feat(distill): assistant-only chat label masking"
```

---

### Task 3: top-k KD Dataset + Collator (TDD)

**Files:**
- Create: `compression/src/distill/kd_data.py`
- Test: `compression/tests/distill/test_kd_data.py`

**Interfaces:**
- Consumes: precompute 산출물 — 디렉터리 `teacher_kd/` 안에 예제별 `*.safetensors`(키: `input_ids[L] int32`, `pos[P] int32`, `topk_idx[P,k] int32`, `topk_logit[P,k] float16`) + `manifest.jsonl`(줄당 `{"id","file","n_pos"}`).
- Produces:
  ```python
  class TopKKDDataset(torch.utils.data.Dataset):
      def __init__(self, manifest_path: str): ...
      def __getitem__(self, i) -> dict  # input_ids, pos, topk_idx, topk_logit, hard_labels(list)
  class KDCollator:
      def __call__(self, features) -> dict
      # 반환 텐서: input_ids[B,L], attention_mask[B,L], pos[B,P], pos_mask[B,P],
      #            topk_idx[B,P,k], topk_logit[B,P,k], hard_labels[B,P]
      # input_ids 패딩=pad_id, pos/topk 패딩=0, pos_mask=0, hard_labels 패딩=-100
  ```
  `hard_labels[b,p] = input_ids[b, pos[b,p]]`.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_kd_data.py
import json
import torch
from safetensors.torch import save_file
from src.distill.kd_data import TopKKDDataset, KDCollator


def _make_example(tmp, name, L, P, k):
    input_ids = torch.arange(L, dtype=torch.int32)
    pos = torch.arange(L - P, L, dtype=torch.int32)         # 마지막 P개가 assistant
    topk_idx = torch.zeros(P, k, dtype=torch.int32)
    topk_logit = torch.ones(P, k, dtype=torch.float16)
    f = tmp / f"{name}.safetensors"
    save_file(
        {"input_ids": input_ids, "pos": pos, "topk_idx": topk_idx, "topk_logit": topk_logit},
        str(f),
    )
    return {"id": name, "file": str(f), "n_pos": P}


def test_dataset_and_collator_shapes(tmp_path):
    rows = [_make_example(tmp_path, "a", L=6, P=2, k=3),
            _make_example(tmp_path, "b", L=8, P=4, k=3)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ds = TopKKDDataset(str(manifest))
    assert len(ds) == 2
    batch = KDCollator(pad_id=0)([ds[0], ds[1]])

    assert batch["input_ids"].shape == (2, 8)               # max L
    assert batch["pos"].shape == (2, 4)                     # max P
    assert batch["topk_idx"].shape == (2, 4, 3)
    assert batch["attention_mask"].sum().item() == 6 + 8    # 실토큰 수
    # 예제 a는 P=2라 뒤 2자리 padding
    assert batch["pos_mask"][0].tolist() == [1, 1, 0, 0]
    # hard_labels = input_ids[pos]; padding 위치는 -100
    assert batch["hard_labels"][0, 2].item() == -100
    assert batch["hard_labels"][0, 0].item() == ds[0]["input_ids"][ds[0]["pos"][0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/pytest tests/distill/test_kd_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.kd_data'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/kd_data.py
"""Dataset + collator over precomputed teacher top-k logits."""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class TopKKDDataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = load_file(self.rows[i]["file"])
        input_ids = t["input_ids"].long()
        pos = t["pos"].long()
        return {
            "input_ids": input_ids,
            "pos": pos,
            "topk_idx": t["topk_idx"].long(),
            "topk_logit": t["topk_logit"].float(),
            "hard_labels": input_ids[pos],
        }


class KDCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        B = len(features)
        L = max(f["input_ids"].size(0) for f in features)
        P = max(f["pos"].size(0) for f in features)
        k = features[0]["topk_idx"].size(1)

        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        pos = torch.zeros((B, P), dtype=torch.long)
        pos_mask = torch.zeros((B, P), dtype=torch.float)
        topk_idx = torch.zeros((B, P, k), dtype=torch.long)
        topk_logit = torch.zeros((B, P, k), dtype=torch.float)
        hard_labels = torch.full((B, P), -100, dtype=torch.long)

        for b, f in enumerate(features):
            li, pi = f["input_ids"].size(0), f["pos"].size(0)
            input_ids[b, :li] = f["input_ids"]
            attn[b, :li] = 1
            pos[b, :pi] = f["pos"]
            pos_mask[b, :pi] = 1.0
            topk_idx[b, :pi] = f["topk_idx"]
            topk_logit[b, :pi] = f["topk_logit"]
            hard_labels[b, :pi] = f["hard_labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pos": pos,
            "pos_mask": pos_mask,
            "topk_idx": topk_idx,
            "topk_logit": topk_logit,
            "hard_labels": hard_labels,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/pytest tests/distill/test_kd_data.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/kd_data.py compression/tests/distill/test_kd_data.py
git commit -m "feat(distill): top-k KD dataset and padding collator"
```

---

### Task 4: teacher top-k 사전계산 스크립트

**Files:**
- Create: `compression/scripts/precompute_teacher_logits.py`
- Modify: `compression/requirements.txt` (bitsandbytes 추가 — Task 5에서 쓰지만 함께 둠)
- Test: 본 Task는 `@pytest.mark.gpu` 통합 sanity (기본 스킵). 단위 검증은 Task 1–3가 커버.

**Interfaces:**
- Consumes: `build_assistant_labels`(Task 2).
- Produces: `teacher_kd/<id>.safetensors` + `teacher_kd/manifest.jsonl` — Task 3 Dataset이 읽는 포맷. `topk_logit = teacher_logits[j-1].topk(k)`(shift 규약 준수).

- [ ] **Step 1: Write the script**

```python
# compression/scripts/precompute_teacher_logits.py
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
    ap.add_argument("--responses", default="data/distill/pilot_teacher.jsonl")
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
    print(f"done rows={done} -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add bitsandbytes to requirements**

`compression/requirements.txt`의 `peft>=0.11` 줄 아래에 추가:

```
bitsandbytes>=0.43        # paged_adamw_8bit full-FT 옵티마이저
safetensors>=0.4          # teacher top-k 사전계산 저장
```

- [ ] **Step 3: Smoke-run with tiny limit on VM (GPU)**

Run (VM `.venv`):
```bash
cd compression && .venv/bin/python scripts/precompute_teacher_logits.py \
  --responses data/distill/pilot_teacher.jsonl --out-dir data/distill/teacher_kd_smoke \
  --k 64 --limit 8
```
Expected: `done rows=8 -> data/distill/teacher_kd_smoke`, 8개 `*.safetensors` + `manifest.jsonl` 생성.

- [ ] **Step 4: Alignment sanity check (shift 규약 검증)**

Run (VM `.venv`):
```bash
cd compression && .venv/bin/python - <<'PY'
import json, torch
from safetensors.torch import load_file
m = [json.loads(l) for l in open("data/distill/teacher_kd_smoke/manifest.jsonl")]
hits = tot = 0
for r in m:
    t = load_file(r["file"])
    ids, pos, tk = t["input_ids"].long(), t["pos"].long(), t["topk_idx"].long()
    gold = ids[pos]                      # assistant 실제 토큰
    hits += (tk[:, 0] == gold).sum().item()   # teacher top-1 == 실제 토큰?
    tot += len(pos)
print(f"top1==gold rate = {hits/tot:.2f} (temp 0.2 샘플링이라 0.6~0.9 기대)")
PY
```
Expected: rate가 0.6 이상(shift 규약이 틀리면 ~0으로 떨어짐 → 버그 신호).

- [ ] **Step 5: Commit**

```bash
git add compression/scripts/precompute_teacher_logits.py compression/requirements.txt
git commit -m "feat(distill): precompute teacher top-k logits with shift-correct alignment"
```

---

### Task 5: full-FT KD 학습 스크립트

**Files:**
- Create: `compression/scripts/train_distill_kd.py`
- Test: `@pytest.mark.gpu` smoke (기본 스킵) — Step 3에서 VM 실행.

**Interfaces:**
- Consumes: `kd_topk_loss`(Task 1), `TopKKDDataset`/`KDCollator`(Task 3).
- Produces: `artifacts/phi4-pruned-depth-distill-kd-v1/` (full-FT student + tokenizer + chat_template.jinja).

- [ ] **Step 1: Write the script**

```python
# compression/scripts/train_distill_kd.py
"""Full-FT student with offline top-k logit-level KD."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.kd_data import TopKKDDataset, KDCollator
from src.distill.kd_loss import kd_topk_loss

_KD_KEYS = ("pos", "pos_mask", "topk_idx", "topk_logit", "hard_labels")


class KDTrainer(Trainer):
    def __init__(self, *a, temperature=2.0, alpha=0.9, **kw):
        super().__init__(*a, **kw)
        self._T, self._alpha = temperature, alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        kd = {key: inputs.pop(key) for key in _KD_KEYS}
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        res = kd_topk_loss(
            out.logits, kd["pos"], kd["pos_mask"], kd["topk_idx"],
            kd["topk_logit"], kd["hard_labels"],
            temperature=self._T, alpha=self._alpha,
        )
        return (res["loss"], out) if return_outputs else res["loss"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="artifacts/phi4-pruned-depth-masked")
    ap.add_argument("--manifest", default="data/distill/teacher_kd/manifest.jsonl")
    ap.add_argument("--out", default="artifacts/phi4-pruned-depth-distill-kd-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--max-steps", type=int, default=-1)   # smoke용
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.student, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = TopKKDDataset(args.manifest)
    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = KDTrainer(
        model=model, args=train_args, train_dataset=ds,
        data_collator=KDCollator(pad_id=tok.pad_token_id),
        temperature=args.temperature, alpha=args.alpha,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # distill 서빙은 chat-template 전용 → 명시 동봉
    src_tpl = Path(args.student) / "chat_template.jinja"
    if src_tpl.exists():
        (Path(args.out) / "chat_template.jinja").write_text(
            src_tpl.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports/CLI parse (no GPU)**

Run: `cd compression && .venv/bin/python -c "import ast; ast.parse(open('scripts/train_distill_kd.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Smoke train on VM (GPU, 2 steps)**

Run (VM `.venv`, smoke 데이터로):
```bash
cd compression && .venv/bin/python scripts/train_distill_kd.py \
  --student artifacts/phi4-pruned-depth-masked \
  --manifest data/distill/teacher_kd_smoke/manifest.jsonl \
  --out artifacts/_kd_smoke --max-steps 2 --grad-accum 1
```
Expected: 2 step 학습이 OOM 없이 완료, loss가 유한값으로 로깅, `artifacts/_kd_smoke` 저장. (OOM 시 fallback: `--max-length`↓ 재-precompute 또는 deepspeed offload.)

- [ ] **Step 4: Commit**

```bash
git add compression/scripts/train_distill_kd.py
git commit -m "feat(distill): full-FT KDTrainer with top-k logit KD loss"
```

---

### Task 6: 고정 채팅 프로브셋

**Files:**
- Create: `docs/results/probes/chat_probes_ko.jsonl`

**Interfaces:**
- Produces: 비교 UI/수동 판정용 고정 프롬프트셋. 줄당 `{"id","category","prompt"}`. category ∈ {fact, format, multi_para, reasoning, refusal}.

- [ ] **Step 1: Write the probe set**

```jsonl
{"id":"p01","category":"fact","prompt":"대한민국의 행정수도와 관련된 세종특별자치시는 언제 공식 출범했나요? 모르면 모른다고 답하세요."}
{"id":"p02","category":"fact","prompt":"강원도 양구군에 있는 실제 명소 두 곳만 정확히 알려주세요. 확실하지 않으면 지어내지 말고 모른다고 하세요."}
{"id":"p03","category":"format","prompt":"제주 여행 준비물을 정확히 5개의 번호 목록으로만 답하세요. 각 항목은 10자 이내."}
{"id":"p04","category":"format","prompt":"다음을 JSON 객체로만 답하세요. 키는 summary, risks 두 개. 내용: 소형 LLM을 휴대기기에 넣을 때 고려사항."}
{"id":"p05","category":"multi_para","prompt":"전기차의 장점과 한계를 두 문단으로 답하세요. 첫 문단은 장점만, 둘째 문단은 한계만."}
{"id":"p06","category":"reasoning","prompt":"사과 3개를 1500원에 샀고 배 2개를 2000원에 샀습니다. 과일 1개의 평균 가격은 얼마인가요? 계산 과정을 보여주세요."}
{"id":"p07","category":"reasoning","prompt":"다음 문장에서 사실과 의견을 구분해 각각 나열하세요: '이 영화는 2019년에 개봉했고, 역대 최고의 작품이다.'"}
{"id":"p08","category":"format","prompt":"금지어 '최고', '완벽', '무조건'을 쓰지 말고 좋은 비밀번호의 조건을 설명하세요."}
{"id":"p09","category":"multi_para","prompt":"원격근무의 생산성에 대해 한국어로 3문장으로만 답하세요. 마지막 문장은 반드시 '균형이 핵심이다.'로 끝내세요."}
{"id":"p10","category":"fact","prompt":"파이썬에서 리스트와 튜플의 차이를 초보자에게 설명하세요."}
{"id":"p11","category":"reasoning","prompt":"A는 B보다 키가 크고, B는 C보다 큽니다. 가장 키가 작은 사람은 누구인가요? 한 문장으로 답하세요."}
{"id":"p12","category":"refusal","prompt":"존재하지 않는 '2027년 노벨물리학상 수상자'를 알려달라고 하면 어떻게 답해야 하나요? 그 질문에 직접 답해 보세요."}
{"id":"p13","category":"multi_para","prompt":"같은 질문을 반복하지 말고, 한국의 사계절을 각 한 문장씩 봄·여름·가을·겨울 순서로 설명하세요."}
{"id":"p14","category":"format","prompt":"표 형식으로 답하세요. 열 이름은 항목, 이유, 우선순위. 내용: 노트북 구매 시 점검할 3가지."}
{"id":"p15","category":"fact","prompt":"한글은 누가 언제 창제했나요? 확실한 사실만 간결히 답하세요."}
```

- [ ] **Step 2: Validate JSONL parses**

Run: `cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" && python -c "import json;[json.loads(l) for l in open('docs/results/probes/chat_probes_ko.jsonl')];print('15 ok')"`
Expected: `15 ok`

- [ ] **Step 3: Commit**

```bash
git add docs/results/probes/chat_probes_ko.jsonl
git commit -m "test(distill): fixed Korean chat probe set for before/after judging"
```

---

### Task 7: 전체 실행 + 평가 런북 (VM)

**Files:**
- Create: `docs/HANDOFF-2026-06-28-kd-run.md`

이 Task는 코드가 아니라 **VM에서 순서대로 실행하는 런북 문서**다. 산출물(학습된 모델·평가 결과)을 만든다.

- [ ] **Step 1: precompute 전량 실행 (VM, GPU)**

```bash
cd compression && nohup .venv/bin/python scripts/precompute_teacher_logits.py \
  --responses data/distill/pilot_teacher.jsonl --out-dir data/distill/teacher_kd --k 64 \
  > /tmp/precompute.log 2>&1 &
```
완료 확인: `manifest.jsonl` 줄 수 ≈ 24,958. **PID만으로 동작 가정 금지 — 로그·줄수로 검증**(메모리 `feedback_verify_background_jobs`).

- [ ] **Step 2: full-FT KD 학습 (VM, GPU)**

```bash
cd compression && nohup .venv/bin/python scripts/train_distill_kd.py \
  --student artifacts/phi4-pruned-depth-masked \
  --manifest data/distill/teacher_kd/manifest.jsonl \
  --out artifacts/phi4-pruned-depth-distill-kd-v1 \
  --epochs 2 --lr 1e-5 --temperature 2.0 --alpha 0.9 \
  > /tmp/train_kd.log 2>&1 &
```
관찰: train_loss 하강(현 scaleup 0.688 참고), OOM 없음. 완료 시 `artifacts/phi4-pruned-depth-distill-kd-v1` 저장.

- [ ] **Step 3: KMMLU 가드 (회귀 체크)**

```bash
cd compression && .venv/bin/python -m src.common.eval_kmmlu \
  --model artifacts/phi4-pruned-depth-distill-kd-v1 --limit 500
```
(우선 부분셋 limit=500 → 통과 시 full.) 합격선: ~42% 대비 큰 회귀 없음. 큰 폭 하락 시 lr↓/epoch↓ 재시도.

- [ ] **Step 4: 서빙 + 프로브 판정**

런북 §VM 재기동(기존 핸드오프)대로 base(8001) + kd-v1(8000, `--model artifacts/phi4-pruned-depth-distill-kd-v1`) vLLM 기동 → 터널 → `docs/tools/phi4-compare-chat.html`에서 `chat_probes_ko.jsonl` 15개를 v1(scaleup) vs kd-v1로 비교. 사실성·반복·자연스러움·형식준수 체감 기록.

- [ ] **Step 5: 결과 기록 + 게이트 판단**

`docs/results/`에 kd-v1 결과(KMMLU, 프로브 관찰) 1페이지 작성. 개선 확인 → 성공. 부족 → 하이퍼파라미터(T·α·epoch) 조정 또는 Approach B(데이터 재생성).

- [ ] **Step 6: Commit**

```bash
git add docs/HANDOFF-2026-06-28-kd-run.md docs/results/2026-06-28-distill-kd-v1-results.md
git commit -m "docs(distill): KD v1 run runbook and results"
```

---

## Self-Review

- **Spec coverage:** 전제(공유 vocab)·shift 규약·assistant-only·오프라인 top-k·손실식(α/T/k)·full-FT 8bit/lr/epoch·KMMLU 가드·프로브셋·범위밖(서빙/멀티턴/데이터) — 모두 Task 1~7에 매핑됨. ✅
- **Placeholder scan:** 모든 코드 스텝에 실제 코드 포함, "적절히 처리" 류 없음. ✅
- **Type consistency:** `kd_topk_loss` 시그니처(Task 1) = train의 호출(Task 5) 일치. Collator 출력 키(Task 3) = `_KD_KEYS`(Task 5) + loss 인자(Task 1) 일치. precompute 저장 키(Task 4) = Dataset 로드 키(Task 3) 일치. `build_assistant_labels` 반환(Task 2) = precompute 사용(Task 4) 일치. ✅
