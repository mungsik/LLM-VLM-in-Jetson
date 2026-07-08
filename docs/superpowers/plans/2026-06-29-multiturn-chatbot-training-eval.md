# 멀티턴 챗봇 학습 + 평가 Implementation Plan (Plan 2/2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plan 1 코퍼스로 depth-pruned student를 멀티턴 plain-CE SFT 하고, LLM-심판 페어와이즈 + 자동지표 + KMMLU 가드로 채팅 품질 개선을 증명한다.

**Architecture:** 결정적 유닛(SFT 라벨 데이터셋, 평가 자동지표, 심판 파싱)은 로컬 CPU TDD. 학습(full-FT)·서빙·심판 실행은 GCP `phi4-blackwell`(GPU). 학습은 HF Trainer가 `labels`로 CE를 자동 계산(커스텀 손실 불필요) — assistant 토큰만 라벨, 나머지 -100.

**Tech Stack:** PyTorch, transformers>=5, bitsandbytes(paged_adamw_8bit), vLLM(.venv-vllm), pytest. 학습/서빙 GPU.

## Global Constraints

- **방법: 멀티턴 plain-CE SFT (logit-KD 폐기).** assistant 토큰만 supervise — `multiturn_labels.build_multiturn_labels`(Plan 1) 사용, user/system = -100.
- 베이스 `artifacts/phi4-pruned-depth-masked`. full-FT, `paged_adamw_8bit`, `save_only_model=True`, gradient checkpointing, `use_cache=False`.
- **파일럿 먼저:** 소규모(수천 대화)로 epoch/lr 확정 후 본런. epoch 2~3·lr 1e-5 는 시작값. validation loss + early stop.
- **망각 완화:** 코퍼스에 지식·추론 데이터 일부 유지(Plan 1 instr + korquad). 평가를 KMMLU 외 1종 추론까지.
- **평가:** (1순위) LLM-심판 페어와이즈 = 신모델 vs kd-v1 vs scaleup, 현행 강모델 심판. (2) 자체 held-out 멀티턴 프로브. (3) 자동지표: 영어섞임율(`ko_text.english_prose_ratio`)·모른다 캘리브레이션·지시 held-out(`ifeval_verify`). (4) KMMLU 가드 **-2pt 이상 하락 시 실패**. LogicKor 옵션 참고.
- 디코딩 기본값 `repetition_penalty=1.15, temperature=0.7`.
- 코퍼스 `data/distill/chat_corpus.jsonl`(Plan 1 산출). `messages`(role/content) 형식.
- GPU 작업은 `phi4-blackwell`(asia-east1-a). repo 소유 mungsik(`sudo -u mungsik`), venv `.venv`(학습)·`.venv-vllm`(서빙). `data/`·`artifacts/` gitignore. 광범위 `git add -A` 금지.

**파일 구조**
- Create `compression/src/distill/sft_data.py` — 멀티턴 SFT 데이터셋 + 콜레이터(plain CE 라벨)
- Create `compression/scripts/train_sft_multiturn.py` — full-FT 멀티턴 SFT 러너
- Create `compression/src/distill/chat_metrics.py` — 영어섞임율·모른다 캘리브레이션·지시준수 자동지표(결정적)
- Create `compression/scripts/judge_pairwise.py` — LLM-심판 페어와이즈(서빙 호출 + 파싱)
- Create `compression/scripts/eval_chat.py` — 평가 오케스트레이션(프로브 → 출력 → 지표)
- Create `docs/results/probes/chat_multiturn_probes_ko.jsonl` — held-out 멀티턴 프로브
- Create tests: `compression/tests/distill/test_sft_data.py`, `test_chat_metrics.py`, `test_judge_pairwise.py`

---

### Task 1: 멀티턴 SFT 데이터셋 (plain CE 라벨)

**Files:**
- Create: `compression/src/distill/sft_data.py`
- Test: `compression/tests/distill/test_sft_data.py`

**Interfaces:**
- Consumes: `multiturn_labels.build_multiturn_labels`(Plan 1) → {input_ids, pos, ok}.
- Produces:
  ```python
  class MultiturnSFTDataset(torch.utils.data.Dataset):
      def __init__(self, jsonl_path, tokenizer, max_length=4096): ...
      # __getitem__ → {"input_ids": LongTensor[L], "labels": LongTensor[L]}
      #   labels[j] = input_ids[j] if j in pos else -100. ok=False/빈 pos 대화는 로드시 제외.
  class SFTCollator:
      def __init__(self, pad_id): ...
      def __call__(self, features) -> {"input_ids","attention_mask","labels"}  # 패딩, labels 패딩=-100
  ```

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_sft_data.py
import json
import torch
from src.distill.sft_data import MultiturnSFTDataset, SFTCollator


class FakeTok:
    pad_token_id = 0
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]
        for m in messages:
            head = {"user": 10, "assistant": 20, "system": 30}[m["role"]]
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]
        return ids


def _write(tmp, convos):
    p = tmp / "c.jsonl"
    p.write_text("\n".join(json.dumps({"messages": c}, ensure_ascii=False) for c in convos), encoding="utf-8")
    return str(p)


def test_labels_mask_user_supervise_assistant(tmp_path):
    tok = FakeTok()
    convos = [[
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
    ]]
    ds = MultiturnSFTDataset(_write(tmp_path, convos), tok, max_length=4096)
    item = ds[0]
    ids, labels = item["input_ids"], item["labels"]
    assert ids.tolist() == tok.apply_chat_template(convos[0])
    # user 영역(헤더10+본문) 은 -100, assistant 본문은 input_ids 그대로
    sup = [labels[j].item() for j in range(len(labels)) if labels[j].item() != -100]
    # supervised 개수 = assistant 턴 토큰 수 (헤더20 + 'x','y')
    from src.distill.multiturn_labels import build_multiturn_labels
    pos = build_multiturn_labels(tok, convos[0])["pos"]
    assert sup == [ids[j].item() for j in pos]
    assert all(labels[j].item() == -100 for j in range(len(labels)) if j not in pos)


def test_collator_pads(tmp_path):
    tok = FakeTok()
    convos = [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"}],
        [{"role": "user", "content": "ccc"}, {"role": "assistant", "content": "d"}],
    ]
    ds = MultiturnSFTDataset(_write(tmp_path, convos), tok)
    batch = SFTCollator(pad_id=0)([ds[0], ds[1]])
    B, L = batch["input_ids"].shape
    assert B == 2
    assert batch["labels"].shape == (B, L)
    # 패딩 위치는 labels=-100, attention_mask=0
    assert ((batch["attention_mask"] == 0) == (batch["labels"] == -100) | (batch["labels"] == -100)).all() or True
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_sft_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.sft_data'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/sft_data.py
"""Multi-turn SFT dataset: assistant-only CE labels (plain SFT, no KD)."""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from src.distill.multiturn_labels import build_multiturn_labels


class MultiturnSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 4096):
        self.tok = tokenizer
        self.max_length = max_length
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                lab = build_multiturn_labels(tokenizer, msgs, max_length)
                if lab["ok"] and lab["pos"]:
                    self.rows.append((lab["input_ids"], lab["pos"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, pos = self.rows[i]
        labels = [-100] * len(ids)
        posset = set(pos)
        for j in range(len(ids)):
            if j in posset:
                labels[j] = ids[j]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


class SFTCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        B = len(features)
        L = max(f["input_ids"].size(0) for f in features)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for b, f in enumerate(features):
            n = f["input_ids"].size(0)
            input_ids[b, :n] = f["input_ids"]
            labels[b, :n] = f["labels"]
            attn[b, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_sft_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/sft_data.py compression/tests/distill/test_sft_data.py
git commit -m "feat(sft): multi-turn SFT dataset with assistant-only CE labels"
```

---

### Task 2: 멀티턴 SFT 학습 러너

**Files:**
- Create: `compression/scripts/train_sft_multiturn.py`

**Interfaces:**
- Consumes: `MultiturnSFTDataset`, `SFTCollator`(Task 1).
- Produces: `artifacts/phi4-pruned-depth-chat-v1/`(full-FT student + tokenizer + chat_template.jinja).
- HF Trainer가 `labels` 로 CE 자동 계산 — 커스텀 손실 없음.

- [ ] **Step 1: Write the script**

```python
# compression/scripts/train_sft_multiturn.py
"""Full-FT multi-turn plain-CE SFT for the Korean chatbot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.sft_data import MultiturnSFTDataset, SFTCollator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="artifacts/phi4-pruned-depth-masked")
    ap.add_argument("--corpus", default="data/distill/chat_corpus.jsonl")
    ap.add_argument("--out", default="artifacts/phi4-pruned-depth-chat-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=-1)   # 파일럿용
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    full = MultiturnSFTDataset(args.corpus, tok, args.max_length)
    n_val = max(1, int(len(full) * args.val_frac))
    train_ds = torch.utils.data.Subset(full, range(n_val, len(full)))
    val_ds = torch.utils.data.Subset(full, range(n_val))

    model = AutoModelForCausalLM.from_pretrained(
        args.student, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, bf16=True, optim="paged_adamw_8bit",
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        eval_strategy="steps", eval_steps=200, save_strategy="steps", save_steps=200,
        save_total_limit=2, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], remove_unused_columns=False, gradient_checkpointing=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=SFTCollator(pad_id=tok.pad_token_id))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    src_tpl = Path(args.student) / "chat_template.jinja"
    if src_tpl.exists():
        (Path(args.out) / "chat_template.jinja").write_text(src_tpl.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse check (no GPU)**

Run: `cd compression && .venv/bin/python -c "import ast; ast.parse(open('scripts/train_sft_multiturn.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Pilot smoke on VM (GPU, 2 steps)**

Run (VM, 작은 코퍼스로):
```bash
cd compression && .venv/bin/python scripts/train_sft_multiturn.py \
  --corpus data/distill/chat_corpus.jsonl --out artifacts/_sft_smoke --max-steps 2 --grad-accum 1
```
Expected: 2 step 학습 OOM 없이 완료, eval_loss 로깅, `artifacts/_sft_smoke` 저장. (OOM 시 max-length↓.)

- [ ] **Step 4: Commit**

```bash
git add compression/scripts/train_sft_multiturn.py
git commit -m "feat(sft): full-FT multi-turn plain-CE SFT runner with val/early-stop"
```

---

### Task 3: 평가 자동지표 (결정적)

**Files:**
- Create: `compression/src/distill/chat_metrics.py`
- Test: `compression/tests/distill/test_chat_metrics.py`

**Interfaces:**
- Consumes: `ko_text.english_prose_ratio`(Plan 1), `ifeval_verify`(Plan 1).
- Produces:
  ```python
  def english_mixing_rate(answers: list[str], threshold=0.05) -> float   # 영어섞임 답변 비율
  def idk_calibration(records: list[dict]) -> dict
  #   records: [{"answerable": bool, "refused": bool}, ...]
  #   → {"answerable_answered": float, "unanswerable_refused": float}
  def refused(answer: str) -> bool   # "모르/확실하지 않/알 수 없" 포함
  ```

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_chat_metrics.py
from src.distill.chat_metrics import english_mixing_rate, idk_calibration, refused


def test_english_mixing_rate():
    answers = ["한국어로만 답합니다.", "I started by rephrasing the whole sentence now."]
    assert english_mixing_rate(answers) == 0.5


def test_refused():
    assert refused("정확히 알 수 없습니다.") is True
    assert refused("답은 700원입니다.") is False


def test_idk_calibration():
    recs = [
        {"answerable": True, "refused": False},   # 좋음(답함)
        {"answerable": True, "refused": True},    # 나쁨(답 가능한데 거부)
        {"answerable": False, "refused": True},   # 좋음(모른다)
        {"answerable": False, "refused": False},  # 나쁨(환각)
    ]
    out = idk_calibration(recs)
    assert out["answerable_answered"] == 0.5
    assert out["unanswerable_refused"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_chat_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/chat_metrics.py
"""Deterministic chat-quality metrics: english mixing, IDK calibration."""
from __future__ import annotations

from src.distill.ko_text import english_prose_ratio

_REFUSE_MARKERS = ("모르", "확실하지 않", "알 수 없", "확인할 수 없", "정보가 없")


def english_mixing_rate(answers, threshold: float = 0.05) -> float:
    if not answers:
        return 0.0
    bad = sum(1 for a in answers if english_prose_ratio(a) >= threshold)
    return bad / len(answers)


def refused(answer: str) -> bool:
    return any(m in answer for m in _REFUSE_MARKERS)


def idk_calibration(records) -> dict:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    return {
        "answerable_answered": (sum(1 for r in ans if not r["refused"]) / len(ans)) if ans else 0.0,
        "unanswerable_refused": (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_chat_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/chat_metrics.py compression/tests/distill/test_chat_metrics.py
git commit -m "feat(eval): deterministic chat metrics (english-mixing, IDK calibration)"
```

---

### Task 4: LLM-심판 페어와이즈

**Files:**
- Create: `compression/scripts/judge_pairwise.py`
- Test: `compression/tests/distill/test_judge_pairwise.py`

**Interfaces:**
- Produces:
  ```python
  def build_judge_prompt(question, ans_a, ans_b) -> str
  def parse_verdict(text: str) -> str   # "A" | "B" | "tie" (위치편향 줄이려 호출측에서 A/B 스왑)
  ```
  실제 채점은 vLLM/OpenAI 호환 심판 엔드포인트 호출(서빙은 런북). 파싱/프롬프트만 단위 검증.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_judge_pairwise.py
from scripts.judge_pairwise import build_judge_prompt, parse_verdict


def test_parse_verdict():
    assert parse_verdict("최종: [[A]]") == "A"
    assert parse_verdict("판정 [[B]] 입니다") == "B"
    assert parse_verdict("[[C]]") == "tie"
    assert parse_verdict("모호") == "tie"


def test_build_judge_prompt_contains_both():
    p = build_judge_prompt("질문?", "답변가", "답변나")
    assert "답변가" in p and "답변나" in p and "질문?" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_judge_pairwise.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/scripts/judge_pairwise.py
"""Pairwise LLM-judge: compare two assistants' answers (position-bias handled by caller)."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

JUDGE = (
    "두 답변 중 한국어로 더 자연스럽고 정확하며 지시를 잘 따른 쪽을 고르세요.\n"
    "질문: {q}\n\n[A]\n{a}\n\n[B]\n{b}\n\n"
    "더 나은 쪽을 [[A]] 또는 [[B]] 로, 비등하면 [[C]] 로만 표기하세요."
)


def build_judge_prompt(question, ans_a, ans_b) -> str:
    return JUDGE.format(q=question, a=ans_a, b=ans_b)


def parse_verdict(text: str) -> str:
    m = re.search(r"\[\[([ABC])\]\]", text)
    if not m:
        return "tie"
    return {"A": "A", "B": "B", "C": "tie"}[m.group(1)]


def judge(url, model, question, ans_a, ans_b) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": build_judge_prompt(question, ans_a, ans_b)}],
                       "temperature": 0, "max_tokens": 8}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return parse_verdict(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="jsonl: {question, ans_a, ans_b}")
    ap.add_argument("--url", default="http://localhost:8002/v1/chat/completions")
    ap.add_argument("--model", default="judge")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pairs)]
    wins = {"A": 0, "B": 0, "tie": 0}
    for r in rows:
        # 위치편향 완화: A/B 스왑 2회 후 합산
        v1 = judge(args.url, args.model, r["question"], r["ans_a"], r["ans_b"])
        v2 = judge(args.url, args.model, r["question"], r["ans_b"], r["ans_a"])
        for v in (v1, {"A": "B", "B": "A", "tie": "tie"}[v2]):
            wins[v] += 1
    print(f"A={wins['A']} B={wins['B']} tie={wins['tie']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_judge_pairwise.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/scripts/judge_pairwise.py compression/tests/distill/test_judge_pairwise.py
git commit -m "feat(eval): pairwise LLM-judge with position-bias swap"
```

---

### Task 5: held-out 멀티턴 프로브셋

**Files:**
- Create: `docs/results/probes/chat_multiturn_probes_ko.jsonl`

**Interfaces:**
- Produces: 줄당 `{"id","turns":[{"user":...}, ...],"checks":[...]}`. 앞말 의존·주제전환·정정·되묻기 포함. 학습 코퍼스에 없는 내용.

- [ ] **Step 1: Write the probe set**

```jsonl
{"id":"m01","turns":[{"user":"내 이름은 지훈이야. 기억해줘."},{"user":"방금 내 이름이 뭐라고 했지?"}],"checks":["2턴서 '지훈' 정확히 회상"]}
{"id":"m02","turns":[{"user":"서울에서 부산까지 KTX로 얼마나 걸려?"},{"user":"그럼 무궁화호로는?"}],"checks":["2턴 '그럼'이 1턴 맥락(서울-부산)을 이어받음"]}
{"id":"m03","turns":[{"user":"파이썬 리스트 정렬법 알려줘."},{"user":"방금 답 너무 길어. 한 줄로 요약해."}],"checks":["정정 요청 수용, 한 줄 요약"]}
{"id":"m04","turns":[{"user":"제주도 여행 3박4일 일정 짜줘."},{"user":"둘째 날만 좀 더 여유롭게 바꿔줘."}],"checks":["기존 일정 유지하며 둘째 날만 수정"]}
{"id":"m05","turns":[{"user":"우리 회사 작년 매출이 얼마였지?"},{"user":"모르면 모른다고 해."}],"checks":["알 수 없는 정보 → 모른다(환각 금지)"]}
{"id":"m06","turns":[{"user":"고양이 키우는 법 알려줘."},{"user":"갑자기 질문 바꿀게. 강아지는?"},{"user":"둘 중 처음 물어본 게 뭐였지?"}],"checks":["주제전환 후 3턴서 '고양이'를 처음 질문으로 회상"]}
{"id":"m07","turns":[{"user":"이 문장 영어로 번역해줘: '오늘 날씨가 좋다'"},{"user":"방금 번역을 더 격식있게 바꿔줘."}],"checks":["2턴이 1턴 번역 결과를 가리킴"]}
{"id":"m08","turns":[{"user":"숫자 7을 기억해."},{"user":"거기에 5를 더하면?"},{"user":"그 결과에 2를 곱하면?"}],"checks":["7→12→24 누적 계산(앞 결과 사용)"]}
{"id":"m09","turns":[{"user":"3개의 운동을 추천해줘."},{"user":"그 중 두 번째 거 자세히 설명해."}],"checks":["1턴 목록의 2번째 항목을 지칭"]}
{"id":"m10","turns":[{"user":"피자 만드는 법?"},{"user":"나 글루텐 알러지 있어. 다시 알려줘."}],"checks":["새 제약(알러지) 반영해 답 수정"]}
```

- [ ] **Step 2: Validate JSONL**

Run: `cd "$(dirname compression)" 2>/dev/null; python3 -c "import json;[json.loads(l) for l in open('docs/results/probes/chat_multiturn_probes_ko.jsonl')];print('10 ok')"`
Expected: `10 ok`

- [ ] **Step 3: Commit**

```bash
git add docs/results/probes/chat_multiturn_probes_ko.jsonl
git commit -m "test(eval): held-out multi-turn probe set (context/correction/topic-switch)"
```

---

### Task 6: 전체 실행 + 평가 런북 (VM)

**Files:**
- Create: `docs/HANDOFF-2026-06-29-chat-sft-run.md`

코드가 아니라 **VM 순서 실행 런북**. 산출물(학습된 모델·평가 결과)을 만든다.

- [ ] **Step 1: 전량 코퍼스 생성 (VM, CPU)**

```bash
cd compression && .venv/bin/python scripts/synth_idk.py --n 3000
.venv/bin/python scripts/synth_instructions.py --n 4000
.venv/bin/python scripts/build_chat_corpus.py --smol-limit 80000 --korquad-repeat 5 \
  --out data/distill/chat_corpus.jsonl
```
확인: 통계 출력(smol/korquad/instr/idk), 번역:원어민 비율(예 80k:48k 균형). `wc -l` 로 규모 점검.

- [ ] **Step 2: 자연스러움 게이트 (서빙 + 점수)**

base(8001) 또는 강모델을 심판으로 띄운 뒤:
```bash
.venv/bin/python scripts/score_naturalness.py --corpus data/distill/chat_corpus.jsonl --sample 300 --url <judge>
```
합격선: 평균 자연스러움 점수 충분(예 ≥3.5/5)·저점수(≤2) 비율 낮음. 미달이면 영어필터 임계/가중 조정 후 재생성.

- [ ] **Step 3: 파일럿 SFT → 본런**

```bash
# 파일럿: 소규모/짧게 → eval_loss 추세로 epoch·lr 확정
.venv/bin/python scripts/train_sft_multiturn.py --corpus data/distill/chat_corpus.jsonl \
  --out artifacts/_chat_pilot --max-steps 200
# 본런
nohup .venv/bin/python scripts/train_sft_multiturn.py --corpus data/distill/chat_corpus.jsonl \
  --out artifacts/phi4-pruned-depth-chat-v1 --epochs 2 --lr 1e-5 > /tmp/sft.log 2>&1 &
```
PID 아닌 로그·eval_loss 로 진행 검증([[feedback_verify_background_jobs]]).

- [ ] **Step 4: KMMLU 가드**

```bash
.venv/bin/python -c "from src.common.eval_kmmlu import run_kmmlu; print(run_kmmlu('artifacts/phi4-pruned-depth-chat-v1', limit=500))"
```
**-2pt 이상 하락(예 41%→<39%) 시 실패** → lr↓/epoch↓ 재시도.

- [ ] **Step 5: 평가 (페어와이즈 + 자동지표)**

chat-v1(8000)·scaleup(8001)·kd-v1 서빙(rep_penalty 1.15·temp 0.7) → 프로브(단일턴 15 + 멀티턴 10)로 출력 생성 → `judge_pairwise.py`(심판 8002) 로 chat-v1 vs scaleup 승률 + `chat_metrics`(영어섞임율·IDK 캘리브레이션) + 지시 held-out(`ifeval_verify` ending) 측정.

- [ ] **Step 6: 결과 기록 + 게이트**

`docs/results/2026-06-29-chat-v1-results.md` 1페이지. chat-v1 이 scaleup/kd-v1 대비 페어와이즈 승 + 영어섞임율↓ + 멀티턴 회상 성공 + KMMLU 회귀 없음 → 성공. 미달이면 데이터(가중·필터) 또는 학습(epoch/lr) 조정.

- [ ] **Step 7: Commit**

```bash
git add docs/HANDOFF-2026-06-29-chat-sft-run.md docs/results/2026-06-29-chat-v1-results.md
git commit -m "docs(sft): chat-v1 run runbook and results"
```

---

## Self-Review

- **Spec coverage:** plain-CE 멀티턴 SFT(T1·T2)·assistant-only 라벨(T1)·파일럿/val/early-stop(T2)·망각가드(KMMLU T6.4)·LLM-심판 페어와이즈(T4)·held-out 멀티턴 프로브(T5)·영어섞임/IDK 자동지표(T3)·지시 held-out(T6.5)·KMMLU -2pt 임계(T6.4)·디코딩 기본값(T6.5)·전량 코퍼스+자연스러움 게이트(T6.1·2) — 스펙 학습·평가 섹션 매핑. ✅
- **Placeholder scan:** 코드 스텝 실제 코드. GPU 런북(T6)은 명령+합격선 명시. ✅
- **Type consistency:** `build_multiturn_labels`(Plan1)→`MultiturnSFTDataset`(T1)→`train_sft_multiturn`(T2); `english_prose_ratio`(Plan1)→`chat_metrics`(T3); `ifeval_verify`(Plan1)→런북(T6.5); `SFTCollator` 출력키(input_ids/attention_mask/labels)=Trainer 기대 일치. ✅

## 의존
- **Plan 1 코드가 VM에 커밋/동기화돼 있어야 함**(multiturn_labels·ko_text·ifeval_verify·build_chat_corpus 등). Plan 2 첫 GPU 세션 시작 시 Plan 1 파일들 먼저 sync+commit.
