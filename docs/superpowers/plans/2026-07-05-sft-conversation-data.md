# SFT 대화데이터 파이프라인 Implementation Plan (Plan 2/5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공인 한국어 대화/instruction 데이터셋(smol-koreantalk, KoAlpaca-RealQA, KoCommercial)을 정규 `messages` JSONL로 변환·정제·병합해, 기존 `MultiturnSFTDataset`이 바로 소비할 SFT 학습 데이터를 생성한다.

**Architecture:** 데이터셋별 raw 스키마를 정규 `{"role","content"}` messages로 바꾸는 순수 어댑터 2종(Alpaca형/messages형)을 만들고, 빌드 CLI가 streaming 로드→어댑터→정제(기존 `corpus.conversation_is_clean`)→가중 병합→JSONL 기록을 엮는다. 라벨링/토크나이즈/학습은 기존 `src/distill`(sft_data, multiturn_labels)을 그대로 재사용한다.

**Tech Stack:** Python 3.10, HuggingFace `datasets`, `pytest`, uv.

## Global Constraints

- 정규 스키마 = JSONL 한 줄당 `{"messages": [{"role": <user|assistant|system>, "content": <str>}, ...]}`. 기존 `MultiturnSFTDataset`(src/distill/sft_data.py)이 이 포맷을 소비하므로 **정확히 이 키**를 쓴다.
- **번역투 억제**: 자연스러움이 핵심(chat-v1 wash 교훈). 네이티브(KoAlpaca-RealQA) 우선, 정제 필터로 영어오염 대화 제거.
- 작업 위치 = repo `compression/`. 명령 cwd = `compression/`. 테스트 = `uv run python -m pytest`.
- **git 커밋은 VM(`phi4-blackwell`)에서만** (Mac gitdir 깨짐). 광범위 `git add -A` 금지 — 명시 경로만.
- 새 어댑터는 기존 chat-corpus 코드와 함께 `src/distill/`에 둔다(코퍼스 어댑터 `korquad_chat.py`와 동일 위치). CPT용 `src/data/`(Plan 1)와 구분.
- 외부 API 불필요(순수 변환). 토크나이즈/라벨링은 학습 시점(MultiturnSFTDataset)에 발생 — 이 플랜은 텍스트 JSONL까지만.

---

## 재사용 (신규 작성 금지 — 그대로 사용)

- `src/distill/sft_data.py` — `MultiturnSFTDataset(jsonl_path, tokenizer, max_length)` + `SFTCollator(pad_id)`. 이 JSONL을 소비.
- `src/distill/multiturn_labels.py`, `chat_labels.py` — assistant-only 라벨 마스킹(chat template 기반).
- `src/distill/corpus.py` — `conversation_is_clean(messages, threshold)`, `is_multiturn(messages)`, `weighted_merge(sources)`.
- `src/distill/ko_text.py` — `english_prose_ratio` (corpus가 사용).

## File Structure

- `compression/src/distill/sft_sources.py` — 신규 어댑터 2종 (생성)
- `compression/scripts/build_sft_data.py` — 신규 빌드 CLI (생성)
- `compression/configs/sft_data.yaml` — 소스/가중치/정제 설정 (생성)
- `compression/tests/distill/test_sft_sources.py` — 어댑터 테스트 (생성)

---

### Task 1: 어댑터 2종 (`sft_sources.py`)

**Files:**
- Create: `compression/src/distill/sft_sources.py`
- Test: `compression/tests/distill/test_sft_sources.py`

**Interfaces:**
- Produces:
  - `alpaca_to_messages(ex, instruction_field="instruction", input_field="input", output_field="output") -> list[dict] | None`
    — Alpaca형 `{instruction, input, output}` → 단일턴 messages. input 있으면 instruction 뒤에 `\n\n`로 붙임. instruction/output 중 빈 게 있으면 None.
  - `messages_to_canonical(ex, messages_field="messages") -> list[dict] | None`
    — 이미 messages형(smol-koreantalk: role/content/content_en 등)에서 유효 role + 비지 않은 content만 추출(부가필드 제거). user·assistant가 최소 1개씩 없으면 None.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_sft_sources.py
from src.distill.sft_sources import alpaca_to_messages, messages_to_canonical


def test_alpaca_basic():
    ex = {"instruction": "질문", "input": "", "output": "답"}
    assert alpaca_to_messages(ex) == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답"},
    ]


def test_alpaca_appends_input():
    ex = {"instruction": "요약해", "input": "본문", "output": "요약"}
    assert alpaca_to_messages(ex)[0]["content"] == "요약해\n\n본문"


def test_alpaca_drops_when_missing():
    assert alpaca_to_messages({"instruction": "", "input": "", "output": "답"}) is None
    assert alpaca_to_messages({"instruction": "q", "input": "", "output": ""}) is None


def test_alpaca_custom_fields():
    ex = {"q": "안녕", "a": "응"}
    out = alpaca_to_messages(ex, instruction_field="q", input_field="none", output_field="a")
    assert out == [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "응"}]


def test_messages_extracts_role_content_dropping_extras():
    ex = {"messages": [
        {"role": "user", "content": "안녕", "content_en": "hi"},
        {"role": "assistant", "content": "응", "content_en": "yes"},
    ]}
    assert messages_to_canonical(ex) == [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "응"},
    ]


def test_messages_requires_user_and_assistant():
    assert messages_to_canonical({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert messages_to_canonical({"messages": []}) is None


def test_messages_skips_blank_and_invalid_roles():
    ex = {"messages": [
        {"role": "user", "content": "  "},          # blank → skip
        {"role": "user", "content": "질문"},
        {"role": "tool", "content": "x"},            # invalid role → skip
        {"role": "assistant", "content": "답"},
    ]}
    assert messages_to_canonical(ex) == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답"},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/distill/test_sft_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.sft_sources'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/sft_sources.py
"""HF instruction/chat 데이터셋을 정규 messages 포맷으로 변환하는 어댑터."""
from __future__ import annotations

_VALID_ROLES = {"user", "assistant", "system"}


def alpaca_to_messages(
    ex: dict,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> list[dict] | None:
    """Alpaca형 {instruction, input, output} → 단일턴 messages."""
    instr = (ex.get(instruction_field) or "").strip()
    inp = (ex.get(input_field) or "").strip()
    out = (ex.get(output_field) or "").strip()
    if not instr or not out:
        return None
    user = f"{instr}\n\n{inp}" if inp else instr
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def messages_to_canonical(ex: dict, messages_field: str = "messages") -> list[dict] | None:
    """이미 messages형인 데이터에서 role/content만 추출(부가필드 제거)."""
    raw = ex.get(messages_field) or []
    msgs = []
    for m in raw:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in _VALID_ROLES and content:
            msgs.append({"role": role, "content": content})
    has_user = any(m["role"] == "user" for m in msgs)
    has_asst = any(m["role"] == "assistant" for m in msgs)
    if not (has_user and has_asst):
        return None
    return msgs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/distill/test_sft_sources.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/distill/sft_sources.py tests/distill/test_sft_sources.py
git commit -m "feat(distill): SFT 소스 어댑터 (Alpaca형/messages형 → 정규 messages)"
```

---

### Task 2: 소스 스키마 확인 (VM, 실데이터)

**Files:** 없음 (확인 전용). 결과를 Task 3의 `configs/sft_data.yaml` 필드값에 반영.

**Interfaces:** 없음.

> `beomi/KoAlpaca-RealQA`는 datasets-server 공개 조회가 401이라 실제 필드를 VM(HF 인증됨)에서 확인해야 한다. smol-koreantalk(`messages`)·KoCommercial(`instruction/input/output`)은 확인됨.

- [ ] **Step 1: 스키마 확인 스크립트 실행 (VM)**

`/tmp/sft_schema.py`:
```python
from datasets import load_dataset

for path, name in [
    ("lemon-mint/smol-koreantalk", None),
    ("beomi/KoAlpaca-RealQA", None),
    ("MarkrAI/KoCommercial-Dataset", None),
]:
    try:
        d = load_dataset(path, name, split="train", streaming=True)
        ex = next(iter(d))
        print(path, "->", list(ex.keys()))
    except Exception as e:
        print(path, "FAIL:", str(e)[:140])
```
Run: `uv run python /tmp/sft_schema.py`
Expected: 각 데이터셋의 키 출력. `beomi/KoAlpaca-RealQA`의 instruction/output 필드명을 확인해 Task 3 config의 `fields`에 기입(다르면 실제값으로 교체). smol-koreantalk에 `messages` 키, KoCommercial에 `instruction/input/output` 키 재확인.

---

### Task 3: 빌드 CLI + 설정 (`build_sft_data.py`)

**Files:**
- Create: `compression/configs/sft_data.yaml`, `compression/scripts/build_sft_data.py`

**Interfaces:**
- Consumes: `alpaca_to_messages`/`messages_to_canonical` (Task 1), `src/distill/corpus.conversation_is_clean`.
- Produces: `--out` 경로에 JSONL(한 줄당 `{"messages":[...]}`). 소스별 streaming 로드→어댑터→정제→가중 반복 기록.

- [ ] **Step 1: 설정 파일 작성** (Task 2에서 확인한 필드로)

```yaml
# compression/configs/sft_data.yaml
clean_threshold: 0.05          # 영어오염 대화 컷 (corpus.conversation_is_clean)
max_rows_per_source: null      # 소스별 상한 (null=전체, 스모크 땐 200 등)

sources:
  - path: "lemon-mint/smol-koreantalk"
    name: null
    split: "train"
    type: "messages"
    fields: {messages_field: "messages"}
    repeat: 1
  # 네이티브 앵커: beomi/KoAlpaca-RealQA (gated → HF 승인 완료, 계정 mungsik).
  # 스키마 = question/answer. VM ~/.cache/huggingface/token 에 토큰 필요.
  # (미승인 환경이면 비게이트 beomi/KoAlpaca-v1.1a, fields instruction/output 로 대체.)
  - path: "beomi/KoAlpaca-RealQA"
    name: null
    split: "train"
    type: "alpaca"
    fields: {instruction_field: "question", input_field: "none", output_field: "answer"}
    repeat: 2                            # 네이티브·소량 → 자연스러움 앵커로 가중 반복
  - path: "MarkrAI/KoCommercial-Dataset"
    name: null
    split: "train"
    type: "alpaca"
    fields: {instruction_field: "instruction", input_field: "input", output_field: "output"}
    repeat: 1
```

- [ ] **Step 2: 빌드 스크립트 작성**

```python
# compression/scripts/build_sft_data.py
"""SFT 대화데이터 빌드 CLI: HF 소스(streaming) → 어댑터 → 정제 → 가중 병합 → JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
from datasets import load_dataset

# 'src' 패키지 import 용 루트 경로 추가 (기존 스크립트와 동일)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distill.corpus import conversation_is_clean
from src.distill.sft_sources import alpaca_to_messages, messages_to_canonical

ADAPTERS = {"alpaca": alpaca_to_messages, "messages": messages_to_canonical}


def _iter_clean_convos(source: dict, max_rows: int | None, threshold: float):
    ds = load_dataset(
        source["path"], source.get("name"), split=source.get("split", "train"), streaming=True
    )
    adapt = ADAPTERS[source["type"]]
    fields = source.get("fields", {})
    for i, ex in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        msgs = adapt(ex, **fields)
        if msgs and conversation_is_clean(msgs, threshold=threshold):
            yield msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_rows = cfg.get("max_rows_per_source")
    threshold = cfg.get("clean_threshold", 0.05)

    total = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for s in cfg["sources"]:
            # repeat 위해 소스 1개분을 리스트로 유지(SFT 규모 수십만 대화 = 텍스트 수 GB,
            # 기존 MultiturnSFTDataset도 eager 적재 — 동일 기조).
            convos = list(_iter_clean_convos(s, max_rows, threshold))
            for _ in range(s.get("repeat", 1)):
                for msgs in convos:
                    w.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    total += 1
            print(f"{s['path']}: {len(convos)} clean convos x{s.get('repeat', 1)}")
    print(f"total lines -> {args.out}: {total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 스모크 실행 (VM, 소량)**

`configs/sft_data.yaml`에서 `max_rows_per_source: 200`로 임시 설정 후:
```bash
uv run python scripts/build_sft_data.py --config configs/sft_data.yaml --out /tmp/sft_smoke.jsonl
```
Expected: 각 소스별 `N clean convos xR` + `total lines` 출력. `/tmp/sft_smoke.jsonl` 생성.

- [ ] **Step 4: 산출물 검증 (VM)**

`/tmp/verify_sft.py`:
```python
import json
from transformers import AutoTokenizer
from src.distill.sft_data import MultiturnSFTDataset

# JSONL 구조 확인
n = 0
with open("/tmp/sft_smoke.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        assert set(row.keys()) == {"messages"}
        assert all("role" in m and "content" in m for m in row["messages"])
        n += 1
print("jsonl rows:", n)

# 기존 MultiturnSFTDataset이 실제로 소비 가능한지(라벨 생성)
tok = AutoTokenizer.from_pretrained("microsoft/phi-4")
dsobj = MultiturnSFTDataset("/tmp/sft_smoke.jsonl", tok, max_length=2048)
print("dataset usable rows (assistant labels 있음):", len(dsobj))
print("sample keys:", list(dsobj[0].keys()))
```
Run: `uv run python /tmp/verify_sft.py`
Expected: `jsonl rows: >0`, `dataset usable rows: >0`, `sample keys: ['input_ids', 'labels']`. 확인 후 `max_rows_per_source`를 `null`로 되돌리고 `/tmp/sft_smoke.jsonl`·`/tmp/*.py` 삭제.

- [ ] **Step 5: Commit** (VM에서)

```bash
git add configs/sft_data.yaml scripts/build_sft_data.py
git commit -m "feat(distill): SFT 데이터 빌드 CLI + 설정 (streaming, 정제, 가중병합 → JSONL)"
```

---

## Self-Review

**1. Spec coverage:** 스펙 §5 "SFT (대화)" 데이터 준비 담당 — smol-koreantalk/KoAlpaca-RealQA/KoCommercial → 정규 messages JSONL. 네이티브(KoAlpaca-RealQA) `repeat:2`로 자연스러움 앵커(§3 원칙). 라벨/학습은 기존 `src/distill` 재사용. ✅
**2. Placeholder scan:** 어댑터 코드 완전. KoAlpaca-RealQA 필드는 Task 2(실데이터 스키마 확인)로 확정 후 config 기입 — 추측 대신 확인. ✅
**3. Type consistency:** 어댑터 반환 = `list[{"role","content"}] | None`. 빌드 CLI가 None 필터 + `conversation_is_clean(messages)` 소비 + `{"messages":[...]}` JSONL 기록 → `MultiturnSFTDataset`가 `json.loads(line)["messages"]`로 소비. 체인 일관. ✅

---

## 다음 플랜 (예정)
- **Plan 3**: 평가 하니스 (KMMLU/PPL/Ko-IFEval + LogicKor·GPT judge/자연스러움/영어섞임). 기존 `chat_metrics.py`·`ifeval_verify.py`·`ko_text.py` 재사용.
- **Plan 4**: 프루닝 스윕 + CPT full-FT 학습 (Unsloth) — Plan 1 데이터 소비.
- **Plan 5**: SFT LoRA(Plan 2 데이터 소비) + GGUF 양자화 + Jetson 배포/실측.
