# 멀티턴 챗봇 데이터 파이프라인 Implementation Plan (Plan 1/2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 멀티턴 한국어 챗봇 학습용 코퍼스를 기성 데이터(smol-koreantalk, korquad-chat) 정제 + 합성(지시·모른다)으로 만들어 단일 `messages` jsonl로 출력한다.

**Architecture:** 전부 CPU. 결정적(deterministic) 유닛은 TDD — 멀티턴 라벨 마스킹, 영어산문 판별, korquad-chat 파서, 형식검증기, 코퍼스 빌더. LLM 의존(자연스러움 점수·합성 답 생성)은 얇은 경계로 두고 주변 파싱/집계만 단위테스트. 산출물은 학습(Plan 2) 입력.

**Tech Stack:** Python 3.11, `datasets`(스트리밍), pytest. transformers는 라벨 마스킹 테스트에 가짜 토크나이저 사용(실모델 불필요).

## Global Constraints

- 출력 포맷: `messages` = `[{"role": "user"|"assistant"|"system", "content": str}, ...]`. 학습은 **assistant 턴만** supervise.
- 영어 필터는 **영어 산문**만 컷 — 코드/JSON 키/URL/대문자약어/제품·모델명 같은 허용 라틴 토큰은 제외하고 센다.
- 코퍼스 가중: **번역(smol):원어민(korquad-chat) 비율을 명시 관리**, korquad-chat 업샘플링.
- "모른다" 데이터 비율 ≤ 전체의 5~10%, answerable 짝 + 되묻기 포함.
- 지시·형식 데이터: 답은 **프로그램 검증 통과분만** 채택(형식 준수 보장, 내용품질은 별도).
- 데이터 작업은 CPU(GPU VM 불필요). 로컬/VM 무관, venv에 `datasets` 필요.
- 결정적 로직은 LLM/네트워크 없이 단위테스트 가능해야 한다.

**파일 구조**
- Create `compression/src/distill/multiturn_labels.py` — 멀티턴 assistant 마스킹
- Create `compression/src/distill/ko_text.py` — 영어산문 비율(허용토큰 제외)
- Create `compression/src/distill/korquad_chat.py` — `<sys>/<usr>/<bot>` 파서
- Create `compression/src/distill/corpus.py` — 정규화·필터·가중·병합 순수함수
- Create `compression/src/distill/ifeval_verify.py` — 형식 검증기(목록/JSON/금지어/끝문장)
- Create `compression/scripts/synth_instructions.py` — 지시·형식 합성(검증기 사용)
- Create `compression/scripts/synth_idk.py` — 모른다 합성(answerable 짝 포함)
- Create `compression/scripts/score_naturalness.py` — LLM 심사 자연스러움 점수(경계)
- Create `compression/scripts/build_chat_corpus.py` — 파이프라인 실행(통합)
- Create tests: `compression/tests/distill/test_multiturn_labels.py`, `test_ko_text.py`, `test_korquad_chat.py`, `test_corpus.py`, `test_ifeval_verify.py`

---

### Task 1: 멀티턴 assistant 라벨 마스킹

**Files:**
- Create: `compression/src/distill/multiturn_labels.py`
- Test: `compression/tests/distill/test_multiturn_labels.py`

**Interfaces:**
- Consumes: `chat_labels._render_ids`(기존, transformers 5.x BatchEncoding 처리).
- Produces:
  ```python
  def build_multiturn_labels(tokenizer, messages, max_length=4096) -> dict
  # {"input_ids": list[int], "pos": list[int], "ok": bool}
  #   pos = 모든 assistant 턴의 토큰 인덱스(예측대상). user/system 은 제외.
  #   각 assistant 턴 i 의 스팬 = render(messages[:i]+gen_prompt) .. render(messages[:i+1]).
  #   중간 truncation 으로 잘린 부분 턴은 제외. prefix 불일치 시 ok=False, pos=[].
  ```

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_multiturn_labels.py
from src.distill.multiturn_labels import build_multiturn_labels


class FakeTok:
    """role 헤더 토큰을 정수로 흉내내는 최소 토크나이저. user=10, assistant=20, system=30."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]  # BOS
        for m in messages:
            head = {"user": 10, "assistant": 20, "system": 30}[m["role"]]
            ids += [head] + [ord(c) % 50 + 100 for c in m["content"]]
        if add_generation_prompt:
            ids += [20]  # assistant 헤더만
        return ids


def _roles_at(messages, input_ids, pos):
    # pos 위치 토큰들이 실제 assistant 본문(>=100)인지 헤더(20) 직후인지 확인용
    return [input_ids[p] for p in pos]


def test_two_assistant_turns_both_supervised():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "xy"},
        {"role": "user", "content": "cd"},
        {"role": "assistant", "content": "z"},
    ]
    out = build_multiturn_labels(tok, msgs, max_length=4096)
    assert out["ok"] is True
    full = tok.apply_chat_template(msgs)
    # 첫 assistant 스팬: prefix1 = render([u1]+gen) 이후 ~ render([u1,a1]) 까지
    p1 = tok.apply_chat_template(msgs[:1], add_generation_prompt=True)
    e1 = tok.apply_chat_template(msgs[:2])
    p2 = tok.apply_chat_template(msgs[:3], add_generation_prompt=True)
    e2 = tok.apply_chat_template(msgs[:4])
    expected = [j for j in range(len(p1), len(e1))] + [j for j in range(len(p2), len(e2))]
    assert out["pos"] == expected
    # user 본문 토큰(ord('c')->..)은 pos에 없어야
    assert all(full[p] != 10 for p in out["pos"])


def test_system_turn_and_user_ending_handled():
    tok = FakeTok()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},   # user 로 끝남(미완성)
    ]
    out = build_multiturn_labels(tok, msgs, max_length=4096)
    # assistant 턴 1개만 supervise, 마지막 user 턴은 무시
    p1 = tok.apply_chat_template(msgs[:2], add_generation_prompt=True)
    e1 = tok.apply_chat_template(msgs[:3])
    assert out["pos"] == [j for j in range(len(p1), len(e1))]


def test_mid_turn_truncation_drops_partial_turn():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "xxxxxxxx"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "yyyyyyyy"},
    ]
    # 두 번째 assistant 턴이 시작되기 전에서 자르면 그 턴은 통째 제외
    p2 = tok.apply_chat_template(msgs[:3], add_generation_prompt=True)
    out = build_multiturn_labels(tok, msgs, max_length=len(p2) + 2)
    assert all(p < len(p2) + 2 for p in out["pos"])
    # 첫 턴 일부는 남고, 두 번째 턴 본문은 잘려 없음
    assert len(out["pos"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_multiturn_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.multiturn_labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/multiturn_labels.py
"""Mark supervised token positions for every assistant turn (multi-turn)."""
from __future__ import annotations

from src.distill.chat_labels import _render_ids


def build_multiturn_labels(tokenizer, messages, max_length: int = 4096) -> dict:
    full = _render_ids(tokenizer, messages, add_generation_prompt=False)[:max_length]
    pos: list[int] = []
    ok = True
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = _render_ids(tokenizer, messages[:i], add_generation_prompt=True)
        end = _render_ids(tokenizer, messages[: i + 1], add_generation_prompt=False)
        # 정렬 가드: prefix/end 가 full 의 실제 접두부인지
        if full[: len(prefix)] != prefix[: min(len(prefix), len(full))]:
            ok = False
            continue
        start = len(prefix)
        stop = min(len(end), len(full))   # 중간 truncation 반영
        pos.extend(j for j in range(start, stop) if j >= 1)
    return {"input_ids": full, "pos": pos, "ok": ok}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_multiturn_labels.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/multiturn_labels.py compression/tests/distill/test_multiturn_labels.py
git commit -m "feat(corpus): multi-turn assistant label masking"
```

---

### Task 2: 영어 산문 비율(허용 토큰 제외)

**Files:**
- Create: `compression/src/distill/ko_text.py`
- Test: `compression/tests/distill/test_ko_text.py`

**Interfaces:**
- Produces:
  ```python
  def english_prose_ratio(text: str) -> float
  # 허용 라틴(코드블록 ```...```, 인라인 `code`, URL, 대문자약어 IEEE/JSON/CPU, 제품·모델명)을
  # 제거한 뒤, 남은 라틴 글자 / 전체 비공백 글자. 영어 "문장"이 섞일수록 높아짐.
  ```
  허용 토큰만 있는 답(예: "리스트는 `append`로 추가합니다")은 ~0, 영어 문장 섞인 답은 높게.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_ko_text.py
from src.distill.ko_text import english_prose_ratio


def test_pure_korean_is_zero():
    assert english_prose_ratio("리스트는 수정 가능한 자료구조입니다.") == 0.0


def test_allowed_tokens_not_counted():
    # 코드/약어/URL 은 영어로 안 침
    t = "파이썬에서 `append`를 쓰거나 CPU 정보를 https://x.io 에서 봅니다."
    assert english_prose_ratio(t) < 0.05


def test_english_sentence_is_high():
    t = "리스트는 mutable. I started by rephrasing the first sentence to make it clearer."
    assert english_prose_ratio(t) > 0.3


def test_empty_is_zero():
    assert english_prose_ratio("") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_ko_text.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill.ko_text'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/ko_text.py
"""Estimate English-prose contamination, excluding allowed Latin tokens."""
from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_URL = re.compile(r"https?://\S+")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")          # CPU, JSON, IEEE, GPU
_PRODUCTish = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")        # Python, Zigbee, Mutable(고유명사풍)


def english_prose_ratio(text: str) -> float:
    if not text:
        return 0.0
    t = _CODE_BLOCK.sub(" ", text)
    t = _INLINE_CODE.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _ACRONYM.sub(" ", t)
    t = _PRODUCTish.sub(" ", t)
    latin = len(re.findall(r"[A-Za-z]", t))
    nonspace = len(re.findall(r"\S", text)) or 1
    return latin / nonspace
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_ko_text.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/ko_text.py compression/tests/distill/test_ko_text.py
git commit -m "feat(corpus): english-prose ratio excluding allowed latin tokens"
```

---

### Task 3: korquad-chat-v1 파서

**Files:**
- Create: `compression/src/distill/korquad_chat.py`
- Test: `compression/tests/distill/test_korquad_chat.py`

**Interfaces:**
- Produces:
  ```python
  def parse_korquad_chat(text: str) -> dict
  # 입력: "<sys>지문...\n<usr>질문\n<bot>답\n<usr>...\n<bot>..." 형태의 한 text 필드.
  # 출력: {"system": str, "messages": [{"role","content"}, ...]}  (usr->user, bot->assistant)
  ```

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_korquad_chat.py
from src.distill.korquad_chat import parse_korquad_chat


def test_parses_sys_usr_bot():
    text = "<sys>바그너는 1839년 파우스트를 읽었다.\n<usr>왜 끌렸나요?\n<bot>심경에 공감해서요.\n<usr>그래서요?\n<bot>교향곡을 구상했어요."
    out = parse_korquad_chat(text)
    assert out["system"].startswith("바그너는")
    assert out["messages"] == [
        {"role": "user", "content": "왜 끌렸나요?"},
        {"role": "assistant", "content": "심경에 공감해서요."},
        {"role": "user", "content": "그래서요?"},
        {"role": "assistant", "content": "교향곡을 구상했어요."},
    ]


def test_missing_sys_returns_empty_system():
    text = "<usr>안녕\n<bot>네 안녕하세요"
    out = parse_korquad_chat(text)
    assert out["system"] == ""
    assert out["messages"][0] == {"role": "user", "content": "안녕"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_korquad_chat.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/korquad_chat.py
"""Parse heegyu/korquad-chat-v1 <sys>/<usr>/<bot> text into messages."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"<(sys|usr|bot)>(.*?)(?=<(?:sys|usr|bot)>|$)", re.S)
_ROLE = {"usr": "user", "bot": "assistant"}


def parse_korquad_chat(text: str) -> dict:
    system = ""
    messages = []
    for tag, body in _TOKEN.findall(text):
        body = body.strip()
        if tag == "sys":
            system = body
        else:
            messages.append({"role": _ROLE[tag], "content": body})
    return {"system": system, "messages": messages}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_korquad_chat.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/korquad_chat.py compression/tests/distill/test_korquad_chat.py
git commit -m "feat(corpus): korquad-chat <sys>/<usr>/<bot> parser"
```

---

### Task 4: 코퍼스 정제·가중·병합 순수함수

**Files:**
- Create: `compression/src/distill/corpus.py`
- Test: `compression/tests/distill/test_corpus.py`

**Interfaces:**
- Consumes: `ko_text.english_prose_ratio`(Task 2).
- Produces:
  ```python
  def conversation_is_clean(messages, threshold=0.05) -> bool
  #   모든 assistant 턴의 english_prose_ratio < threshold 면 True.
  def is_multiturn(messages) -> bool   # assistant 턴 >= 2
  def weighted_merge(sources: list[dict]) -> list[dict]
  #   sources: [{"name","convos":[messages,...],"repeat":int}, ...]
  #   각 소스를 repeat 배 복제해 합침(원어민 업샘플링용). 반환: messages 리스트.
  ```

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_corpus.py
from src.distill.corpus import conversation_is_clean, is_multiturn, weighted_merge


def _conv(*pairs):
    out = []
    for u, a in pairs:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def test_clean_passes_dirty_fails():
    clean = _conv(("질문", "한국어로만 답합니다."))
    dirty = _conv(("질문", "I started by rephrasing the sentence to make it clearer."))
    assert conversation_is_clean(clean) is True
    assert conversation_is_clean(dirty) is False


def test_is_multiturn():
    assert is_multiturn(_conv(("a", "b"))) is False           # assistant 1개
    assert is_multiturn(_conv(("a", "b"), ("c", "d"))) is True  # assistant 2개


def test_weighted_merge_upsamples():
    native = _conv(("원", "어민"))
    trans = _conv(("번", "역"))
    merged = weighted_merge([
        {"name": "native", "convos": [native], "repeat": 3},
        {"name": "trans", "convos": [trans], "repeat": 1},
    ])
    assert len(merged) == 4          # 3 + 1
    assert merged.count(native) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/corpus.py
"""Pure filtering / weighting / merge helpers for the chat corpus."""
from __future__ import annotations

from src.distill.ko_text import english_prose_ratio


def conversation_is_clean(messages, threshold: float = 0.05) -> bool:
    asst = [m["content"] for m in messages if m["role"] == "assistant"]
    return all(english_prose_ratio(c) < threshold for c in asst)


def is_multiturn(messages) -> bool:
    return sum(1 for m in messages if m["role"] == "assistant") >= 2


def weighted_merge(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        for _ in range(s["repeat"]):
            out.extend(s["convos"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_corpus.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/corpus.py compression/tests/distill/test_corpus.py
git commit -m "feat(corpus): clean/multiturn filters + weighted merge"
```

---

### Task 5: 지시·형식 검증기

**Files:**
- Create: `compression/src/distill/ifeval_verify.py`
- Test: `compression/tests/distill/test_ifeval_verify.py`

**Interfaces:**
- Produces:
  ```python
  def verify_list_count(answer: str, n: int) -> bool       # 정확히 n개 번호목록
  def verify_json_keys(answer: str, keys: list[str]) -> bool  # JSON 파싱 + 키 일치
  def verify_forbidden(answer: str, words: list[str]) -> bool # 금지어 없음
  def verify_ending(answer: str, ending: str) -> bool      # 해당 문장으로 끝남
  ```
  합성 답 채택 게이트에 사용(통과분만 학습 데이터로).

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/distill/test_ifeval_verify.py
from src.distill.ifeval_verify import (
    verify_list_count, verify_json_keys, verify_forbidden, verify_ending,
)


def test_list_count():
    good = "1. 가\n2. 나\n3. 다"
    assert verify_list_count(good, 3) is True
    assert verify_list_count(good, 5) is False


def test_json_keys():
    assert verify_json_keys('{"summary": "x", "risks": "y"}', ["summary", "risks"]) is True
    assert verify_json_keys('{"summary": "x"}', ["summary", "risks"]) is False
    assert verify_json_keys("not json", ["a"]) is False


def test_forbidden():
    assert verify_forbidden("좋은 방법입니다.", ["최고", "완벽"]) is True
    assert verify_forbidden("이게 최고입니다.", ["최고", "완벽"]) is False


def test_ending():
    assert verify_ending("결론은 균형이 핵심이다.", "균형이 핵심이다.") is True
    assert verify_ending("균형이 핵심이다. 추가로...", "균형이 핵심이다.") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_ifeval_verify.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/distill/ifeval_verify.py
"""Programmatic verifiers for instruction-following synthetic answers."""
from __future__ import annotations

import json
import re


def verify_list_count(answer: str, n: int) -> bool:
    items = re.findall(r"(?m)^\s*\d+\.\s+\S", answer)
    return len(items) == n


def verify_json_keys(answer: str, keys: list[str]) -> bool:
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return False
    return isinstance(obj, dict) and sorted(obj.keys()) == sorted(keys)


def verify_forbidden(answer: str, words: list[str]) -> bool:
    return not any(w in answer for w in words)


def verify_ending(answer: str, ending: str) -> bool:
    return answer.strip().endswith(ending) and answer.strip().count(ending) >= 1 \
        and answer.strip()[-len(ending):] == ending
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_ifeval_verify.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add compression/src/distill/ifeval_verify.py compression/tests/distill/test_ifeval_verify.py
git commit -m "feat(corpus): instruction-format verifiers (list/json/forbidden/ending)"
```

---

### Task 6: "모른다" 합성 생성기

**Files:**
- Create: `compression/scripts/synth_idk.py`

**Interfaces:**
- Produces: jsonl(`messages` + `meta.kind` ∈ {idk_unanswerable, idk_grounded, answerable_pair, clarify}). answerable 짝과 되묻기 포함.

- [ ] **Step 1: Write the script**

```python
# compression/scripts/synth_idk.py
"""Synthesize Korean uncertainty/refusal data with answerable counterparts.

규모 작게(수천). 비율은 build_chat_corpus 에서 상한 적용.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

UNANSWERABLE = [
    "2027년 노벨물리학상 수상자는 누구인가요?",
    "강원도 양구군에 있는 '청록별빛마을'의 정확한 주소를 알려주세요.",
    "내년 로또 1등 당첨 번호를 알려주세요.",
    "제 옆자리 동료의 이름이 뭔가요?",
]
IDK_ANSWERS = [
    "그건 아직 일어나지 않은 일이라 확실하지 않습니다. 발표 후에 확인하실 수 있어요.",
    "정확히 확인할 수 없는 정보예요. 잘못된 정보를 드리지 않기 위해 모른다고 말씀드릴게요.",
]
# (지문, 지문에 답이 있는 질문, 지문에 없는 질문)
GROUNDED = [
    ("세종특별자치시는 2012년 7월 1일 출범했다.",
     "세종특별자치시는 언제 출범했나요?", "세종시의 초대 시장은 누구인가요?",
     "세종특별자치시는 2012년 7월 1일에 출범했습니다."),
]
CLARIFY = [
    ("그거 어떻게 해?", "어떤 것을 말씀하시는지 조금 더 알려주시겠어요? 맥락을 알면 정확히 도와드릴 수 있어요."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/synth_idk.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        r = rng.random()
        if r < 0.4:
            q = rng.choice(UNANSWERABLE); a = rng.choice(IDK_ANSWERS)
            rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "meta": {"kind": "idk_unanswerable"}})
        elif r < 0.8:
            sys, qa, qn, ans = rng.choice(GROUNDED)
            # answerable 짝
            rows.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": qa}, {"role": "assistant", "content": ans}], "meta": {"kind": "answerable_pair"}})
            rows.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": qn}, {"role": "assistant", "content": "그 내용은 지문에 없어 알 수 없습니다."}], "meta": {"kind": "idk_grounded"}})
        else:
            q, a = rng.choice(CLARIFY)
            rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "meta": {"kind": "clarify"}})
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and sanity check**

Run: `cd compression && .venv/bin/python scripts/synth_idk.py --out data/distill/synth_idk.jsonl --n 200`
Expected: `saved ... rows>=200`, 각 줄 `messages`+`meta.kind`, answerable/idk 짝 존재.

- [ ] **Step 3: Commit**

```bash
git add compression/scripts/synth_idk.py
git commit -m "feat(corpus): synthesize IDK data with answerable pairs + clarify"
```

---

### Task 7: 지시·형식 합성 생성기

**Files:**
- Create: `compression/scripts/synth_instructions.py`

**Interfaces:**
- Consumes: `ifeval_verify`(Task 5).
- Produces: jsonl(`messages` + `meta.constraint`). **검증기 통과분만** 기록. held-out 제약유형은 `--holdout` 로 분리.

- [ ] **Step 1: Write the script**

```python
# compression/scripts/synth_instructions.py
"""Synthesize instruction/format-constrained pairs; keep only verifier-passing answers.

답 생성은 템플릿(결정적)으로 — 형식 제약은 템플릿으로 정확히 만족 가능. (LLM 불필요)
다양성: 주제 풀 + 제약 풀을 곱집합으로 확장.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.ifeval_verify import verify_list_count, verify_json_keys, verify_forbidden, verify_ending

TOPICS = [
    "제주 여행 준비물", "노트북 구매 점검사항", "재택근무 장점", "건강한 아침 루틴",
    "초보자를 위한 등산 팁", "전기차 고려사항", "독서 습관 만들기", "예산 절약 방법",
]

def gen_list(topic, n):
    items = [f"{i+1}. {topic} 항목{i+1}" for i in range(n)]
    return f"{topic}에 대한 목록입니다.\n" + "\n".join(items), ("list_count", n)

def gen_json(topic):
    a = json.dumps({"summary": f"{topic} 요약", "risks": f"{topic} 위험"}, ensure_ascii=False)
    return a, ("json_keys", ["summary", "risks"])

def gen_forbidden(topic):
    return f"{topic}은 신중히 고르면 좋은 선택이 됩니다.", ("forbidden", ["최고", "완벽", "무조건"])

def gen_ending(topic):
    return f"{topic}에 대해 말하자면, 결국 균형이 핵심이다.", ("ending", "균형이 핵심이다.")

GENS = {"list_count": gen_list, "json_keys": gen_json, "forbidden": gen_forbidden, "ending": gen_ending}

def prompt_for(kind, topic, n):
    return {
        "list_count": f"{topic}을(를) 정확히 {n}개의 번호 목록으로만 답하세요.",
        "json_keys": f"{topic}을(를) JSON 객체로만 답하세요. 키는 summary, risks 두 개.",
        "forbidden": f"금지어 '최고','완벽','무조건'을 쓰지 말고 {topic}을(를) 한 문장으로 설명하세요.",
        "ending": f"{topic}에 대해 답하되 마지막 문장은 반드시 '균형이 핵심이다.'로 끝내세요.",
    }[kind]

def verify(kind, ans, arg):
    if kind == "list_count": return verify_list_count(ans, arg)
    if kind == "json_keys": return verify_json_keys(ans, arg)
    if kind == "forbidden": return verify_forbidden(ans, arg)
    if kind == "ending": return verify_ending(ans, arg)
    return False

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/synth_instructions.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--holdout", default="ending", help="평가용으로 빼는 제약유형")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kinds = [k for k in GENS if k != args.holdout]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with out.open("w", encoding="utf-8") as f:
        while kept < args.n:
            kind = rng.choice(kinds); topic = rng.choice(TOPICS); n = rng.choice([3, 4, 5])
            if kind == "list_count":
                ans, (vk, varg) = gen_list(topic, n)
            else:
                ans, (vk, varg) = GENS[kind](topic)
            if not verify(vk, ans, varg):
                dropped += 1; continue
            row = {"messages": [{"role": "user", "content": prompt_for(kind, topic, n)},
                                {"role": "assistant", "content": ans}],
                   "meta": {"constraint": kind}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); kept += 1
    print(f"saved {out} kept={kept} dropped={dropped} (holdout={args.holdout})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify all kept rows pass their constraint**

Run:
```bash
cd compression && .venv/bin/python scripts/synth_instructions.py --out data/distill/synth_instructions.jsonl --n 300
.venv/bin/python - <<'PY'
import json
from src.distill.ifeval_verify import verify_list_count, verify_json_keys, verify_forbidden
rows=[json.loads(l) for l in open("data/distill/synth_instructions.jsonl")]
print("rows", len(rows), "constraints", {r["meta"]["constraint"] for r in rows})
PY
```
Expected: 300행, holdout(ending) 미포함, 제약 종류 다양.

- [ ] **Step 3: Commit**

```bash
git add compression/scripts/synth_instructions.py
git commit -m "feat(corpus): synthesize verifier-passing instruction-format data"
```

---

### Task 8: 자연스러움 LLM 심사 점수 (경계 + 집계 TDD)

**Files:**
- Create: `compression/scripts/score_naturalness.py`

**Interfaces:**
- Produces: 입력 messages 표본에 대해 LLM 심사 자연스러움 점수(1~5). LLM 호출은 vLLM/OpenAI 호환 엔드포인트(Plan 2에서 서빙). 본 스크립트는 **프롬프트 구성 + 응답 파싱 + 집계**가 핵심이며 그 부분만 단위 검증.

- [ ] **Step 1: Write the script (LLM 호출은 함수 분리)**

```python
# compression/scripts/score_naturalness.py
"""LLM-judge naturalness scoring for sampled conversations (translationese detection)."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

JUDGE_PROMPT = (
    "다음 한국어 답변이 얼마나 자연스러운 원어민 한국어인지 1~5로 평가하세요"
    "(1=어색한 번역투, 5=완전 자연스러움). 숫자만 답하세요.\n\n답변:\n{ans}"
)


def build_prompt(answer: str) -> str:
    return JUDGE_PROMPT.format(ans=answer)


def parse_score(text: str) -> int | None:
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None


def judge(url: str, model: str, answer: str) -> int | None:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": build_prompt(answer)}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return parse_score(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--model", default="judge")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.corpus)][: args.sample]
    scores = []
    for r in rows:
        asst = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        if not asst:
            continue
        s = judge(args.url, args.model, asst[0])
        if s is not None:
            scores.append(s)
    if scores:
        print(f"n={len(scores)} mean_naturalness={sum(scores)/len(scores):.2f} "
              f"low(<=2)={sum(1 for s in scores if s<=2)/len(scores)*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Unit-test the deterministic parts**

Create `compression/tests/distill/test_naturalness.py`:
```python
from scripts.score_naturalness import build_prompt, parse_score


def test_parse_score():
    assert parse_score("4") == 4
    assert parse_score("점수: 2점") == 2
    assert parse_score("모르겠음") is None


def test_build_prompt_includes_answer():
    assert "안녕하세요" in build_prompt("안녕하세요")
```
Run: `cd compression && .venv/bin/python -m pytest tests/distill/test_naturalness.py -v`
Expected: PASS (2 passed). (실제 LLM 채점은 Plan 2에서 judge 서빙 후 통합 실행.)

- [ ] **Step 3: Commit**

```bash
git add compression/scripts/score_naturalness.py compression/tests/distill/test_naturalness.py
git commit -m "feat(corpus): LLM-judge naturalness scorer (deterministic parts tested)"
```

---

### Task 9: 코퍼스 빌더 (통합 실행)

**Files:**
- Create: `compression/scripts/build_chat_corpus.py`

**Interfaces:**
- Consumes: `corpus`, `ko_text`, `korquad_chat`, 합성 jsonl(Task 6·7).
- Produces: `data/distill/chat_corpus.jsonl`(통일 `messages`) + 통계 출력. 번역:원어민 비율·모른다 비율 상한 적용.

- [ ] **Step 1: Write the script**

```python
# compression/scripts/build_chat_corpus.py
"""Build the unified multi-turn chat corpus from filtered + synthesized sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill.corpus import conversation_is_clean, is_multiturn, weighted_merge
from src.distill.korquad_chat import parse_korquad_chat


def load_smol(limit, threshold):
    ds = load_dataset("lemon-mint/smol-koreantalk", split="train", streaming=True)
    keep = []
    for row in ds:
        msgs = [{"role": m.get("role"), "content": m.get("content", "")} for m in row.get("messages", [])]
        msgs = [m for m in msgs if m["role"] in ("user", "assistant", "system")]
        if is_multiturn(msgs) and conversation_is_clean(msgs, threshold):
            keep.append(msgs)
        if limit and len(keep) >= limit:
            break
    return keep


def load_korquad():
    ds = load_dataset("heegyu/korquad-chat-v1", split="train")
    out = []
    for row in ds:
        p = parse_korquad_chat(row["text"])
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + p["messages"]
        if is_multiturn(p["messages"]):
            out.append(msgs)
    return out


def load_jsonl_msgs(path):
    if not path or not Path(path).exists():
        return []
    return [json.loads(l)["messages"] for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/chat_corpus.jsonl")
    ap.add_argument("--smol-limit", type=int, default=80000)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--korquad-repeat", type=int, default=5)   # 원어민 업샘플링
    ap.add_argument("--idk", default="data/distill/synth_idk.jsonl")
    ap.add_argument("--instructions", default="data/distill/synth_instructions.jsonl")
    ap.add_argument("--idk-max-frac", type=float, default=0.10)
    args = ap.parse_args()

    smol = load_smol(args.smol_limit, args.threshold)
    korquad = load_korquad()
    idk = load_jsonl_msgs(args.idk)
    instr = load_jsonl_msgs(args.instructions)

    base = weighted_merge([
        {"name": "smol", "convos": smol, "repeat": 1},
        {"name": "korquad", "convos": korquad, "repeat": args.korquad_repeat},
        {"name": "instructions", "convos": instr, "repeat": 1},
    ])
    # 모른다 비율 상한
    cap = int(len(base) * args.idk_max_frac / (1 - args.idk_max_frac))
    idk = idk[:cap]
    allc = base + idk

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for msgs in allc:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"saved {out} total={len(allc)} | smol={len(smol)} korquad={len(korquad)}x{args.korquad_repeat} "
          f"instr={len(instr)} idk={len(idk)} (cap {cap})")
    print(f"번역:원어민 ≈ {len(smol)} : {len(korquad)*args.korquad_repeat}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end (small), inspect stats**

Run (synth 먼저 만든 뒤):
```bash
cd compression
.venv/bin/python scripts/synth_idk.py --n 500
.venv/bin/python scripts/synth_instructions.py --n 1000
.venv/bin/python scripts/build_chat_corpus.py --smol-limit 3000 --korquad-repeat 5
```
Expected: `chat_corpus.jsonl` 생성, 통계에 smol/korquad/instr/idk 분포 + 번역:원어민 비율 출력. idk 비율 ≤10%.

- [ ] **Step 3: Validate output schema**

Run:
```bash
cd compression && .venv/bin/python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/distill/chat_corpus.jsonl")]
assert all("messages" in r for r in rows)
mt=sum(1 for r in rows if sum(1 for m in r["messages"] if m["role"]=="assistant")>=2)
print(f"rows={len(rows)} multiturn={mt/len(rows)*100:.1f}%")
PY
```
Expected: 멀티턴 비율 높음(>60%), 모든 행 `messages` 보유.

- [ ] **Step 4: Commit**

```bash
git add compression/scripts/build_chat_corpus.py
git commit -m "feat(corpus): build unified multi-turn chat corpus with weighting + caps"
```

---

## Self-Review

- **Spec coverage:** 멀티턴 마스킹(T1)·토큰종류 영어필터(T2)·korquad 파서(T3)·정제/가중/병합(T4·T9)·지시검증+합성(T5·T7)·모른다 합성+answerable짝(T6)·자연스러움 점수(T8)·번역:원어민 비율관리+idk상한(T9) — 스펙 데이터 섹션 전부 매핑. 학습·평가는 Plan 2. ✅
- **Placeholder scan:** 모든 코드 스텝에 실제 코드. "적절히" 류 없음. ✅
- **Type consistency:** `english_prose_ratio`(T2)→`conversation_is_clean`(T4·T9), `parse_korquad_chat`(T3)→builder(T9), `verify_*`(T5)→synth_instructions(T7), `weighted_merge` 시그니처(T4)=builder 호출(T9) 일치. ✅

## 다음 (Plan 2)
코퍼스 검증(자연스러움 점수·멀티턴 비율·번역:원어민) 통과 후: 멀티턴 plain-CE SFT(train_sft_multiturn) + 평가(LLM 페어와이즈·held-out 멀티턴·KMMLU 임계). GPU VM 필요.
