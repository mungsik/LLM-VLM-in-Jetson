# 평가 하니스 Implementation Plan (Plan 3/5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 파이프라인 각 단계(프루닝/CPT/SFT)를 측정하는 평가 하니스를 만든다 — PPL(한/영), Ko-IFEval(지시 따르기), 대화품질 페어와이즈(개인 GPT judge), 그리고 기존 지표(KMMLU·영어섞임·자연스러움·IDK)를 묶는 단계별 리포트.

**Architecture:** 순수 로직(PPL 집계식, judge 프롬프트/판정 파싱/위치편향 보정, IFEval 디스패치, 리포트 조립)을 분리해 TDD로 검증하고, 모델 추론·GPT API·lm-eval 같은 통합부는 얇은 래퍼 + 스모크로 검증한다. 기존 `src/distill`(chat_metrics/ifeval_verify/ko_text)·`src/common`(eval_kmmlu)을 재사용한다.

**Tech Stack:** Python 3.10, transformers, `openai`(개인 GPT judge), lm-eval-harness(기존), pytest, uv.

## Global Constraints

- **평가 judge = 개인 GPT API** (이 프로젝트는 고객사 데이터 아님). 키는 로컬 `source-code/.env`의 `OPENAI_API_KEY` → VM 환경변수로 주입. LogicKor/MT-Bench 표준 방식.
- **MCQA 단일 지표 신뢰 금지** — 생성·대화 품질 지표 병행(chat-v1 wash 교훈).
- **페어와이즈는 위치편향 보정 필수** — A/B 순서를 바꿔 2회 판정 후 합산.
- 작업 위치 = `compression/`. 테스트 = `uv run python -m pytest`. 커밋은 VM에서만. 광범위 `git add -A` 금지.
- 새 코드는 `src/eval/`(신규 모듈)에. 기존 `src/distill`·`src/common` 지표는 재사용(재작성 금지).

## 재사용 (그대로)

- `src/common/eval_kmmlu.py` — `run_kmmlu(model_path, limit, device, batch_size)`.
- `src/distill/chat_metrics.py` — `english_mixing_rate(answers)`, `refused(answer)`, `idk_calibration(records)`.
- `src/distill/ifeval_verify.py` — `verify_list_count`, `verify_json_keys`, `verify_forbidden`, `verify_ending`.
- `src/distill/ko_text.py` — `english_prose_ratio(text)`.

## File Structure

- `compression/src/eval/__init__.py`
- `compression/src/eval/ppl.py` — perplexity 집계 + 모델 기반 계산
- `compression/src/eval/judge.py` — GPT 페어와이즈 judge (프롬프트/파싱/위치편향/호출)
- `compression/src/eval/ifeval_run.py` — Ko-IFEval 프롬프트셋 + 러너 (verifier 재사용)
- `compression/src/eval/report.py` — 단계별 리포트 조립
- `compression/scripts/eval_chat.py` — 생성 + judge 페어와이즈 CLI
- `compression/tests/eval/{__init__.py,test_ppl.py,test_judge.py,test_ifeval_run.py,test_report.py}`

---

### Task 1: PPL (`ppl.py`)

**Files:**
- Create: `compression/src/eval/__init__.py`, `compression/src/eval/ppl.py`
- Test: `compression/tests/eval/__init__.py`, `compression/tests/eval/test_ppl.py`

**Interfaces:**
- Produces:
  - `perplexity_from_nll(total_nll: float, n_tokens: int) -> float` — `exp(total_nll / n_tokens)`. n_tokens=0이면 `float('inf')`.
  - `compute_ppl(model, tokenizer, texts: list[str], max_length: int = 2048) -> float` — 각 텍스트를 causal LM으로 통과시켜 토큰 NLL 합/개수 누적 후 `perplexity_from_nll`. (통합; 스모크로 검증)

- [ ] **Step 1: Write the failing test** (순수 집계식만 단위테스트)

```python
# compression/tests/eval/test_ppl.py
import math

from src.eval.ppl import perplexity_from_nll


def test_perplexity_uniform():
    # 평균 NLL = 1.0 → ppl = e
    assert abs(perplexity_from_nll(10.0, 10) - math.e) < 1e-9


def test_perplexity_zero_nll_is_one():
    assert perplexity_from_nll(0.0, 5) == 1.0


def test_perplexity_no_tokens_is_inf():
    assert perplexity_from_nll(0.0, 0) == float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/test_ppl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.ppl'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/eval/__init__.py
# (empty)
```
```python
# compression/tests/eval/__init__.py
# (empty)
```
```python
# compression/src/eval/ppl.py
"""Perplexity 평가 — 집계식(순수) + 모델 기반 계산(통합)."""
from __future__ import annotations

import math


def perplexity_from_nll(total_nll: float, n_tokens: int) -> float:
    if n_tokens <= 0:
        return float("inf")
    return math.exp(total_nll / n_tokens)


def compute_ppl(model, tokenizer, texts: list[str], max_length: int = 2048) -> float:
    """causal LM NLL 누적 → perplexity. 실모델 필요(스모크 검증)."""
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = ids["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # HF loss = 평균 NLL(shift 반영). 토큰수 = n-1.
        n = input_ids.size(1) - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    return perplexity_from_nll(total_nll, total_tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/test_ppl.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/eval/__init__.py src/eval/ppl.py tests/eval/__init__.py tests/eval/test_ppl.py
git commit -m "feat(eval): PPL 집계식 + 모델 기반 perplexity"
```

---

### Task 2: GPT 페어와이즈 judge (`judge.py`)

**Files:**
- Create: `compression/src/eval/judge.py`
- Test: `compression/tests/eval/test_judge.py`

**Interfaces:**
- Produces:
  - `build_pairwise_prompt(question: str, answer_a: str, answer_b: str) -> str` — judge 지시(한국어 대화품질 비교, 판정을 `[[A]]`/`[[B]]`/`[[C]]`(무승부)로 출력하라고 명시).
  - `parse_pairwise_verdict(text: str) -> str` — judge 응답에서 마지막 `[[A|B|C]]`를 추출해 `"A"|"B"|"tie"`. 없으면 `"tie"`.
  - `aggregate_two_orders(v_ab: str, v_ba: str) -> str` — 위치편향 보정: 1차(A=our,B=base), 2차(순서 뒤집음, A=base,B=our) 판정을 합쳐 `"win"|"loss"|"tie"`(our 기준). 두 판정이 일관되게 our 우세면 win, 일관 열세면 loss, 그 외 tie.
  - `judge_pairwise(client, question, our_answer, base_answer, model="gpt-4o") -> str` — 두 순서로 GPT 호출 후 `aggregate_two_orders`. (통합; 스모크)

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/eval/test_judge.py
from src.eval.judge import (
    aggregate_two_orders,
    build_pairwise_prompt,
    parse_pairwise_verdict,
)


def test_prompt_contains_question_and_both_answers():
    p = build_pairwise_prompt("질문?", "답A", "답B")
    assert "질문?" in p and "답A" in p and "답B" in p
    assert "[[A]]" in p and "[[B]]" in p and "[[C]]" in p  # 출력 형식 명시


def test_parse_last_verdict():
    assert parse_pairwise_verdict("설명...\n최종 판정: [[A]]") == "A"
    assert parse_pairwise_verdict("[[A]] 였다가 정정 [[B]]") == "B"   # 마지막 우선
    assert parse_pairwise_verdict("[[C]]") == "tie"
    assert parse_pairwise_verdict("판정 불가") == "tie"                # 없으면 tie


def test_aggregate_consistent_win():
    # 1차: A=our 우세(A). 2차(순서뒤집음): B=our 우세(B). → 일관 win
    assert aggregate_two_orders("A", "B") == "win"


def test_aggregate_consistent_loss():
    # 1차: B=base 우세(B). 2차: A=base 우세(A). → 일관 loss
    assert aggregate_two_orders("B", "A") == "loss"


def test_aggregate_inconsistent_is_tie():
    assert aggregate_two_orders("A", "A") == "tie"   # 위치편향(항상 A) → tie
    assert aggregate_two_orders("C", "C") == "tie"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.judge'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/eval/judge.py
"""GPT 페어와이즈 대화품질 judge (LogicKor/MT-Bench식, 위치편향 보정)."""
from __future__ import annotations

import re

_VERDICT = re.compile(r"\[\[([ABC])\]\]")

_PROMPT = """당신은 한국어 대화 품질을 평가하는 심사위원입니다.
아래 [질문]에 대한 두 답변 [답변 A]와 [답변 B]를 비교하세요.
평가 기준: 한국어의 자연스러움(번역투 감점), 질문 의도 충족, 정확성, 도움이 되는 정도.
간단한 근거를 쓴 뒤, 마지막 줄에 반드시 판정을 다음 중 하나로 출력하세요:
[[A]] (A가 더 나음) / [[B]] (B가 더 나음) / [[C]] (무승부).

[질문]
{q}

[답변 A]
{a}

[답변 B]
{b}
"""


def build_pairwise_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return _PROMPT.format(q=question, a=answer_a, b=answer_b)


def parse_pairwise_verdict(text: str) -> str:
    matches = _VERDICT.findall(text or "")
    if not matches:
        return "tie"
    last = matches[-1]
    return "tie" if last == "C" else last


def aggregate_two_orders(v_ab: str, v_ba: str) -> str:
    """v_ab: our=A,base=B 판정. v_ba: base=A,our=B 판정(순서뒤집음).
    our 우세 신호 = 1차 'A' + 2차 'B'. base 우세 = 1차 'B' + 2차 'A'."""
    our_first = v_ab == "A"
    our_second = v_ba == "B"
    base_first = v_ab == "B"
    base_second = v_ba == "A"
    if our_first and our_second:
        return "win"
    if base_first and base_second:
        return "loss"
    return "tie"


def judge_pairwise(client, question, our_answer, base_answer, model: str = "gpt-4o") -> str:
    """두 순서로 GPT 호출 후 합산. client = openai.OpenAI(). (통합, 스모크 검증)"""
    def _ask(a, b):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_pairwise_prompt(question, a, b)}],
            temperature=0,
        )
        return parse_pairwise_verdict(resp.choices[0].message.content)

    v_ab = _ask(our_answer, base_answer)   # A=our
    v_ba = _ask(base_answer, our_answer)   # A=base
    return aggregate_two_orders(v_ab, v_ba)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/test_judge.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/eval/judge.py tests/eval/test_judge.py
git commit -m "feat(eval): GPT 페어와이즈 judge (위치편향 보정)"
```

---

### Task 3: Ko-IFEval 러너 (`ifeval_run.py`)

**Files:**
- Create: `compression/src/eval/ifeval_run.py`
- Test: `compression/tests/eval/test_ifeval_run.py`

**Interfaces:**
- Consumes: `src/distill/ifeval_verify` (검증기 4종).
- Produces:
  - `IFEVAL_PROMPTS: list[dict]` — 각 `{ "prompt": str, "verify": callable(answer)->bool }`. 소규모 in-repo 한국어 지시셋(번호목록/JSON/금칙어/문장끝).
  - `run_ifeval(generate_fn, prompts=IFEVAL_PROMPTS) -> dict` — 각 프롬프트에 `generate_fn(prompt)`로 답 생성 후 verify. 반환 `{"pass_rate": float, "n": int, "passed": int}`.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/eval/test_ifeval_run.py
from src.eval.ifeval_run import IFEVAL_PROMPTS, run_ifeval


def test_prompts_have_prompt_and_verify():
    assert len(IFEVAL_PROMPTS) >= 4
    for p in IFEVAL_PROMPTS:
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert callable(p["verify"])


def test_run_all_pass_with_oracle():
    # oracle generate_fn: 각 프롬프트의 정답을 그대로 돌려주면 pass_rate=1.0
    def oracle(prompt):
        return next(p["gold"] for p in IFEVAL_PROMPTS if p["prompt"] == prompt)

    res = run_ifeval(oracle)
    assert res["pass_rate"] == 1.0
    assert res["n"] == len(IFEVAL_PROMPTS)


def test_run_all_fail_with_empty():
    res = run_ifeval(lambda prompt: "")
    assert res["passed"] == 0
    assert res["pass_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/test_ifeval_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.ifeval_run'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/eval/ifeval_run.py
"""Ko-IFEval: 소규모 한국어 지시 프롬프트셋 + 러너 (검증기 재사용)."""
from __future__ import annotations

from src.distill.ifeval_verify import (
    verify_ending,
    verify_forbidden,
    verify_json_keys,
    verify_list_count,
)

# 각 항목: prompt(모델 입력), verify(답 검증), gold(오라클 테스트용 정답 예시)
IFEVAL_PROMPTS: list[dict] = [
    {
        "prompt": "사과의 장점을 정확히 3개, 번호 목록으로만 답하세요.",
        "verify": lambda a: verify_list_count(a, 3),
        "gold": "1. 비타민이 풍부하다\n2. 포만감을 준다\n3. 보관이 쉽다",
    },
    {
        "prompt": '이름과 나이를 "name","age" 키를 가진 JSON으로만 답하세요.',
        "verify": lambda a: verify_json_keys(a, ["name", "age"]),
        "gold": '{"name": "홍길동", "age": 30}',
    },
    {
        "prompt": "여행의 좋은 점을 설명하되 '돈'이라는 단어는 절대 쓰지 마세요.",
        "verify": lambda a: verify_forbidden(a, ["돈"]),
        "gold": "여행은 새로운 경험과 견문을 넓혀 줍니다.",
    },
    {
        "prompt": "짧게 답하고 반드시 '끝.'으로 문장을 마치세요.",
        "verify": lambda a: verify_ending(a, "끝."),
        "gold": "오늘 할 일을 마쳤습니다. 끝.",
    },
]


def run_ifeval(generate_fn, prompts: list[dict] = IFEVAL_PROMPTS) -> dict:
    passed = 0
    for p in prompts:
        ans = generate_fn(p["prompt"])
        if p["verify"](ans):
            passed += 1
    n = len(prompts)
    return {"pass_rate": passed / n if n else 0.0, "n": n, "passed": passed}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/test_ifeval_run.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/eval/ifeval_run.py tests/eval/test_ifeval_run.py
git commit -m "feat(eval): Ko-IFEval 프롬프트셋 + 러너"
```

---

### Task 4: 단계별 리포트 (`report.py`)

**Files:**
- Create: `compression/src/eval/report.py`
- Test: `compression/tests/eval/test_report.py`

**Interfaces:**
- Produces:
  - `build_stage_report(stage: str, metrics: dict) -> dict` — `{"stage", "metrics", "flags"}` 반환. `flags`는 경고 리스트: 영어섞임율>0.10 → "english_mixing_high"; ppl_ko가 있고 baseline_ppl_ko보다 크면 "korean_ppl_regressed"; ppl_en이 baseline_ppl_en의 1.2배 초과면 "english_ppl_regressed"(망각 경보).
  - `format_report_md(report: dict) -> str` — 사람이 읽는 마크다운.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/eval/test_report.py
from src.eval.report import build_stage_report, format_report_md


def test_flags_english_mixing_high():
    r = build_stage_report("sft", {"english_mixing_rate": 0.2})
    assert "english_mixing_high" in r["flags"]


def test_flags_english_ppl_regression():
    r = build_stage_report("cpt", {"ppl_en": 30.0, "baseline_ppl_en": 20.0})
    assert "english_ppl_regressed" in r["flags"]   # 30 > 20*1.2


def test_no_flags_when_healthy():
    r = build_stage_report("cpt", {
        "english_mixing_rate": 0.01,
        "ppl_ko": 8.0, "baseline_ppl_ko": 12.0,     # 개선
        "ppl_en": 21.0, "baseline_ppl_en": 20.0,    # 1.2배 이내
    })
    assert r["flags"] == []


def test_format_md_contains_stage_and_metric():
    md = format_report_md(build_stage_report("prune", {"kmmlu": 0.38}))
    assert "prune" in md and "kmmlu" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/eval/report.py
"""단계별 평가 리포트 조립 + 경고 플래그."""
from __future__ import annotations


def build_stage_report(stage: str, metrics: dict) -> dict:
    flags: list[str] = []
    m = metrics

    if m.get("english_mixing_rate", 0.0) > 0.10:
        flags.append("english_mixing_high")

    if "ppl_ko" in m and "baseline_ppl_ko" in m and m["ppl_ko"] > m["baseline_ppl_ko"]:
        flags.append("korean_ppl_regressed")

    if "ppl_en" in m and "baseline_ppl_en" in m and m["ppl_en"] > m["baseline_ppl_en"] * 1.2:
        flags.append("english_ppl_regressed")

    return {"stage": stage, "metrics": m, "flags": flags}


def format_report_md(report: dict) -> str:
    lines = [f"## 평가 리포트 — {report['stage']}", "", "| 지표 | 값 |", "|---|---|"]
    for k, v in report["metrics"].items():
        lines.append(f"| {k} | {v} |")
    if report["flags"]:
        lines += ["", "**⚠️ 경고:** " + ", ".join(report["flags"])]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/test_report.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/eval/report.py tests/eval/test_report.py
git commit -m "feat(eval): 단계별 평가 리포트 + 회귀/망각 경고 플래그"
```

---

### Task 5: 대화 평가 CLI + GPT judge 스모크 (`eval_chat.py`)

**Files:**
- Create: `compression/scripts/eval_chat.py`

**Interfaces:**
- Consumes: `judge_pairwise`(Task 2), `english_mixing_rate`/`refused`(chat_metrics), `run_ifeval`(Task 3), `build_stage_report`/`format_report_md`(Task 4).
- Produces: 두 모델(our vs base)의 답을 eval 질문셋에 대해 생성→페어와이즈 judge→영어섞임/자연스러움 집계→단계 리포트 출력. (통합; 스모크)

> 이 태스크는 실모델 생성 + GPT API가 필요하므로 단위테스트 없이 **스모크**로 검증한다. `OPENAI_API_KEY`는 환경변수로 주입.

- [ ] **Step 1: openai 의존성 확인 (VM)**

Run: `uv run python -c "import openai; print(openai.__version__)"`
Expected: 버전 출력. 없으면 `uv pip install openai` (uv sync 금지 — 운영 dep prune 방지 기조).

- [ ] **Step 2: 스크립트 작성**

```python
# compression/scripts/eval_chat.py
"""대화 평가 CLI: our vs base 페어와이즈(GPT judge) + 영어섞임/자연스러움 + 리포트."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distill.chat_metrics import english_mixing_rate
from src.eval.judge import judge_pairwise
from src.eval.report import build_stage_report, format_report_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True,
                    help='JSONL: {"question","our","base"} 라인들')
    ap.add_argument("--stage", default="sft")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rows = []
    with open(args.answers, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wins = losses = ties = 0
    for r in rows:
        v = judge_pairwise(client, r["question"], r["our"], r["base"], model=args.model)
        wins += v == "win"
        losses += v == "loss"
        ties += v == "tie"

    our_answers = [r["our"] for r in rows]
    metrics = {
        "n": len(rows),
        "win": wins, "loss": losses, "tie": ties,
        "win_rate": wins / len(rows) if rows else 0.0,
        "english_mixing_rate": english_mixing_rate(our_answers),
    }
    report = build_stage_report(args.stage, metrics)
    print(format_report_md(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 스모크 실행 (VM, 소량, GPT API)**

`/tmp/eval_smoke.jsonl` (3줄 예시):
```json
{"question": "사과의 좋은 점을 알려줘", "our": "사과는 비타민이 풍부하고 포만감을 줍니다.", "base": "Apple is good because it has vitamins and helps you feel full."}
```
Run:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uv run python scripts/eval_chat.py --answers /tmp/eval_smoke.jsonl --stage sft
```
Expected: 마크다운 리포트 출력 — win/loss/tie 집계 + `english_mixing_rate` + (base가 영어라 our가 이기고 base쪽 섞임 높게 나오는 방향). 에러 없이 GPT judge 왕복 완료. 확인 후 `/tmp/eval_smoke.jsonl` 삭제.

- [ ] **Step 4: Commit** (VM에서)

```bash
git add scripts/eval_chat.py
git commit -m "feat(eval): 대화 평가 CLI (페어와이즈 GPT judge + 지표 리포트)"
```

---

## Self-Review

**1. Spec coverage:** 스펙 §6 평가 전 항목 커버 — PPL(한/영, Task1), Ko-IFEval(Task3), 대화 페어와이즈 GPT judge(Task2·5), 영어섞임/자연스러움(chat_metrics·ko_text 재사용), KMMLU(eval_kmmlu 재사용), 회귀/망각 경고(Task4). judge=개인 GPT API, 위치편향 보정. ✅
**2. Placeholder scan:** 순수 로직(집계식/파싱/디스패치/리포트) 전부 실제 코드+테스트. 통합부(compute_ppl/judge_pairwise/eval_chat)는 스모크로 검증(모델·API 필요라 단위테스트 대신). ✅
**3. Type consistency:** `perplexity_from_nll`(Task1)→compute_ppl 소비, `parse_pairwise_verdict`/`aggregate_two_orders`(Task2)→judge_pairwise 소비, `run_ifeval`(Task3) `{pass_rate,n,passed}`, `build_stage_report`(Task4) `{stage,metrics,flags}`→eval_chat 소비. 체인 일관. ✅

---

## 다음 플랜 (예정)
- **Plan 4**: 프루닝 스윕 + CPT full-FT 학습 (Unsloth) — Plan 1 데이터 소비, 각 단계 Plan 3 하니스로 평가.
- **Plan 5**: SFT LoRA(Plan 2 데이터) + GGUF 양자화 + Jetson 배포/실측.
