# CPT 코퍼스 파이프라인 Implementation Plan (Plan 1/5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한국어 raw 텍스트(WanJuan-KO + KoWiki)를 Phi-4 토크나이저로 packing하고 영어 replay를 섞어, CPT(continued pre-training)에 바로 먹일 수 있는 토큰화·패킹된 데이터셋을 생성한다.

**Architecture:** 순수 로직 3개(`packing`, `replay`, `cpt_corpus`)를 분리하고, CLI 스크립트가 이를 엮어 `datasets` 포맷으로 디스크에 저장한다. 순수 함수는 tiny fake 토크나이저로 단위 테스트하고, 코퍼스 빌더는 인메모리 tiny 데이터셋으로 통합 테스트한다.

**Tech Stack:** Python 3.10, HuggingFace `datasets` + `transformers`, `pytest`, uv.

## Global Constraints

- 하드웨어 타겟 = **Jetson Orin Nano 8GB 고정**. 이 플랜은 데이터 생성이라 직접 관련은 없으나, block_size 등 기본값은 학습 설정(seq_len 4096)과 정합해야 함.
- 작업 위치 = repo의 `compression/` 디렉토리. 모든 명령의 cwd = `compression/`.
- 테스트 = `uv run python -m pytest`. 네트워크 없는 단위 테스트 우선(fake 토크나이저/인메모리 데이터).
- **git 커밋은 VM(`phi4-blackwell`)에서만** (이 Mac은 gitdir 깨짐). 커밋 명령은 VM 쉘 기준. 광범위 `git add -A` 금지 — 명시 경로만 add.
- CPT 라벨 = `input_ids`와 동일(전체 토큰 loss). SFT의 completion-only 마스킹과 다름.
- 기존 `compression/src/{common,prune,distill}` 코드와 **겹치지 않게** 새 `compression/src/data/` 모듈에 둔다.

---

## File Structure

- `compression/src/data/__init__.py` — 새 데이터 모듈 (생성)
- `compression/src/data/packing.py` — 토큰 리스트를 고정 길이 블록으로 패킹 (순수 로직)
- `compression/src/data/replay.py` — 영어 replay를 목표 비율로 결정적 인터리브 (순수 로직)
- `compression/src/data/cpt_corpus.py` — HF 데이터셋 로드→토큰화→패킹→replay 혼합 (조립)
- `compression/scripts/build_cpt_data.py` — CLI: 위를 엮어 디스크에 저장
- `compression/configs/cpt_data.yaml` — 데이터 소스/파라미터 설정
- `compression/tests/data/__init__.py`
- `compression/tests/data/test_packing.py`
- `compression/tests/data/test_replay.py`
- `compression/tests/data/test_cpt_corpus.py`

---

### Task 1: 패킹 로직 (`packing.py`)

**Files:**
- Create: `compression/src/data/__init__.py`, `compression/src/data/packing.py`
- Test: `compression/tests/data/__init__.py`, `compression/tests/data/test_packing.py`

**Interfaces:**
- Produces: `pack_token_lists(token_lists: list[list[int]], block_size: int, eos_id: int) -> list[list[int]]`
  — 각 문서 뒤에 `eos_id`를 붙여 이어붙인 뒤 `block_size`로 잘라 **완전한 블록만** 반환(마지막 잔여 조각은 버림).

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/data/test_packing.py
from src.data.packing import pack_token_lists


def test_packs_and_inserts_eos_dropping_remainder():
    docs = [[1, 2, 3], [4, 5]]
    # stream = 1,2,3,eos(0),4,5,eos(0) -> blocks of 3: [1,2,3],[0,4,5]; remainder [0] dropped
    out = pack_token_lists(docs, block_size=3, eos_id=0)
    assert out == [[1, 2, 3], [0, 4, 5]]


def test_empty_input_returns_empty():
    assert pack_token_lists([], block_size=4, eos_id=0) == []


def test_all_blocks_have_exact_block_size():
    docs = [list(range(1, 21))]
    out = pack_token_lists(docs, block_size=4, eos_id=0)
    assert all(len(b) == 4 for b in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/data/test_packing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/data/__init__.py
# (empty)
```
```python
# compression/tests/data/__init__.py
# (empty)
```
```python
# compression/src/data/packing.py
"""고정 길이 블록 패킹 — CPT용 (문서 사이 EOS 삽입, 잔여 조각 폐기)."""
from __future__ import annotations


def pack_token_lists(
    token_lists: list[list[int]], block_size: int, eos_id: int
) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    buffer: list[int] = []
    for doc in token_lists:
        buffer.extend(doc)
        buffer.append(eos_id)
    n_full = len(buffer) // block_size
    return [buffer[i * block_size : (i + 1) * block_size] for i in range(n_full)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/data/test_packing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/data/__init__.py src/data/packing.py tests/data/__init__.py tests/data/test_packing.py
git commit -m "feat(data): CPT용 토큰 패킹 로직"
```

---

### Task 2: 영어 replay 인터리브 (`replay.py`)

**Files:**
- Create: `compression/src/data/replay.py`
- Test: `compression/tests/data/test_replay.py`

**Interfaces:**
- Produces: `interleave(primary: list, replay: list, replay_ratio: float, seed: int = 0) -> list`
  — primary는 전부 소비. 결과에서 replay가 차지하는 비율 ≈ `replay_ratio`. 결정적(seed 고정 시 동일 출력). `replay_ratio<=0`이거나 replay가 비면 primary 그대로 반환. replay 풀이 모자라면 순환 사용.

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/data/test_replay.py
from src.data.replay import interleave


def test_hits_target_ratio_and_consumes_all_primary():
    primary = list(range(1000))          # < 10000
    replay = list(range(10000, 20000))   # >= 10000
    mixed = interleave(primary, replay, replay_ratio=0.2, seed=0)
    n_replay = sum(1 for x in mixed if x >= 10000)
    assert abs(n_replay / len(mixed) - 0.2) < 0.02
    assert sum(1 for x in mixed if x < 1000) == 1000  # 모든 primary 유지


def test_zero_ratio_returns_primary_unchanged():
    primary = [1, 2, 3]
    assert interleave(primary, [9, 9], replay_ratio=0.0) == [1, 2, 3]


def test_deterministic_with_seed():
    p, r = list(range(50)), list(range(100, 200))
    assert interleave(p, r, 0.3, seed=7) == interleave(p, r, 0.3, seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/data/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.replay'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/data/replay.py
"""primary 스트림에 replay를 목표 비율로 결정적 인터리브 (망각 방지용)."""
from __future__ import annotations

import random


def interleave(primary: list, replay: list, replay_ratio: float, seed: int = 0) -> list:
    if replay_ratio <= 0 or not replay:
        return list(primary)
    if not 0 < replay_ratio < 1:
        raise ValueError("replay_ratio must be in (0, 1)")
    rng = random.Random(seed)
    pool = list(replay)
    rng.shuffle(pool)
    factor = replay_ratio / (1 - replay_ratio)  # primary 1개당 삽입할 replay 기대수
    out: list = []
    n_primary = 0
    n_replay = 0
    ri = 0
    for item in primary:
        out.append(item)
        n_primary += 1
        while n_replay < factor * n_primary:
            out.append(pool[ri % len(pool)])
            ri += 1
            n_replay += 1
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/data/test_replay.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/data/replay.py tests/data/test_replay.py
git commit -m "feat(data): 영어 replay 결정적 인터리브"
```

---

### Task 3: CPT 코퍼스 빌더 (`cpt_corpus.py`)

**Files:**
- Create: `compression/src/data/cpt_corpus.py`
- Test: `compression/tests/data/test_cpt_corpus.py`

**Interfaces:**
- Consumes: `pack_token_lists` (Task 1), `interleave` (Task 2)
- Produces:
  - `tokenize_texts(texts: list[str], tokenizer) -> list[list[int]]` — 각 텍스트를 `add_special_tokens=False`로 토큰화.
  - `build_cpt_dataset(primary_texts, replay_texts, tokenizer, block_size, replay_ratio, seed=0) -> datasets.Dataset` — 토큰화→패킹→replay 혼합. 반환 컬럼: `input_ids`, `labels`(= input_ids 복사).

- [ ] **Step 1: Write the failing test**

```python
# compression/tests/data/test_cpt_corpus.py
from src.data.cpt_corpus import build_cpt_dataset, tokenize_texts


class FakeTokenizer:
    """공백 분할 → 정수 id. eos_token_id 제공. add_special_tokens 무시."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [(abs(hash(tok)) % 1000) + 1 for tok in text.split()]
        return {"input_ids": ids}


def test_tokenize_texts_shape():
    tok = FakeTokenizer()
    out = tokenize_texts(["a b c", "d e"], tok)
    assert [len(x) for x in out] == [3, 2]


def test_build_dataset_blocks_and_labels():
    tok = FakeTokenizer()
    primary = ["w " * 10 for _ in range(20)]   # 문서당 10토큰
    ds = build_cpt_dataset(
        primary_texts=primary, replay_texts=[], tokenizer=tok,
        block_size=8, replay_ratio=0.0, seed=0,
    )
    assert set(ds.column_names) == {"input_ids", "labels"}
    assert all(len(r) == 8 for r in ds["input_ids"])
    assert ds["labels"][0] == ds["input_ids"][0]   # CPT: labels == input_ids


def test_build_dataset_mixes_replay():
    tok = FakeTokenizer()
    primary = ["ko " * 8 for _ in range(50)]
    replay = ["en " * 8 for _ in range(50)]
    ds = build_cpt_dataset(primary, replay, tok, block_size=8, replay_ratio=0.2, seed=0)
    # replay 섞였으니 블록 수가 primary-only보다 많아야 함
    ds_only = build_cpt_dataset(primary, [], tok, block_size=8, replay_ratio=0.0)
    assert len(ds) > len(ds_only)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/data/test_cpt_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.cpt_corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# compression/src/data/cpt_corpus.py
"""CPT 코퍼스 빌더: 텍스트 → 토큰화 → 패킹 → 영어 replay 혼합 → datasets.Dataset."""
from __future__ import annotations

from datasets import Dataset

from src.data.packing import pack_token_lists
from src.data.replay import interleave


def tokenize_texts(texts: list[str], tokenizer) -> list[list[int]]:
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def build_cpt_dataset(
    primary_texts: list[str],
    replay_texts: list[str],
    tokenizer,
    block_size: int,
    replay_ratio: float,
    seed: int = 0,
) -> Dataset:
    eos = tokenizer.eos_token_id
    primary_blocks = pack_token_lists(tokenize_texts(primary_texts, tokenizer), block_size, eos)
    replay_blocks = (
        pack_token_lists(tokenize_texts(replay_texts, tokenizer), block_size, eos)
        if replay_texts and replay_ratio > 0
        else []
    )
    blocks = (
        interleave(primary_blocks, replay_blocks, replay_ratio, seed)
        if replay_blocks
        else primary_blocks
    )
    return Dataset.from_dict({"input_ids": blocks, "labels": [list(b) for b in blocks]})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/data/test_cpt_corpus.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit** (VM에서)

```bash
git add src/data/cpt_corpus.py tests/data/test_cpt_corpus.py
git commit -m "feat(data): CPT 코퍼스 빌더 (토큰화+패킹+replay)"
```

---

### Task 4: 설정 파일 + 빌드 CLI (`build_cpt_data.py`)

**Files:**
- Create: `compression/configs/cpt_data.yaml`, `compression/scripts/build_cpt_data.py`

**Interfaces:**
- Consumes: `build_cpt_dataset` (Task 3), 실제 Phi-4 토크나이저(`transformers.AutoTokenizer`), 실제 HF 데이터셋.
- Produces: 디스크에 저장된 `datasets` 데이터셋 (`--out` 경로). 스크립트는 `datasets.load_dataset`으로 소스를 읽어 `text_field`를 뽑아 `build_cpt_dataset`에 전달.

> **주의(스키마 확인):** 소스 데이터셋의 컬럼명은 사용 전 반드시 확인한다. `configs/cpt_data.yaml`의 `text_field`를 실제 카드에 맞춰 채운다. WanJuan-Korean/KoWiki의 실제 필드는 아래 Step 1에서 확인 후 config에 기입.

- [ ] **Step 1: 실제 소스 스키마 확인 (VM에서)**

Run:
```bash
uv run python -c "from datasets import load_dataset; d=load_dataset('maywell/korean_wikipedia', split='train', streaming=True); print(next(iter(d)).keys())"
```
Expected: 컬럼 키 출력(예: `dict_keys(['text', ...])`). 출력된 텍스트 필드명을 config `text_field`에 기입.
(WanJuan-Korean도 동일 방식으로 `load_dataset` streaming 후 키 확인. 접근이 gated면 HF 토큰 필요 — `.env` 참조.)

- [ ] **Step 2: 설정 파일 작성**

```yaml
# compression/configs/cpt_data.yaml
tokenizer: "microsoft/phi-4"        # 또는 프루닝 산출물 경로
block_size: 4096
replay_ratio: 0.15
seed: 0

primary_sources:
  - path: "maywell/korean_wikipedia"   # Step 1에서 확인한 실제 경로/필드로 교체
    name: null
    split: "train"
    text_field: "text"
  # - path: "opendatalab/WanJuan-Korean"  # 스키마 확인 후 추가
  #   text_field: "content"

replay_sources:
  - path: "wikitext"
    name: "wikitext-103-raw-v1"
    split: "train"
    text_field: "text"

# 스모크 테스트용 상한 (None이면 전체)
max_docs_per_source: null
```

- [ ] **Step 3: 빌드 스크립트 작성**

```python
# compression/scripts/build_cpt_data.py
"""CPT 데이터 빌드 CLI: HF 소스 로드 → build_cpt_dataset → 디스크 저장."""
from __future__ import annotations

import argparse

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.cpt_corpus import build_cpt_dataset


def _load_texts(sources: list[dict], max_docs: int | None) -> list[str]:
    texts: list[str] = []
    for s in sources:
        ds = load_dataset(s["path"], s.get("name"), split=s.get("split", "train"))
        field = s["text_field"]
        for i, ex in enumerate(ds):
            if max_docs is not None and i >= max_docs:
                break
            val = ex.get(field)
            if val:
                texts.append(val)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt_data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    max_docs = cfg.get("max_docs_per_source")
    primary = _load_texts(cfg["primary_sources"], max_docs)
    replay = _load_texts(cfg.get("replay_sources", []), max_docs)

    ds = build_cpt_dataset(
        primary_texts=primary,
        replay_texts=replay,
        tokenizer=tok,
        block_size=cfg["block_size"],
        replay_ratio=cfg["replay_ratio"],
        seed=cfg["seed"],
    )
    ds.save_to_disk(args.out)
    print(f"saved {len(ds)} blocks (block_size={cfg['block_size']}) -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 스모크 실행 (VM에서, 소량)**

`configs/cpt_data.yaml`에서 `max_docs_per_source: 100`로 임시 설정 후:
```bash
uv run python scripts/build_cpt_data.py --config configs/cpt_data.yaml --out /tmp/cpt_smoke
```
Expected: `saved N blocks (block_size=4096) -> /tmp/cpt_smoke` 출력, 에러 없음. `/tmp/cpt_smoke`에 `dataset_info.json` 생성 확인.
확인 후 `max_docs_per_source`를 다시 `null`로 되돌린다.

- [ ] **Step 5: Commit** (VM에서)

```bash
git add configs/cpt_data.yaml scripts/build_cpt_data.py
git commit -m "feat(data): CPT 데이터 빌드 CLI + 설정"
```

---

## Self-Review

**1. Spec coverage:** 스펙 §5 "CPT (raw 텍스트)" 데이터 준비를 이 플랜이 담당(WanJuan-KO/KoWiki + 영어 replay 15% + packing seq 4096). ✅ SFT 데이터(§5 SFT)는 기존 `src/distill` 재사용 검토가 필요해 Plan 2로 분리 — 의도된 범위 밖.
**2. Placeholder scan:** 모든 코드 단계에 실제 코드 포함. `text_field` 확정은 Task 4 Step 1(실제 스키마 확인)으로 처리 — 추측 대신 확인 절차. ✅
**3. Type consistency:** `pack_token_lists`/`interleave`/`build_cpt_dataset` 시그니처가 Task 1→2→3에서 일관. `tokenizer.eos_token_id`·`tokenizer(text, add_special_tokens=False)["input_ids"]` 인터페이스가 FakeTokenizer(테스트)와 AutoTokenizer(실사용) 모두에서 성립. ✅

---

## 다음 플랜 (예정)
- **Plan 2**: SFT 대화데이터 (기존 `src/distill` 재사용 + smol-koreantalk/KoAlpaca-RealQA 어댑터, 챗 템플릿, assistant-only 라벨 마스킹)
- **Plan 3**: 평가 하니스 (KMMLU/PPL/Ko-IFEval + LogicKor·GPT judge/자연스러움/영어섞임)
- **Plan 4**: 프루닝 스윕 + CPT full-FT 학습 (Unsloth)
- **Plan 5**: SFT LoRA + GGUF 양자화 + Jetson 배포/실측
