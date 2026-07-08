# Phi-4 구조적 프루닝 방법 종합 비교 (2026-06-24, 발표용)

Phi-4(14.66B) → Jetson Orin Nano 8GB 목표. 구조적 프루닝 방법들을 실제 Phi-4에 적용·비교.
실험 GPU: RTX PRO 6000 Blackwell 96GB(GCP, asia-east1) / A100 40GB×2(asia-northeast3).

---

## 1. 핵심 비교

### width / depth 프루닝 — KMMLU (한국어 보정셋, limit=20)

| 방법 | 축 | 중요도 | 감축률 | KMMLU | 원본대비 |
|---|---|---|---:|---:|---:|
| 원본 Phi-4 | — | — | — | 34.11% | 100% |
| **depth (ShortGPT)** | depth | Block Influence | 27.9% | **31.89%** | **~94%** |
| width (Minitron) | width(MLP) | 활성값 | 22.5% | 18.33% | ~54% |
| width (magnitude) | width(MLP) | weight \|W\| | 22.5% | 7.33% | ~21% |

→ **depth(ShortGPT) 압도적**(더 줄이고 정확도 거의 유지) · **활성값 ≫ magnitude**(같은 30% MLP인데 2.5배).

> 측정 조건 주의: 위 표는 메서드 간 공정 비교를 위해 **모두 limit=20(45과목×20=900문항)** 으로 통일. 900문항이라 ±2~3%p 노이즈가 있어 **depth≫width·활성값≫magnitude 같은 큰 격차 결론은 견고**하나, depth 31.89 vs 원본 34.11 같은 근접 비교는 노이즈 구간으로 본다.
> depth 모델을 **full(35,030문항)** 로 재측정하면 **34.92%** 로, 원본(34.11%)을 오히려 소폭 상회 — 즉 depth 프루닝은 **사실상 무손실**(별도 distillation 문서와 동일 모델·full 기준).

### SliceGPT — wikitext2 PPL (공식 라이브러리)

| 모델 | sparsity | 파라미터 | PPL(원본→슬라이스) |
|---|---:|---|---:|
| Phi-3-mini | 25% | 3.82B → 3.24B (15.25%↓) | 6.01 → 9.30 |
| **Phi-4** | 30% | 14.66B → 11.48B (21.7%↓) | **6.46 → 9.15** |

→ SliceGPT는 **hidden 차원**을 PCA로 슬라이스(width의 다른 축). PPL 상승은 회복 파인튜닝 전이라 정상.
※ SliceGPT는 KMMLU 아닌 PPL/wikitext2(영어) 기준이라 위 KMMLU 표와 직접 비교 불가 — 별도 지표.

---

## 2. 구현 — 라이브러리 검증 결과 (직접 돌려본 결론)

| 라이브러리 | 결과 | 근거 |
|---|---|---|
| **Torch-Pruning** | ❌ transformers 5.x Phi-4 추적 중 무한루프 폭주 | **CPU·GPU 둘 다** 직접 확인(90초 timeout/38분). device 무관, 호환성 문제 |
| **LLM-Pruner** | ❌ Phi-4 적용 불가 | 엔진=torch-pruning(위 폭주). requirement.txt=`transformers>=4.28`(느슨→5.x→폭주). 지원모델=LLaMA계열, **Phi 미지원**. 실행 안 함 |
| **SliceGPT** (microsoft/TransformerCompression) | ✅ **Phi-4 정상 동작** | phi3_adapter 이름매칭만 patch(`Phi-3-mini`→`phi-4`). transformers 4.41 + torch 2.4.1 |

→ width(Minitron식 활성값 슬라이싱)·depth(ShortGPT Block Influence)는 **PyTorch로 직접 구현**(torch-pruning 폭주 회피, Phi-4 바로 동작). SliceGPT만 공식 lib 사용.

### ⚠️ 정정 (이전 기록의 오류)
- 이전에 "SliceGPT가 Blackwell/torch에서 크래시"라 기록했으나 **틀림.** 실제 원인은 실행 시 붙인 **`--ppl-only` 플래그**(="압축 말고 PPL만 측정") 때문에 슬라이싱을 건너뛰고 정상 종료한 것. 플래그 제거하니 Blackwell·A100 모두 **정상 슬라이싱**됨.

---

## 2.5 코드 리뷰(codex) → BI 패딩 마스킹 교정 (2026-06-27)

depth(ShortGPT)가 최고 성능이라 `compute_block_influence` 코드를 codex(gpt-5.5)로 리뷰 → **실제 버그 1건 발견·수정**.

**버그:** 보정 토크나이즈가 `padding="max_length"`로 패딩하면서 `attention_mask`를 버려, BI 코사인 평균이 **패딩 토큰까지 포함**해 계산됨. 패딩 구간은 입력≈출력이라 BI를 왜곡.

**측정한 오염 규모:** 보정 배치(255×1024)에서 **패딩이 전체 토큰의 48.5%** — BI의 절반이 "레이어가 패딩을 어떻게 바꾸나"에 좌우되고 있었음.

**교정(`attention_mask` 전달 + 마스킹 집계 + `use_cache=False`) 후 비교 (동일 보정데이터·단일 모델로드):**

| | BI(앞쪽 레이어 예) | 제거 레이어 (ratio 0.30) | KMMLU full |
|---|---|---|---:|
| old (패딩 포함) | L0=0.716, L1=0.409 | [23, 27–37] | 34.92% |
| **new (패딩 마스킹)** | L0=0.444, L1=0.181 | [27–37, **39**] | **34.92%** |

- BI **순위상관 0.973** → "중후반 레이어가 잉여"라는 큰 결론은 그대로.
- 제거 12개 중 **11개 동일, 1개만 플립**(layer 23↔39, 둘 다 진짜 저-BI 잉여 레이어).
- **KMMLU full은 34.92%로 불변** → depth 프루닝은 이 보정 노이즈에 **견고**.

→ 결론: 패딩 버그는 실재·대규모였으나 **결과를 뒤집지 않고 신뢰도를 높임**. (코드 수정·테스트는 `recovery-task0-6` 반영 대상)

---

## 3. 인사이트 (발표 포인트)

1. **structured pruning 두 축**: width(차원) vs depth(레이어). 우리 결과 **Phi-4엔 depth가 유리**(잉여 레이어 제거 → 손실 최소).
2. **중요도 기준이 결정적**: 활성값 기반(Minitron) ≫ weight magnitude. "한국어에서 실제 반응하는 뉴런"을 살림.
3. **Block Influence**로 레이어 잉여성 측정 → 중간~후반 레이어(27~37)가 잉여, 레이어 0·끝부분 중요 (ShortGPT 논문 Phi-4 재현).
4. **structured만 메모리 실감소** (llama.cpp dense). masking(Wanda 등)·2:4는 sparse 런타임(TensorRT/vLLM) 한정.
5. 라이브러리는 신버전 transformers와 자주 깨짐 → **직접 구현이 견고**(우리 width/depth는 Phi-4 바로 동작).

## 4. 메모리 (4bit 추정)
원본 7.33GB → depth 5.28GB / width 5.68GB → **프루닝만으로 Jetson 8GB 적재 가능 수준.**

## 5. 다음
- distillation으로 pruned student의 한국어 성능 회복(이미 진행: **depth**-pruned 10.6B distill → KMMLU 42.38%, full 기준. 별도 문서 `2026-06-23-distillation-results.md`)
- GGUF 양자화 → Jetson 실측
