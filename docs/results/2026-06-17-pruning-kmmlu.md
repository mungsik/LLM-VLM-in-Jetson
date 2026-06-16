# Phi-4 구조적 프루닝 방법 비교 + KMMLU 결과 (2026-06-17)

## 환경
- GPU: NVIDIA **RTX PRO 6000 Blackwell 96GB** (GCP `phi4-blackwell`, `g4-standard-48`, asia-east1-a)
  - 드라이버 580, torch 2.12.0+cu130 (Blackwell sm_120 지원)
- 모델: `microsoft/phi-4` (14.66B, Phi-3 아키텍처, fused gate_up_proj)
- 보정셋: `MarkrAI/KoCommercial-Dataset`(한국어, 우리 프루닝) / wikitext2(SliceGPT 기본, 영어)
- 평가: KMMLU (lm-eval, limit=20, 4지선다)

## 핵심 비교표 (KMMLU, limit=20)

| 방법 | 축 | 중요도 | 감축률 | KMMLU | 원본 대비 | 구현 |
|---|---|---|---|---|---|---|
| 원본 Phi-4 | — | — | — | **34.11%** | 100% | — |
| **depth (ShortGPT)** | depth | Block Influence | **27.9%** | **31.89%** | **~94%** 🏆 | 직접 구현 |
| width (Minitron) | width(MLP) | **활성값** | 22.5% | 18.33% | ~54% | 직접 구현 |
| width (magnitude) | width(MLP) | weight \|W\| | 22.5% | 7.33% | ~21% | 직접 구현 |
| SliceGPT | width(hidden) | PCA | (미완) | — | — | 공식 lib |

메모리(4bit 추정): 원본 7.33GB → depth 5.28GB / width 5.68GB.

## 발견

1. **depth(ShortGPT)가 압도적**: 더 많이 줄이고(27.9%) 정확도는 거의 안 떨어짐(34.11→31.89, −2.2점). Block Influence 측정 결과 **레이어 0이 가장 중요, 중간~후반(27~37)이 잉여** → 논문 주장 Phi-4에서 재현. 잉여 레이어 제거라 손실 최소.
2. **활성값 >> magnitude** (width): 같은 30% MLP 프루닝인데 18.33% vs 7.33% (**2.5배**). "어떤 뉴런 남기냐"가 결정적 → Minitron이 활성값 쓰는 이유 입증.
3. 모두 **distillation 회복 전** 수치. depth는 회복 거의 불필요(94%), width는 회복 필요.
4. caveat: limit=20 소표본(노이즈), KMMLU=지식 객관식이라 depth에 유리할 수 있음(태스크 의존).

## SliceGPT (미완 — 내일 이어서)
- 공식 lib `microsoft/TransformerCompression` (별도 venv, transformers==4.41, torch 업그레이드 cu130).
- **Phi-4 지원 패치 필요**: `src/slicegpt/adapters/phi3_adapter.py`의 매칭 조건이 `"microsoft/Phi-3-mini-4k-instruct"` 하드코딩 → `"microsoft/phi-4"`도 허용하도록 수정(line 240,260). 패치 후 **Phi-4 로딩 성공**.
- **원본 wikitext2 PPL = 6.46** 측정됨.
- **문제**: 슬라이싱(PCA 회전 계산) 단계에서 **에러·OOM 없이 조용히 죽음**(faulthandler로도 스택 안 잡힘). RAM 충분(172GB여유), OOM 아님 → Blackwell+Phi-4 회전 연산 저수준 세그폴트 추정. **미해결.**
- 내일 재현: 인스턴스 재시작 → `~/TransformerCompression`에서 `run_slicegpt.py --model microsoft/phi-4 ...`. 디버깅 방향: (a) Phi-3-mini로 먼저 동작 확인 (b) CPU device로 회전만 시도 (c) torch eigh 등 특정 연산 격리.

## 산출물 (인스턴스 디스크, 정지 상태로 보존)
- `phi4-blackwell:~/LLM-VLM-in-Jetson/compression/artifacts/`
  - `phi4-pruned-act` (활성값 width) / `phi4-pruned-smoke` (magnitude width) / `phi4-pruned-depth` (ShortGPT)
- 장기 보존 필요시 GCS 업로드 권장 (인스턴스 삭제 시 소실).

## 다음
- SliceGPT 크래시 디버깅 마무리 (또는 Phi-3-mini로 데이터포인트 확보)
- distillation(PR3)으로 width 하락 회복 → depth+distill 조합
- GGUF 양자화 → Jetson 벤치
