# Phi-4 경량화 파이프라인 for Jetson Orin Nano 8GB — 설계 문서

- **작성일**: 2026-06-08
- **레포**: Pseudo-Lab/LLM-VLM-in-Jetson
- **상태**: 설계 확정, 구현 계획(plan) 작성 직전

---

## 1. 목표 (Goal)

Phi-4(14.7B)를 **구조적 프루닝 → distillation 회복 → GGUF 양자화** 파이프라인으로 압축하여,
**Jetson Orin Nano 8GB**에서 구동 가능한 형태로 만든다.

이번 단계의 **성공 기준**: *동작하는 시작 코드 + GitHub PR 공유*.
(전체 정확도 회복·E2E 완성은 후속 단계. 1차는 "경량화 코드 구현의 시작"을 보여주는 것이 목적.)

**배포 타겟**: **범용 한국어** (Jetson Orin Nano에서 **에이전트**로 활용). → distillation은 한국어 데이터, 평가는 한국어 벤치(KMMLU)로 진행.

### 비교 대상 모델

| 모델 | 역할 |
|---|---|
| Phi-4 (14.7B) 원본 | 압축 baseline 겸 distillation teacher |
| **Phi-4-mini (3.8B)** | 동일 계열 "작은 모델" 성능 비교 기준 (오프라인 테스트 공유용) |
| 압축본 (pruned → distilled → quantized) | 본 프로젝트 결과물 |

---

## 2. 핵심 제약: 왜 3종 기법을 모두 쓰는가

Jetson Orin Nano 8GB는 CPU/GPU가 8GB LPDDR5를 **공유**하고, OS+CUDA 컨텍스트가 1.5~2GB를 점유한다.
실사용 가능 메모리는 약 **5.5~6.5GB**.

| 구성 | 가중치 메모리 |
|---|---|
| Phi-4 (14.7B) FP16 | ~29 GB |
| INT4 (4-bit) | **~7.3 GB** ← 단순 4비트로는 OOM 위험 |
| INT3 | ~5.5 GB |
| INT2 | ~3.7 GB |

→ 단순 4비트 양자화만으로는 8GB에 안전하게 안 들어간다.
따라서 **구조적 프루닝(파라미터 차원 자체 감축) + distillation(성능 회복) + 양자화(배포)** 를 모두 적용하는 것이
단순 학습 목적이 아니라 *14B를 8GB에 넣기 위한 정공법*이다.

---

## 3. 아키텍처: 3-스테이지 파이프라인

```
                  ┌─ calibration data ─┐
Phi-4 (HF, BF16) ─┤                    ├─→ [1] 구조적 Pruning ─→ pruned (≈8~10B)
   │ (teacher)    └────────────────────┘                              │
   │                                                                   ▼
   └──────────────── logit/hidden KD ──────────→ [2] Distillation 회복학습 ─→ pruned+distilled
                                                                       │
                                                                       ▼
                                            [3] HF→GGUF + imatrix 양자화 (Q4/IQ3/IQ2)
                                                                       │
                                                                       ▼
                                                  Jetson Orin Nano 8GB (llama.cpp)
```

### [1] Pruning — Minitron 방식 (구조적 width + depth)

- **기법 확정: NVIDIA Minitron** (활성값 기반 구조적 프루닝). 근거는 §6.
- **무엇을 자르나 (2축):**
  - *Width*: MLP intermediate 뉴런, attention/KV head, hidden 채널
  - *Depth*: 중요도 낮은 transformer 레이어 통째로 제거
- **중요도 지표**: 활성값 기반(activation magnitude). 소규모 보정셋(~1024 샘플)을 forward만 통과시켜 산출(gradient 불필요), single-shot.
- **Phi-4 특화 주의:**
  - **결합구조 그룹핑 필수** — 잔차 스트림을 공유하는 가중치(한 head의 Q/K/V/O, MLP의 up/gate/down)는 함께 잘라야 차원이 깨지지 않음.
  - **GQA 존중** — Phi-4의 grouped-query attention 때문에 KV head는 쿼리 그룹 단위로 프루닝.
- **목표 압축률(초기값, 튜닝 대상)**: 14.7B → ~9~10B (약 30~35% 감축). 8GB 예산에 맞춰 조정.

> **첫 구현(PR2)은 Minitron식 활성값 기반 width 프루닝**으로 시작한다.
> 즉 보정셋 forward로 뉴런/head 중요도를 산출해 width를 우선 줄이고(결합구조 그룹핑·GQA 존중),
> 이후 depth 프루닝과 distillation 회복을 더해 Minitron 풀 파이프라인으로 확장한다. (단일 기법으로 일관 유지)

### [2] Distillation — 회복 학습

- **Teacher = 원본 Phi-4 (BF16) 고정**, **Student** = 프루닝본. (teacher 교체 불가 사유는 §6 카드 참조)
- **Loss**: logit KL divergence + (옵션) hidden-state/중간블록 distillation. (Minitron: embedding + logit + 중간블록 loss)
- **데이터 = 범용 한국어** (에이전트 용도이므로 텍스트 유창성 + instruction-following 둘 다 보존). 2층 구성:
  - ① **범용 한국어 텍스트**: **CulturaX(ko)** (주) + 한국어 위키(kowiki) (보강). [옵션 FineWeb-2 ko]
  - ② **한국어 instruction/agent**: **KULLM-v2** + **KoAlpaca-RealQA**(native). [옵션 KOR-OpenOrca로 추론 보강]
  - 입력 시퀀스(①텍스트 + ②instruction 프롬프트)에 대해 teacher=Phi-4의 logit을 student가 따라감 → Phi-4의 한국어 처리 방식 보존.
  - 믹스 비율 초기값 ~70%(텍스트)/30%(instruction), config로 튜닝.
- **학습 규모**: "충분한 GPU" 전제. full fine-tune 기본, LoRA는 예산 옵션. 토큰 예산은 config화하여 작게 시작 후 확장.

> **한국어 천장 주의**: teacher가 영어 중심 Phi-4라 student의 한국어 능력 상한 = Phi-4 수준. 한국어 데이터는 그 능력을 *보존*하는 용도이지 *향상*시키지 못함. (향상하려면 §6의 sequence-level KD 카드 참조)

### [3] Quantization — GGUF (llama.cpp)

- `convert_hf_to_gguf.py`로 HF→GGUF 변환 후 `llama-quantize`로 양자화.
- **imatrix 보정** 적용 — 저비트(IQ3/IQ2) 품질 방어.
- **비트 선택 전략**: Q4_K_M 우선 → 8GB 초과 시 IQ3 → IQ2로 단계 하향 (정확도 trade-off 감수).

---

## 4. 런타임 & 평가 환경

```
[학습 서버 — 큰 GPU]                       [Jetson Orin Nano 8GB]
 ├ 프루닝 실행                               └ 최종 압축모델 추론
 ├ distillation 실행                           → llama.cpp (GGUF)
 └ FP16 모델 평가 (vLLM 서빙)                   → 메모리·tok/s 실측
```

- **배포 런타임**: **llama.cpp (GGUF)** — Orin Nano 8GB에서 가장 실용적이며, 서브-4비트(IQ3/IQ2) 유연성이 핵심.
- **평가 서빙**: **vLLM** — 학습 서버에서 양자화 안 된 FP16 큰 모델(원본 Phi-4, Phi-4-mini, 프루닝본)을 고처리량으로 서빙하여 평가 가속.
- **평가 도구**: lm-eval-harness (한국어 태스크).

### 평가 지표

- **품질(주 지표) = KMMLU** (HAERAE-HUB, 한국어 MMLU·native) — 범용 한국어 지식/이해 측정.
- **보조**: 한국어 perplexity (CulturaX-ko / kowiki held-out).
- (향후 옵션) 에이전트 생성 품질 평가 — KMMLU는 객관식이라 생성 품질은 직접 측정 못 함.
- **Jetson 실측**: 메모리 풋프린트, tokens/sec, 로드 성공 여부.
- 5개 모델(원본 / mini / pruned / distilled / quantized)을 KMMLU·perplexity·Jetson 실측 한 표로 비교.

---

## 5. 레포 구조 & PR 단계

### 디렉토리 구조 (Pseudo-Lab/LLM-VLM-in-Jetson 내부)

```
compression/
  configs/            # 스테이지별 yaml 설정
  src/
    common/           # 모델 로딩·eval·utils (Phi-4 추상화 레이어 포함)
    prune/            # 구조적 pruning (Minitron)
    distill/          # KD 회복학습
    quantize/         # HF→GGUF + imatrix
    pipeline.py       # prune→distill→quant 오케스트레이션
  scripts/
    eval.py
    jetson/           # 배포·벤치마크 + 셋업 문서
  requirements.txt
  README.md
```

### PR 단계 (각 단계 독립 공유 가능)

| PR | 내용 | 비고 |
|---|---|---|
| **PR1** | scaffold + common(Phi-4/mini 로딩) + eval 하네스 골격 + README | **비교모델(Phi-4-mini) 공유 요건 충족** |
| **PR2** | `prune/` 구조적 프루닝 + before/after 파라미터·ppl 리포트 | **"구현 시작" 핵심 산출물** |
| PR3 | `distill/` KD 회복학습 | |
| PR4 | `quantize/` GGUF+imatrix + Jetson 벤치 | |
| PR5 | `pipeline.py` E2E + 결과표 | |

→ **이번 발표의 현실적 제출 범위 = PR1 + PR2(프루닝) 시작 지점까지.**

---

## 6. 기법 선정 근거 (탐색 결과)

prune→distill 계열 최신 기법을 비교한 결과, **Minitron이 본 프로젝트에 최선**으로 판정.

| 기법 | 특징 | 판정 |
|---|---|---|
| **Minitron** (NVIDIA) | 활성값 기반 구조적 width+depth + KD distillation. 논문 비교에서 LLM-Pruner·SliceGPT·LaCo·ShortGPT·Sheared-LLaMA를 모두 능가 | ✅ **채택** — prune→distill과 정확히 일치, 구현·이해 용이, ~14B급 양산 검증 |
| SliceGPT / FLAP / FASP / SlimLLM | 재학습 불필요 구조적 프루닝 | ✗ 우리는 GPU 보유 + distillation 예정이라 "무재학습" 강점이 안 살아남 |
| Wanda | 비구조적(또는 2:4) 프루닝 | ✗ **제외** (아래) |

### Wanda 제외 근거

- Wanda는 unstructured(또는 semi-structured 2:4) 프루닝 → 가중치 차원(shape)이 그대로, 일부만 0.
- **llama.cpp(GGUF)는 sparsity를 활용하지 못함** → 0이 많아도 메모리·속도 이득 0.
- 즉 우리의 배포 경로(Jetson/llama.cpp)에서 Wanda는 메모리를 줄이지 못하므로 배포 후보로서 무의미.
- 단, **TensorRT-LLM + 2:4 sparsity** 경로에서는 Orin Nano의 Ampere 텐서코어가 2:4를 HW 가속하므로 의미가 생김 → 아래 "향후 카드"로만 기록.

### Teacher 모델 = Phi-4 고정 (Qwen 등 교체 불가 사유)

distillation teacher는 **반드시 원본 Phi-4**여야 하며, Qwen 등 다른 계열로 교체할 수 없다.

1. **Vocab/tokenizer 불일치** — Minitron의 distillation은 logit/hidden state matching이라 teacher·student가 **같은 tokenizer·vocabulary**를 써야 한다. Phi-4(tiktoken 계열, vocab ~100K)와 Qwen2.5(자체 BPE, vocab ~151K)는 logit 차원·토큰 매핑이 달라 KL divergence 계산 자체가 불가능.
2. **복원 논리** — 회복학습은 "프루닝으로 망가진 Phi-4를 *원본 자기 자신*으로 되돌리는" 과정. student가 Phi-4를 잘라 만든 것이므로 teacher는 정의상 원본 Phi-4.

→ 한국어 능력을 *주입*하려면(천장 상향) logit KD가 아닌 **sequence-level KD**(teacher 생성 한국어 텍스트로 SFT)가 필요한데, 이는 tokenizer 무관하나 "Phi-4 복원"이 아닌 "새 능력 이식"이라 프로젝트 scope가 바뀜 → 아래 향후 카드.

### 향후 개선 카드 (spec 기록만, 1차 범위 외)

1. **TensorRT-LLM + Wanda(2:4)** — 다른 배포 런타임으로의 비교 실험. 2:4는 50% 고정 sparsity라 structured(Minitron)를 대체하지 않고 추가 옵션.
2. **sequence-level KD로 한국어 능력 주입** — Qwen/EXAONE 등 한국어 강한 모델이 생성한 한국어 텍스트로 student를 SFT. tokenizer 무관하게 가능하나 "Phi-4 경량화 + 타 모델 한국어 이식"으로 성격이 바뀌고 작업량 증가.

---

## 7. 리스크 & 대비

| 리스크 | 대비 |
|---|---|
| 35% 프루닝 + 4bit 후에도 8GB가 빡빡할 수 있음 | IQ3/IQ2로 비트 하향 또는 프루닝률 상향 (정확도 trade-off 감수). 폴백 경로 미리 둠 |
| Phi-4 특화 아키텍처로 프루닝 라이브러리 어댑팅 필요 | `common/`에 모델 추상화 레이어를 두어 흡수 |
| Jetson 소프트웨어 스택(JetPack/CUDA/llama.cpp 빌드) 마찰 | `scripts/jetson/`에 셋업 절차 문서화 |
| Distillation 비용/시간 | 토큰 예산을 config화, 작게 시작 후 확장 |

---

## 8. 참고 자료

- Minitron 논문: https://arxiv.org/html/2408.11796v1
- Minitron 실전 블로그(Llama-3.1-8B→4B): https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/
- SliceGPT: https://arxiv.org/pdf/2401.15024
- Wanda: https://arxiv.org/abs/2306.11695

### 한국어 데이터 & 평가
- CulturaX (다국어, ko subset): https://huggingface.co/datasets/uonlp/CulturaX
- KULLM (고려대): https://github.com/nlpai-lab/KULLM · KoAlpaca: https://github.com/Beomi/KoAlpaca
- 한국어 데이터셋 모음: https://github.com/gyunggyung/LLM-Ko-Datasets · KIT-19: https://arxiv.org/pdf/2403.16444
- KMMLU / 한국어 벤치 평가 코드: https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset
- Open Ko-LLM Leaderboard2: https://arxiv.org/abs/2410.12445
