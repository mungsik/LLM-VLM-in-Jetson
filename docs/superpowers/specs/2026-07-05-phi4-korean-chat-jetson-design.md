# Phi-4 한국어 대화 모델 경량화 (Jetson Orin Nano 8GB) — 설계

- **날짜**: 2026-07-05
- **상태**: 설계 확정 (구현 계획 작성 예정)
- **프로젝트**: Pseudo-Lab LLM-VLM-in-Jetson (경량화 스터디)
- **선행 문서**: `2026-06-08-phi4-compression-jetson-design.md`(초기 압축 설계), `2026-06-29-multiturn-chatbot-data-design.md`(멀티턴 데이터)

---

## 1. 목적 / 성공 기준

**목적**: Phi-4를 경량화하여 **Jetson Orin Nano 8GB 안에서** 자연스러운 **일반 한국어 대화**가
실용 속도로 돌아가게 만든다.

**성공 기준 (한 줄)**:
> "8GB Orin Nano에서 자연스러운 한국어 대화가 실용 속도로 돌아간다."

구체 판정:
- **Fit**: GGUF 모델 파일이 Jetson Orin Nano 8GB 통합메모리에 KV캐시 포함해 적재됨 (모델 파일 ≲ 4.5GB)
- **속도**: 대화 가능한 tok/s (목표는 구현 중 실측하여 확정)
- **품질**: 한국어 대화가 번역투 없이 자연스럽고 지시를 따름 (아래 §6 평가)

**이 프로젝트는 niceinfo(나이스평가정보)와 무관하다.** 도메인은 특정 없이 **일반 대화**.

---

## 2. 제약 (고정)

- **하드웨어 = Jetson Orin Nano 8GB. 변경 불가.** 8GB 안에서 돌리는 것 자체가 프로젝트의 목적.
- 통합(LPDDR5) 메모리 8GB → OS/런타임 제외 시 모델+KV캐시에 **~5.5GB**, 실무상 모델 파일 **~4.5GB 이하** 권장.
- 공격적 압축은 회피 대상이 아니라 **본질적 챌린지**.
- **평가 judge에 개인 GPT API 사용 가능** — 이 프로젝트는 일반 대화(고객사 데이터 아님)라 niceinfo의 "외부 API 금지" 제약이 **적용되지 않음**. LogicKor/MT-Bench 표준(GPT-4급 judge)과도 정합. 학습·생성 대상은 자체 Phi 모델.

---

## 3. 파이프라인 개요

```
Phi-4-14B
  → ① 프루닝 (ShortGPT, 프루닝률 스윕)
  → ② CPT (한국어 주입 + 프루닝 복구 겸함, full-FT)
  → ③ 대화 SFT (가볍게, LoRA)
  → ④ GGUF 양자화 (fit 맞춰 Q4/Q3)
  → ⑤ Jetson Orin Nano 배포 + 실측
```

**핵심 설계 판단**:
1. **프루닝률은 고정값이 아니라 스윕 파라미터** — 8GB fit이 목표함수, 품질-크기 곡선을 그려 결정.
2. **순서 = 프루닝 → CPT** (기존 메모의 "CPT→압축"과 반대). 프루닝은 사후 복구학습이 필수인데,
   그 복구를 한국어 CPT가 겸하게 하여 한 번의 학습으로 ①손상복구 ②한국어주입을 동시 처리.
3. **CPT = full-FT** (QLoRA는 새 언어 주입에 약함). pruned 모델이 작아 단일 96GB GPU에서 가능.
4. **SFT는 절제** — chat-v1 wash 재발 방지 (원인: 번역투 지배 + 과한 SFT).

---

## 4. 단계별 상세

### ① 프루닝
- **입력**: Phi-4-14.7B
- **방법**: ShortGPT BI(Block Influence) depth pruning + 필요 시 width(`structured_prune`). 기존 `compression/` 코드 활용.
- **프루닝률**: **스윕**. 후보 25 / 35 / 45%. config 파라미터(`configs/prune_phi4.yaml`의 ratio).
- **목표**: 이후 양자화와 합쳐 8GB fit. 양자화 레벨과 커플링된 레버.
- **fit 역산(근사)**:
  | 프루닝 | 크기 | 양자화 | 파일 | 8GB |
  |---|---|---|---|---|
  | ~40% | ~9B | Q3_K_M | ~4.5GB | ✅ |
  | ~50% | ~7B | Q4_K_M | ~4.4GB | ✅ |
- **calibration**: 한국어 텍스트로 BI 측정(프루닝 대상 레이어 선택). 데이터 자유(§5).

### ② CPT (Continued Pre-Training = 한국어 mid-training)
- **입력**: pruned 모델(~7–9B)
- **목적**: (a) 프루닝 손상 복구 + (b) 한국어 유창성·지식 주입
- **목적함수**: causal LM (next-token), 전체 토큰 loss. 챗 템플릿 없음.
- **데이터**: §5 CPT 항목. packing, `max_seq_length=4096`.
- **하이퍼파라미터(초안)**:
  - full-FT, `learning_rate=3e-5`, `embedding_learning_rate=5e-6`(임베딩은 10배 작게)
  - `num_train_epochs=1`, `lr_scheduler=cosine`, `warmup_ratio=0.05`
  - **영어 replay 15%** 혼합(catastrophic forgetting 방지)
- **도구**: Unsloth + TRL, Blackwell 96GB 단일. (fit 안 되면 gradient checkpointing / QLoRA fallback)

### ③ 대화 SFT
- **입력**: CPT 완료 모델
- **목적**: 멀티턴 지시 따르기 / 대화 능력
- **방식**: **LoRA, 가볍게 1~2 epoch, 낮은 LR**. 멀티턴 챗 템플릿(Phi-4 포맷).
- **데이터**: §5 SFT 항목. **네이티브 한국어 우선**, 번역 데이터는 볼륨 보충용 최소화.
- **원칙**: 절제. MCQA(KMMLU)만 보고 판단 금지 — CPT 이득을 SFT가 washout하지 않게.

### ④ 양자화
- **방법**: llama.cpp `llama-quantize` → **GGUF Q4_K_M** (기본). 8GB 초과 시 **Q3_K_M**.
- 양자화 레벨은 프루닝률과 함께 fit을 맞추는 레버.

### ⑤ Jetson 배포
- Jetson Orin Nano 8GB에서 llama.cpp 구동.
- **실측**: 메모리 적재 성공 여부(fit), tok/s, 실제 한국어 대화 품질.
- 이 실측이 **성공/실패 판정의 최종 게이트**.

---

## 5. 데이터 (공인 데이터만, 라이선스 확인 전제)

출처 참조: `github.com/gyunggyung/LLM-Ko-Datasets` (커뮤니티 큐레이션), KORMo 논문(공인 소스로 CulturaX/FineWeb-2/OSCAR 명시).

### CPT (raw 텍스트)
| 데이터 | 규모 | 라이선스 | 역할 |
|---|---|---|---|
| **WanJuan-Korean** | 280GB+ | CC BY 4.0 | 메인 (대용량·상업가능) |
| KoWiki (2024) | ~500MB | CC BY-SA | 정제 보강 |
| 영어 replay (기존 코퍼스 일부) | 15% | — | 망각 방지 |

### SFT (대화)
| 데이터 | 규모 | 라이선스 | 역할 |
|---|---|---|---|
| **smol-koreantalk** | 460K | Apache 2.0 | 멀티턴 메인 |
| KoAlpaca-RealQA | 18K | CC BY-SA | **네이티브 → 자연스러움 앵커** |
| KoCommercial-Dataset | 1.44M | 상업가능 | 볼륨 |

- **폐기**: `LLM-OS-Models/LFM2.5-KO-CPT-Full-Raw-Mix` (개인 제작·비공인·금융/법률 도메인 편향·라이선스 "other").
- **주의**: 번역 데이터(sharegpt_deepl_ko 등)는 번역투 유입 → 최소화. 라이선스 "–" 데이터는 사용 전 카드 확인.

---

## 6. 평가 (단계별, judge = 개인 GPT API)

| 시점 | 지표 |
|---|---|
| 프루닝 후 | KMMLU·PPL (손상 정도, 견고성) |
| CPT 후 | 한국어 PPL↓, **영어/추론 회귀 감시**, Ko-IFEval |
| SFT 후 | 대화품질 — LogicKor/Ko-MT-Bench식 페어와이즈, 자연스러움, **영어섞임율**, 멀티턴 일관성, IDK율 |
| Jetson | 메모리 fit, tok/s |

- **원칙**: MCQA 단일 지표 신뢰 금지. 생성·대화 품질 지표 병행.

---

## 7. 컴퓨트 / 도구

- **학습**: GCP Blackwell 96GB (`phi4-blackwell`, asia-east1-a) 단일. Unsloth + TRL.
- **프루닝**: 기존 `compression/` (ShortGPT). uv 환경.
- **양자화·배포**: llama.cpp (VM에 존재) → GGUF → Jetson.
- **judge**: 개인 GPT API (평가 채점용, GPT-4급). LogicKor/MT-Bench 표준 방식.
- **git**: 이 Mac은 gitdir 깨짐 → 커밋은 VM에서. 광범위 `git add -A` 금지.

---

## 8. 리스크 & 완화

1. **공격적 프루닝(40~50%) → 대화품질 붕괴** → 프루닝률 스윕으로 보수적 시작, Q3로 fit 여지 확보, 품질 곡선 보며 결정.
2. **CPT catastrophic forgetting(영어/추론 상실)** → 영어 replay 15% + 낮은 LR + 1 epoch.
3. **SFT washout (chat-v1 재발)** → 네이티브 데이터 우선 + SFT 절제.
4. **8GB fit이 모든 걸 옭아맴** → 프루닝률·양자화레벨을 커플링된 레버로 함께 조정.
5. **번역투 유입** → 번역 데이터 최소화, 자연스러움/영어섞임율 지표로 감시.

---

## 9. 오픈 이슈 (구현 계획에서 확정)

- 프루닝률 최종값 (스윕 결과 의존)
- CPT 토큰 분량 (수억~수B 규모, 컴퓨트/시간 예산 의존)
- tok/s 목표 수치 (Jetson 실측 후 확정)
- Q4 vs Q3 최종 선택 (fit 결과 의존)
- SFT epoch/LR 정밀값

---

## 10. 범위 밖 (YAGNI)

- 어휘 확장(vocabulary expansion) — 1차 생략, CPT 효과 확인 후 고려.
- DPO/RLHF — 1차 파이프라인엔 미포함. SFT까지만.
- 도메인 특화(금융/법률 등) — 일반 대화가 목표, 특화 안 함.
- 멀티모달(VLM) — 텍스트 대화만.
- gpt-oss 등 타 베이스 모델 — Phi-4 프루닝 트랙으로 확정.
