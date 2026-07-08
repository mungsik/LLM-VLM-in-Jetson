# Phi-4 한국어 경량화 파이프라인 — 실행 결과 (2026-07-06)

**목표:** Phi-4를 경량화해 **Jetson Orin Nano 8GB**에서 자연스러운 한국어 대화가 돌게.
**파이프라인:** `Phi-4-14B → ShortGPT 프루닝 → 한국어 CPT → SFT → Q3 양자화 → Jetson`

---

## 최종 결과 요약

| 단계 | 방법 | 결과 |
|---|---|---|
| ① 프루닝 | **ShortGPT** BI depth pruning | 40층 → **26층 (9.89B)**, ratio 0.35 |
| ② CPT | full-FT, 110M 토큰(KoWiki+영어replay15%) | **한국어 PPL 108.6 → 4.45** |
| ③ SFT | LoRA, 57k 대화(멀티턴) | 대화 능력 부여 |
| ④ 양자화 | llama.cpp GGUF **Q3_K_M** | ~4.8GB (Jetson 8GB fit) |
| ⑤ 배포 | Jetson Orin Nano | GGUF 파일 → 물리 기기에서 실측(별도) |

---

## ① 프루닝 스윕 (ShortGPT depth, Block Influence 기반)

phi-4 40개 레이어의 BI 측정 → 낮은 레이어 제거. 4개 비율 실측:

| ratio | 크기 | KMMLU(프루닝후) | Q4 파일 | Q3 파일 | 8GB fit |
|---|---|---|---|---|---|
| 0.25 | 11.25B | 0.360 | 6.33GB | 5.48GB | ❌ |
| **0.35** | **9.89B** | **0.351** | 5.56GB | 4.82GB | Q3 |
| 0.40 | 9.21B | 0.296 | 5.18GB | 4.49GB | ✅ |
| 0.45 | 8.53B | 0.288 | 4.80GB | 4.16GB | ✅ |

- **품질 절벽이 0.35↔0.40 사이** (0.351 → 0.296). 0.35 = 절벽 직전 최고품질.
- Q4로는 아무것도 8GB fit 안 됨 → **Q3 필요**.
- **선택: ratio 0.35 @ Q3** (품질 우선, CPT로 복구 전제).
- 제거된 레이어: 중후반 저-BI 레이어 연속 제거 (`Layers kept: [0~22, 25, 26, 38]`).

## ② 한국어 CPT (full-FT, 손상복구 + 한국어주입)

- **데이터:** KoWiki(`wikimedia/wikipedia` 20231101.ko) 40k docs + 영어 replay 15%(`Salesforce/wikitext`) → **110M 토큰**(26,868 blocks × 4096).
- **학습:** HF Trainer + paged_adamw_8bit(bnb), 유효배치 32, LR 3e-5, cosine, 1 epoch, 8h9m (RTX PRO 6000 Blackwell 96GB).
- **loss: 4.5 → 1.62.**

**평가 (held-out, 학습 이후 구간):**

| 지표 | 프루닝 직후(CPT 전) | CPT 후 | 판정 |
|---|---|---|---|
| **한국어 PPL** | 108.6 | **4.45** | ✅ **24배 개선** |
| **영어 PPL** | 164.6 | **19.3** | ✅ 망각 없음(오히려 회복, replay 효과) |
| **KMMLU** | 0.351 | **0.387** | ✅ 지식 회복(원본 phi-4 0.408 근접) |

→ **가설 입증:** 프루닝으로 손상(PPL 108/164)된 모델을 CPT가 **①손상복구 + ②한국어특화 + ③영어유지**를 한 번에 달성. Plan 3 하니스 경고 플래그 **0건**.

## ③ SFT (LoRA, 대화 능력)

- **데이터:** `smol-koreantalk`(13.6k) + `beomi/KoAlpaca-RealQA`(14.4k×2, 네이티브 앵커) + `MarkrAI/KoCommercial`(14.9k) → **57,249 대화**. 영어오염 정제 필터 적용.
- **학습:** LoRA(r=32, 72M param=0.73%) on phi4-cpt, 1500 steps, LR 1e-5, batch2×accum8, ~1.5h. 끝에 merge_and_unload로 표준 모델 저장.
- 산출물: `artifacts/phi4-sft`.

## ④ Q3 양자화 (Jetson fit)

- llama.cpp: HF → GGUF f16 → **Q3_K_M** 양자화.
- 산출물: `artifacts/phi4-sft-Q3_K_M.gguf` (~4.8GB, 8GB Orin Nano fit).
- **최종 GGUF 크기는 Telegram(Hermes) 완료 알림에서 확인.**

## ⑤ Jetson 배포 (별도)

- 위 GGUF를 **실제 Jetson Orin Nano 8GB**에 올려 llama.cpp로 구동 → 메모리 fit + tok/s 실측.
- 물리 기기 필요 → 원격 자동화 범위 밖. GGUF 파일까지 완성.

---

## 산출물 위치 (VM `phi4-blackwell`, `compression/artifacts/`)
- `prune_sweep/ratio_0.35/` — ShortGPT 프루닝 베이스 (9.89B)
- `phi4-cpt/` — CPT 완료 모델 (한국어)
- `phi4-sft/` — SFT 완료 모델 (대화)
- `phi4-sft-Q3_K_M.gguf` — **최종 배포 산출물**

## 인프라 메모
- 학습: GCP `phi4-blackwell` (RTX PRO 6000 Blackwell 96GB, asia-east1-a).
- CPT 트레이너 = HF Trainer + paged_adamw_8bit (Unsloth 아님 — Blackwell sm_120/torch2.12 env 안전).
- VM 정지: Cloud Scheduler `phi4-blackwell-oneshot-stop` (02:00 KST 1회성) — **다음 세션에서 이 잡 삭제 권장**.
- 브랜치: `feat/prune-sweep-cpt` (VM). 코드 커밋됨(80ac5f4..b282199 + eval/data fix).
