# 설계: distill 채팅 품질 개선 — full-FT + logit-level KD (2026-06-28)

## 배경 / 문제

depth-pruned Phi-4 student(10.6B/28층)를 Phi-4 14B teacher로 distill한 현재 모델
(`phi4-pruned-depth-distill-merged-scaleup`)은:

- **이긴 곳:** KMMLU 42.38%(원본 34.11%·baseline 30.33% 압도), PPL 5.31.
- **진 곳:** 실제 한국어 채팅 체감품질 — 사실성(없는 지명 생성 등), 지시준수, 반복.

진단(2026-06-27 핸드오프): 약함의 원인은 **얇은 학습**이다.
현재 = hard-target sequence-SFT, **1 epoch, LoRA r16, train_loss 0.688**.
teacher의 "분포"가 아니라 "텍스트(argmax 시퀀스)"만 흉내냈고, 저rank·1epoch라 덜 학습됨.

## 목표

단일턴 한국어 채팅의 **체감품질(사실성·일관성·반복없음·자연스러움)** 을 올린다.
- 성공 측정 = 고정 프로브셋으로 비교 UI에서 before/after 체감 판정(지표 Ko-IFEval 무관, 사용자 결정).
- 제약 가드 = KMMLU 42% 지식 우위를 큰 회귀 없이 유지.
- 데이터는 고정(기존 25k). 주 레버는 **학습 방식**(full-FT + logit-KD).
- 단, student 베이스는 scaleup이 쓴 `phi4-pruned-depth` → 교정본 `phi4-pruned-depth-masked`로 교체(사용자 결정).
  두 베이스의 **KMMLU는 34.92%로 동일**하므로 비교의 추가 변수는 작지만, 엄밀히는 "베이스+방식" 동시 변경임을 명시.

## 전제조건 (성립 확인)

student는 Phi-4에서 **레이어만 드롭**한 모델 → teacher와 **vocab/토크나이저 완전 동일(100,352)**.
따라서 teacher logit을 student logit에 인덱스 단위로 직접 매칭하는 logit-level KD가 성립한다.
이것이 본 방법의 핵심 전제이며 충족된다.

## 학습 데이터 (고정, 신규 생성 없음)

VM의 기존 **`data/distill/pilot_teacher.jsonl`** (24,958쌍, 단일턴 `(지시 → Phi-4 답변)`).

프롬프트 출처(`scripts/prepare_distill_prompts.py`):

| source | n |
|---|---:|
| MarkrAI/KoCommercial-Dataset | 15,000 |
| MarkrAI/KOpen-HQ-Hermes-2.5-60K | 10,000 |
| nlpai-lab/kullm-v2 | 10,000 |
| squarelike/OpenOrca-gugugo-ko | 5,000 |
| synthetic-ko-ifeval-style (형식·제약 합성) | 5,000 |

teacher 답변 = Phi-4 생성(temp 0.2, max_new 768). 데이터를 손대는 것은 본 단계 밖(추후 Approach B).

## 아키텍처 — teacher logit "오프라인 사전계산" 방식

teacher를 학습 루프에 올리지 않는다. teacher를 **한 번만** 돌려 top-k logit을 디스크에 저장 →
이후 학습은 teacher-free(메모리 여유 + 하이퍼파라미터 반복이 쌈).

```
pilot_teacher.jsonl (25k 응답)
        │
        ▼  precompute_teacher_logits.py  (teacher Phi-4, 1회)
teacher_topk/<shard>.safetensors   (id별: position, top-k 인덱스·logit, fp16)
        │
        ▼  train_distill_kd.py  (full-FT student, teacher-free)
artifacts/phi4-pruned-depth-distill-kd-v1
        │
        ▼  vLLM 서빙 (8000) ↔ base (8001)
비교 UI 판정 + KMMLU 가드
```

### 컴포넌트

**1. (신규) `compression/scripts/precompute_teacher_logits.py`**
- 입력: `pilot_teacher.jsonl`, teacher `microsoft/phi-4`.
- 각 응답을 **학습과 100% 동일한 chat_template 렌더링**(`apply_chat_template(..., add_generation_prompt=False)`)으로 토크나이즈 → teacher forward.
- assistant 토큰 구간(= 학습 시 label != -100 위치)의 **position별 top-k(k=64) (인덱스, logit)** 저장.
- 정렬 안정성: 각 예제에 대해 position 인덱스를 명시 저장(id로 정렬).
- 포맷: id별 safetensors shard, logit은 fp16. 용량 ≈ 25k×~768×64 → ~10GB대.
- 한 일관성: prompt-only 구간은 마스킹(KD/CE 모두 assistant 구간만).

**2. (신규) `compression/scripts/train_distill_kd.py`** (기존 `train_lora_sft.py`는 fallback용 보존)
- student = `artifacts/phi4-pruned-depth-masked`(BI 패딩버그 교정본, 28층 keep=[0–26,38]), **full-FT**(LoRA 제거).
- 옵티마이저 `paged_adamw_8bit`, gradient checkpointing, bf16, lr **1e-5**, epoch **2**, bs1/grad-accum.
- 커스텀 `compute_loss`:
  - `Loss = α · T² · KL(teacher_topk ‖ student) + (1−α) · CE(hard token)`
  - 기본값 **T=2, α=0.9, k=64**.
  - teacher top-k를 T로 소프트닝 후 top-k 위에서 renormalize. student logit을 동일 k 인덱스에서 gather → 같은 k-simplex 위에서 KL.
  - CE는 teacher가 실제 생성한 hard token(기존 SFT 라벨)에 대해.
- collator: top-k 텐서(인덱스·logit·position)를 label position에 정렬해 패딩 처리.
- 산출물: full-FT라 merge 불필요 → `artifacts/phi4-pruned-depth-distill-kd-v1`에 바로 저장(+ chat_template.jinja 동봉).

**3. 정합성 가드(필수)**
- precompute와 train의 토크나이저·chat_template 렌더링 동일.
- sanity 테스트: 저장된 top-k의 argmax == teacher가 실제 생성한 토큰인지 일부 위치 검증.
- collator 정렬 테스트(작은 합성 배치): KD 손실이 label 구간에만 적용되고 prompt/padding은 0 기여인지.

### 메모리 (96GB RTX PRO 6000 Blackwell)

full-FT student: 가중치 21GB + grad 21GB + 8bit optim ~21GB + activations(grad ckpt, bs1) ≈ **65~70GB / 96GB → 들어감**.
teacher는 train 시 미로딩(오프라인 precompute). 빠듯할 경우 fallback: seq len↓, 또는 옵티마이저/파라미터 CPU offload.

## 평가 / 완료 게이트

- **KMMLU 가드**: 부분셋 먼저 → 통과 시 full(`src/common/eval_kmmlu.py` / lm-eval). ~42% 대비 큰 회귀 없어야(full-FT는 LoRA보다 망각 위험↑ → 핵심 가드).
- **고정 프로브셋 10~15개**(사실성·형식제약·멀티문단 등 관찰된 실패유형) before/after를 비교 UI(`docs/tools/phi4-compare-chat.html`)에서 v1(scaleup) vs kd-v1 동일 조건 판정.
- 성공 기준: 비교 UI에서 사실성·반복·자연스러움 체감 개선 + KMMLU 큰 회귀 없음.
- 실패 시 다음 게이트: 하이퍼파라미터(T·α·k·epoch) 조정 → 그래도 부족하면 Approach B(데이터 재생성).

## 리스크

| 리스크 | 완화 |
|---|---|
| 정렬 버그(가장 흔한 실패) | position 명시 저장 + argmax sanity 테스트 + collator 정렬 테스트 |
| full-FT 지식 망각 | lr 1e-5·epoch 2·KMMLU 가드 |
| 메모리 빠듯 | 오프라인 precompute로 teacher 제외, fallback seq↓/offload |
| 하이퍼파라미터 미검증 | T=2·α=0.9·k=64 1차값, 결과 보고 조정 |

## 범위 밖 (YAGNI)

- 서빙단 repetition_penalty / decoding 튜닝 — 나중에 UI에서 조정(학습 무관).
- 멀티턴 대화 — 단일턴 품질 확인 후 결정.
- 데이터 재생성·다양화 — Approach B로 보류.

## 실행 환경 메모

- VM `phi4-blackwell`(asia-east1-a, RTX PRO 6000 96GB, driver 580/CUDA 13). 학습/데이터 전부 VM.
- 학습 코드 테스트 venv `.venv`(py3.11), vLLM 서빙 venv `.venv-vllm`.
- 이 repo는 코드만 git 동기화(`data/`·`artifacts/`는 gitignore, VM-only).
- 이 Mac은 gitdir 부재 → 커밋은 원본/ VM 머신에서. 미커밋 distillation 작업 섞여 있어 광범위 `git add -A` 금지.
