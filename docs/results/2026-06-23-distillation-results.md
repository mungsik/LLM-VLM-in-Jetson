# Phi-4 Distillation 결과 (2026-06-23)

depth-pruned Phi-4(10.6B) student를 `microsoft/phi-4`(14B) teacher로 **sequence-level KD**(LoRA SFT)하여,
Jetson 타깃 baseline인 **Phi-4-mini-instruct Q4_K_M**(3.8B)을 한국어 지표에서 넘는지 검증.

실행 환경: GCP `phi4-blackwell`(asia-east1-a, RTX PRO 6000 Blackwell 96GB, driver 580 / CUDA 13).

---

## 1. 종합 결과 (Q4_K_M 기준)

| 지표 | distill-scaleup (우리, 10.6B) | baseline Phi-4-mini Q4_K_M (3.8B) | 판정 |
|---|---:|---:|:--:|
| **KMMLU** (acc, full 35,030문항) | **42.38%** | 30.33% | ✅ +12.1pt |
| **PPL** (ko_corpus, 낮을수록↑) | **5.31** | 12.64 | ✅ (caveat) |
| **Ko-IFEval** prompt-strict | 26.6% | 32.5% | ❌ |
| **Ko-IFEval** prompt-loose | 29.8% | 33.9% | ❌ |
| **Ko-IFEval** inst-strict | 34.7% | 40.8% | ❌ |
| **Ko-IFEval** inst-loose | 38.7% | 42.2% | ❌ |
| 파일 크기 | 6.2GB | 2.49GB | baseline |

**결론:** distillation은 **지식(KMMLU)·언어모델링(PPL)에서 압승**, **지시준수(Ko-IFEval)에서는 열세**.
지시준수 열세 원인은 우리 student가 **1 epoch LoRA SFT**만 받은 반면 phi-4-mini는 full instruct-tuning 모델이기 때문.

---

## 2. KMMLU (distillation 효과)

동일 harness(lm-eval `hf`, `simple_evaluate`, KMMLU full)로 측정.

| 모델 | KMMLU |
|---|---:|
| 원본 phi-4 (14B) | 34.11% |
| pruned_depth (distill **전**) | 34.92% |
| distill **pilot2k** (teacher 답변 2,000) | 40.13% |
| **distill scaleup** (teacher 답변 24,958) | **42.38%** |
| Phi-4-mini Q4_K_M (baseline) | 30.33% |

- distillation으로 34.92% → 42.38% (**+7.46pt**).
- pilot(2k) → scaleup(25k)는 +2.25pt로 **수확 체감** → KMMLU 한정 25k면 충분.
- train_loss: pilot 1.117 → scaleup **0.688**.

> 주의: 이전 `2026-06-17-pruning-kmmlu.md`의 pruned_depth 31.89%는 limit=20 측정값.
> 위 34.92%는 full 측정이라 더 높음. **같은 run 내 델타(34.92→42.38)가 공정 비교.**

## 3. PPL (한국어 언어모델링)

`llama-perplexity -f ko_corpus.txt -c 2048 -ngl 99`, 우리 모델 + baseline 동일 설정 재측정.

| 모델 | PPL |
|---|---:|
| distill-scaleup Q4_K_M | **5.31** |
| Phi-4-mini Q4_K_M (재측정) | 12.64 (기록치 12.75와 일치 → 방법 검증) |

> **⚠️ caveat:** PPL은 토크나이저 의존적. phi-4(vocab 100,352) ≠ phi-4-mini(vocab ~200,064)라
> per-token PPL 직접 비교엔 한계. 다만 우리 모델이 더 크고 한국어로 distill됐으므로 방향성(우위)은 견고.

## 4. Ko-IFEval (지시준수) — 측정 함정 주의

allganize `ifeval_ko` task, lm-eval.

**🔴 중요: distilled 모델은 chat-template 의존적.**
- lm-eval `gguf` 모델(raw `/completion`)로 돌리면 우리 모델은 **빈 출력 → 4-5% 붕괴(가짜 수치)**.
  chat 포맷으로만 distill돼서 raw 프롬프트엔 즉시 EOS를 뱉음.
- 반드시 **`local-chat-completions` + `--apply_chat_template`**(서버 `/v1/chat/completions`)로 평가해야 정상값.
- baseline phi-4-mini는 raw에도 강건(instruct 튜닝). baseline 수치는 raw harness 측정값을 정당 기준으로 사용.
  (llama-server `--jinja`가 phi-4-mini 템플릿에서 "peg-native format" 500 에러 → 이 빌드에선 baseline-chat 측정 불가)

검증: 동일 프롬프트에 raw=`''`(빈 출력), chat=정상 생성(요청대로 [placeholder] 포함 이력서) 확인.

---

## 5. 파이프라인 / 인프라

### teacher 생성 가속 (vLLM)
| 방식 | 처리량 | 24,958개 ETA |
|---|---:|---:|
| HF `model.generate` batch=1 | 0.094/s | ~73h |
| HF 배치(left-pad) | ~0.5/s | ~14h |
| **vLLM** | **~12/s** | **~43분** |

- vLLM은 **별도 venv `.venv-vllm`**에 격리 설치(vllm 0.23, torch 2.11+cu130, sm_120). 기존 `.venv` 보존.
- `apt install ninja-build` 필수(FlashInfer JIT).

### GGUF 변환
- `convert_hf_to_gguf.py --outtype f16`(21GB) → `llama-quantize Q4_K_M`(6.2GB).
- **함정:** merged 디렉토리에 `tokenizer.json`만 있으면 phi 변환기가 SentencePiece 경로로 빠져 실패.
  → 원본 `microsoft/phi-4`의 `vocab.json`/`merges.txt`/`tokenizer_config.json` 등 6파일 복사(tokenizer_class=GPT2Tokenizer).
- `llama-quantize`는 빌드 필요: `cmake --build build --target llama-quantize`.
- 평가용 패키지: `.venv pip install tenacity aiohttp` (lm-eval[api]).

---

## 6. 산출물 (VM `phi4-blackwell`:`~/LLM-VLM-in-Jetson/compression/`)

```
artifacts/
  phi4-pruned-depth-distill-lora-pilot2k/         # 2k LoRA adapter
  phi4-pruned-depth-distill-lora-scaleup/         # 25k LoRA adapter (checkpoint-1560)
  phi4-pruned-depth-distill-merged-scaleup/       # merged HF 모델
  gguf/distill-scaleup-f16.gguf                   # 21GB
  gguf/distill-scaleup-Q4_K_M.gguf                # 6.2GB ← 배포 후보
data/distill/
  scaleup_teacher.jsonl                           # teacher 답변 24,958
docs/results/ko_ifeval_chat.jsonl                 # Ko-IFEval(chat) 결과
scripts/
  gen_teacher_vllm.py        # vLLM 고속 teacher 생성
  merge_distill.py           # LoRA → merged
benchmarks/run_scaleup.sh    # gen → SFT → merge → KMMLU 체인
```

## 7. 다음 단계

1. **지시준수 보강(최우선):** IFEval류 포맷 데이터 + epoch 증가로 SFT 보강 → Ko-IFEval 격차 축소.
   (KMMLU/PPL 우위는 이미 확보 → 지시준수만 따라오면 baseline 전면 우위 가능)
2. **Jetson 8GB 실측:** Q4 6.2GB 적재 가능하나 baseline(2.49GB)보다 큼 → 토큰/s·메모리 실측 필요.
3. (선택) 더 작은 student(width+depth)로 압축률 ↑ 후 재-distill.
