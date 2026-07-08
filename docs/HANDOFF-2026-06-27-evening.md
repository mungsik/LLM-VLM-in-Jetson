# HANDOFF — 2026-06-27 저녁 이어작업용 (다른 컴퓨터)

이 문서는 iCloud 동기화되는 워킹트리에 있으니 다른 컴퓨터에서 그대로 보입니다.
(메모리는 머신별로 갈리므로, 크로스머신 컨텍스트는 이 파일이 전달자입니다.)

---

## 오늘(6/27) 한 일 요약

### A. codex 코드리뷰 → BI 패딩 버그 수정·재검증 (완료)
- **버그:** `compute_block_influence`가 `padding="max_length"` 패딩 토큰을 `attention_mask` 없이 BI 코사인 평균에 포함 → 보정배치의 **48.5%가 패딩**으로 BI 오염.
- **수정 파일(로컬·VM 반영, git 미커밋):**
  - `compression/src/prune/calibration.py` — `tokenize_texts(..., return_mask=True)` 추가(기본은 Tensor 반환, 하위호환)
  - `compression/src/prune/depth_prune.py` — `compute_block_influence`가 dict 배치({input_ids, attention_mask}) 허용 + 마스킹 집계 + `use_cache=False` + 훅 try/finally
  - `compression/scripts/run_depth_prune.py` — mask 받아 dict 배치 전달
  - `compression/tests/prune/test_depth_prune.py` — 마스크 반영 테스트 추가 (로컬 prune 9 passed)
- **재검증 결과(GPU):** 제거 레이어 12개 중 **11개 동일, 1개 플립**(old [23,27–37] → new [27–37,39]), BI 순위상관 0.97. **KMMLU full = 34.92%로 불변**(old=new). → 버그는 컸으나 ShortGPT depth는 견고 = 신뢰도↑.
- 결과 문서: `docs/results/2026-06-24-pruning-methods-comparison.md` **§2.5**(작성됨).
- VM 산출물: `artifacts/phi4-pruned-depth-masked`(28층, keep=[0–26,38]).

### B. distillation 모델 vLLM 서빙 + 한국어 직접 테스트 (완료)
- **distill(10.6B)** vs **원본 Phi-4(14.7B)** 2열 비교 채팅 UI 제작 → `docs/tools/phi4-compare-chat.html` (브라우저로 열면 됨, vLLM CORS=* 라 직접 호출).
- **관찰:** distill은 한국어 **유창하나 사실성·지시준수·반복에서 약함**(없는 지명 생성 등). 원본은 정확·간결. ↔ KMMLU(distill 42%>base 34%)와 별개.
- **원인 분석(결론):** KMMLU(객관식 logprob)·PPL은 **지식·언어모델링**만 측정 → distill이 회복한 영역. **대화 품질(지시준수·일관성·멀티턴)** 은 측정 안 됐고, 그건 **Ko-IFEval에서 이미 졌던 약점**(distill 26~38% < base 32~42%). 근본 원인 = **얇은 1-epoch LoRA SFT + 지식QA(단일턴) 데이터**. 프루닝 실패 아님.

---

## 열린 결정 (저녁에 이어서 정할 것)
채팅 품질 개선 방향. **멀티턴 데이터 확보가 1순위는 아님.** 우선순위:
1. **지시준수 SFT 강화**(다양한 지시·형식·제약 데이터 + epoch↑ + LoRA→full FT 검토) — 단일턴 품질이 더 큰 문제.
2. **teacher로 타깃 생성**: Phi-4 teacher가 지시·멀티턴 답변을 직접 생성(데이터 "구하기" 아님) → 재distill. (vLLM 25k/43분 파이프라인 재활용)
3. 멀티턴 데이터는 2순위.
- **먼저 확정할 것: Jetson 용도(개방형 채팅 vs 지식 QA/RAG).** QA/RAG면 현재 KMMLU 강점이 이미 쓸만 → 채팅 광택 우선순위 낮음.

---

## VM 재기동 런북 (저녁에 그대로 실행)

```bash
P=polarpulse-dev-mungsik; Z=asia-east1-a; I=phi4-blackwell
# 0) (필요시) 재인증: gcloud auth login mungsik@polarpulse.ai
# 1) 인스턴스 기동
gcloud compute instances start $I --zone=$Z --project=$P

# 2) BASE 먼저 (8001, util 0.38, enforce-eager, 16384) — SSH 안에서:
#    repo=/home/mungsik/LLM-VLM-in-Jetson/compression, 실행유저 sudo -u mungsik, venv .venv-vllm
gcloud compute ssh $I --zone=$Z --project=$P --tunnel-through-iap --command='
  B=/home/mungsik/LLM-VLM-in-Jetson/compression
  sudo -u mungsik bash -c "cd $B && nohup .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 --served-model-name phi4-base --trust-remote-code \
    --host 0.0.0.0 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.38 \
    --enforce-eager --max-num-seqs 8 > /tmp/vllm_base.log 2>&1 &"'

# 3) base "Application startup complete" 확인 후 DISTILL (8000, util 0.42, enforce-eager, 16384)
gcloud compute ssh $I --zone=$Z --project=$P --tunnel-through-iap --command='
  B=/home/mungsik/LLM-VLM-in-Jetson/compression
  sudo -u mungsik bash -c "cd $B && nohup .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model artifacts/phi4-pruned-depth-distill-merged-scaleup --served-model-name phi4-distill-ko \
    --chat-template artifacts/phi4-pruned-depth-distill-merged-scaleup/chat_template.jinja --trust-remote-code \
    --host 0.0.0.0 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.42 \
    --enforce-eager --max-num-seqs 8 > /tmp/vllm_distill.log 2>&1 &"'

# 4) 터널 2개 (각각 백그라운드 유지)
gcloud compute ssh $I --zone=$Z --project=$P --tunnel-through-iap -- -N -L 8000:localhost:8000 &
gcloud compute ssh $I --zone=$Z --project=$P --tunnel-through-iap -- -N -L 8001:localhost:8001 &

# 5) UI 열기
open docs/tools/phi4-compare-chat.html
```

### 메모 / 주의
- **순서 중요:** base→distill 순차 기동(동시 기동 시 KV 캐시 메모리 부족으로 실패). util 합 0.80, `--enforce-eager` 필수(메모리 절약).
- vLLM은 **`.venv-vllm`**(vllm 0.23), 프루닝 코드 테스트는 **`.venv`**(py3.11).
- distill 모델은 **chat-template 전용** — `/v1/chat/completions`만 사용(raw `/v1/completions`엔 빈 출력).
- max_tokens는 입력+출력 ≤ 16384. 초과 시 400 거부(UI가 이제 에러 표시).
- **git 커밋 주의:** 이 repo 워킹트리에 다른 컴퓨터 distillation 작업이 미커밋 상태로 섞여 있음 → `git add -A` 광범위 커밋 금지.
