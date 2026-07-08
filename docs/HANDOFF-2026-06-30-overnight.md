# HANDOFF — 2026-06-30 야간 자동 실행 (아침에 확인용)

밤새 두 트랙이 **자동 실행 → 완료 시 인스턴스 자동 종료**되도록 설정함.
**핵심 보장: 작업이 실패해도 15h 하드 백스톱 타이머가 무조건 인스턴스를 끔(과금 방지).**

---

## 트랙 1 — 멀티턴 챗봇 SFT (인스턴스 `phi4-blackwell`, asia-east1-a)
- 본런: `train_sft_multiturn.py` (Nano… 아니라 depth-pruned Phi-4 student, 2048/batch2/accum8, max-steps 4000, ~10h)
- 산출물: `artifacts/phi4-pruned-depth-chat-v1` (멀티턴 한국어 챗봇)
- **자동화:** `track1auto.service` = SFT 완료 대기 → chat-v1 **KMMLU(limit500)** 측정 → **shutdown**
- **결과 위치(영구):** `/opt/track1_RESULTS.log` (인스턴스 재시작해도 보존)
- 다음(수동): 페어와이즈 평가(chat-v1 vs scaleup vs kd-v1) + 영어섞임율/IDK 지표 — Plan 2 Task 5~6

## 트랙 2 — Minitron NeMo (인스턴스 `minitron-blackwell`, asia-east1-a)
- ✅ Blackwell에서 NeMo 25.04 동작(sm_120) + Llama-3.1-Nemotron-Nano-8B → NeMo2 import 완료
- ✅ width prune 8B→4B **파이프라인 검증**(importance 99% GPU) — **단 현재 MOCK calibration이라 pruned 품질은 무의미**
- **자동화:** `track2auto.service` = mock prune 완료 대기 → **기본 모델 KMMLU 측정** → shutdown
  - [1] `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` base KMMLU
  - [2] `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` base KMMLU (하이브리드 Mamba-Transformer)
- **결과 위치(영구):** `/opt/minitron_ws/RESULTS.md`

### ⚠️ 트랙 2에서 "아직 안 한 것" (다음 supervised 세션)
사용자가 원한 **"vanilla prune 후 한국어 성능"** 의 *의미있는* 수치는 아직임. 이유:
- 의미있는 prune은 **실 Korean calibration 데이터**(Megatron 포맷 전처리) 필요 — mock은 랜덤 프루닝.
- pruned NeMo2 → HF export → lm-eval 도 필요(export 명령 검증 필요).
- 그 다음 **distillation(KD)** 으로 회복해야 논문 수준.
→ 오늘 자동 실행은 **(a) 파이프라인 검증 + (b) 기본 모델 한국어 베이스라인**까지. 실 prune+distill 비교는 깨어있을 때 단계별로.

### 최신 기법 메모 (검색 결과, 2025~2026)
- **MiniPuzzle**(Minitron+Puzzle NAS): HW 제약 타깃 압축 → Jetson 8GB에 이상적. (신버전 ModelOpt/NeMo 필요)
- **Nemotron Elastic**: 라우터 공동학습 → 한 번에 여러 크기 student.
- **Minitron-SSM**(NeurIPS'25): Mamba/하이브리드 압축.
- 모델 트렌드: 하이브리드 Mamba-Transformer(**Nemotron-3 Nano 4B** = 로컬AI용, Jetson 직접 후보).
- 참고: **Bielik-Minitron-7B**(폴란드어) = 언어특화 Minitron 선례, 우리 한국어와 동일 패턴.

---

## 아침에 결과 보는 법
두 인스턴스는 작업 끝나면 **STOPPED** 상태(과금 없음). 결과 로그는 디스크에 보존됨:
```bash
P=polarpulse-dev-mungsik; Z=asia-east1-a
# 트랙1
gcloud compute instances start phi4-blackwell --zone=$Z --project=$P
gcloud compute ssh phi4-blackwell --zone=$Z --project=$P --tunnel-through-iap --command='sudo cat /opt/track1_RESULTS.log'
# 트랙2
gcloud compute instances start minitron-blackwell --zone=$Z --project=$P
gcloud compute ssh minitron-blackwell --zone=$Z --project=$P --tunnel-through-iap --command='sudo cat /opt/minitron_ws/RESULTS.md'
# 확인 후 다시 stop 잊지 말 것
```
(상태 확인: `gcloud compute instances list --project=$P --filter="name~blackwell"`)

## 인스턴스 정리 메모
- A100 `minitron-nemo`는 **삭제됨**(Blackwell sm_120 동작 확인 후 불필요).
- 두 Blackwell은 자동 정지되지만, 더 안 쓸 거면 **삭제**해 디스크 과금도 정리 권장.
