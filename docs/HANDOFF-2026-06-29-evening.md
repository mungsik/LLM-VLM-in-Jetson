# HANDOFF — 2026-06-29 저녁 (다른 컴퓨터 이어작업용)

이 문서는 iCloud 동기화되는 워킹트리에 있으니 다른 컴퓨터에서 그대로 보입니다.
(메모리는 머신별로 갈리므로, 크로스머신 컨텍스트는 이 파일이 전달자입니다.)

> 이전 맥락: kd-v1(full-FT logit-KD)은 지식(KMMLU 41%)은 지켰으나 채팅 품질이 scaleup과 wash였고,
> **데이터가 원인**(단일턴 100%·영어섞임 41.6%·"모른다" 0.5%)임을 파일 분석으로 확인. → **Approach B**로 전환.

---

## 지금 무엇을 하고 있나 (Approach B)

**목표:** depth-pruned Phi-4 student(10.6B)를 **멀티턴 한국어 챗봇**으로 재학습. 용도=계속 주고받는 대화챗봇(사용자 확정).
**방법:** logit-KD 폐기 → 기성 한국어 멀티턴 데이터 정제 + 합성으로 코퍼스를 새로 만들고 **멀티턴 plain-CE SFT**.

설계/계획 문서(읽어볼 것):
- 스펙: `docs/superpowers/specs/2026-06-29-multiturn-chatbot-data-design.md`
- Plan 1(데이터): `docs/superpowers/plans/2026-06-29-multiturn-chatbot-data-pipeline.md`
- Plan 2(학습+평가): `docs/superpowers/plans/2026-06-29-multiturn-chatbot-training-eval.md`

---

## 데이터 결정 (전부 실측 검증함)

- **메인 뼈대:** `lemon-mint/smol-koreantalk` — 멀티턴 75%, 평균 4.6메시지. **단, assistant 답변 44.6%가 영어섞임** → 강한 영어필터로 깨끗한 ~33%(≈11만 멀티턴)만 추출.
- **원어민·지문근거 보강:** `heegyu/korquad-chat-v1` — 9.6k, 자연 한국어·영어없음·지문근거. 형식은 `text` 안 `<sys>/<usr>/<bot>` → 파서로 messages 변환.
- **합성 보강:** "모른다"(answerable 짝+되묻기) + 지시·형식(프로그램 검증 통과분만).
- **제외:** `junelee/sharegpt_deepl_ko`, `maywell/koVast`(둘 다 번역투).
- 코퍼스 가중: 번역(smol):원어민(korquad ×5) 비율 관리 + 모른다(거부) ≤10% 캡.

## 평가 결정

- 한국어 멀티턴 **표준 벤치 부재**(LogicKor 2024-10 archived). → **벤치가 아니라 방법으로**: LLM-심판 페어와이즈(현행 강모델, chat-v1 vs scaleup vs kd-v1) + 자체 held-out 멀티턴 프로브.
- KMMLU 가드(**-2pt 이상 하락 시 실패**) + 영어섞임율·IDK 캘리브레이션 자동지표.
- 디코딩 기본값 `repetition_penalty=1.15, temperature=0.7`(kd-v1에서 반복=디코딩 artifact 확인).

---

## 완료된 것 (코드 전부 iCloud 동기화됨, 로컬 37 테스트 green)

**Plan 1 (데이터 파이프라인) — 9 task 전부 구현·테스트:**
- `compression/src/distill/multiturn_labels.py` — 멀티턴 assistant 라벨 마스킹(엣지케이스: 잘린턴 제외·prefix fail-closed)
- `compression/src/distill/ko_text.py` — 영어산문 비율(연속 영어단어 런; 코드/약어/한글붙은영어 허용)
- `compression/src/distill/korquad_chat.py` — `<sys>/<usr>/<bot>` 파서
- `compression/src/distill/corpus.py` — conversation_is_clean / is_multiturn / weighted_merge
- `compression/src/distill/ifeval_verify.py` — 형식 검증기(목록/JSON/금지어/끝문장, 엄격)
- `compression/scripts/synth_idk.py` — 모른다 합성(거부+answerable짝+되묻기)
- `compression/scripts/synth_instructions.py` — 지시·형식 합성(검증 통과분만, holdout=ending)
- `compression/scripts/score_naturalness.py` — LLM-심판 자연스러움 점수(첫+마지막 턴)
- `compression/scripts/build_chat_corpus.py` — 통합 빌더(영어필터·가중·거부캡·누락경고·frac검증)

**Plan 2 (학습+평가) — CPU 유닛(1·3·4·5) 구현·테스트:**
- `compression/src/distill/sft_data.py` — 멀티턴 SFT 데이터셋(assistant-only CE 라벨) + SFTCollator
- `compression/src/distill/chat_metrics.py` — english_mixing_rate / refused / idk_calibration
- `compression/scripts/judge_pairwise.py` — 페어와이즈 심판(위치편향 스왑, invalid 별도집계)
- `docs/results/probes/chat_multiturn_probes_ko.jsonl` — held-out 멀티턴 프로브 10개

**테스트:** `compression/tests/distill/test_*.py` (multiturn_labels·ko_text·korquad_chat·corpus·ifeval_verify·sft_data·chat_metrics·judge_pairwise) — **37 passed.**

**codex MCP 리뷰 2회(Plan1 1회 + Plan2 2회) → 실버그 다수 수정·재검증.** (예: 잘린턴 부분라벨·Title-case 영어 필터구멍·심판 형식오류 tie묻힘·refused 오탐. **CE shift 규약은 정확함 확인.**)

---

## 아직 안 한 것 (다음 세션)

1. **git 커밋 — 전부 미커밋.** 이 Mac은 gitdir 깨짐(`/Users/mungsik/.gitdirs/phi4-jetson` 부재). **커밋은 VM에서만 가능.** branch `recovery-task0-6`. 광범위 `git add -A` 금지(미커밋 작업 섞임) — 위 파일들만 명시 add.
2. **Plan 2 GPU 본런(미구현/미실행):**
   - `compression/scripts/train_sft_multiturn.py` — Plan 2 Task 2에 완성코드 있음, 파일만 생성하면 됨(GPU 본런).
   - 전량 코퍼스 생성(`build_chat_corpus.py --smol-limit 80000`) → 자연스러움 게이트 → 파일럿 SFT → 2-epoch 본런 → KMMLU 가드 → 페어와이즈 평가. (Plan 2 Task 6 런북)

---

## 다른 컴퓨터에서 이어받는 법

### A) 코드 보기/수정/CPU 테스트 (GPU 불필요)
코드는 iCloud로 다 동기화됨. CPU 테스트(37개)를 돌리려면 venv 하나 만들면 됨:
```bash
python3 -m venv /tmp/ds_venv
/tmp/ds_venv/bin/pip install -q datasets torch safetensors transformers pytest
cd <repo>/compression
/tmp/ds_venv/bin/python -m pytest tests/distill/ -q   # 37 passed 나와야 정상
```
(주의: conftest.py 가 torch/transformers import 하므로 그 둘도 설치 필요. 새 코드 유닛 자체는 stdlib+torch만.)
데이터 생성도 CPU로 가능: `synth_idk.py`, `synth_instructions.py`, `build_chat_corpus.py`(smol/korquad는 HF 스트리밍/다운로드).

### B) GPU 학습/평가/커밋 (VM 필요)
VM: GCP `phi4-blackwell`(project `polarpulse-dev-mungsik`, zone `asia-east1-a`, RTX PRO 6000 96GB).
```bash
P=polarpulse-dev-mungsik; Z=asia-east1-a; I=phi4-blackwell
gcloud compute instances start $I --zone=$Z --project=$P
gcloud compute ssh $I --zone=$Z --project=$P --tunnel-through-iap
```
- repo `/home/mungsik/LLM-VLM-in-Jetson` 는 **mungsik 소유 700 → 모든 명령 `sudo -u mungsik`**.
- venv: 학습 `.venv`(py3.10, torch2.12+cu130), 서빙 `.venv-vllm`. `data/`·`artifacts/` gitignore(VM 전용).
- **이 Mac의 새 코드를 VM에 반영하려면 scp 동기화 후 `sudo -u mungsik` 로 커밋**(이 세션도 그렇게 함: tar→scp /tmp→`sudo -u mungsik tar -x`).
- git identity 이미 설정됨(user.name=mungsik, user.email=mungsik@polarpulse.ai).
- **주의:** 디스크가 한 번 100% 찼었음(현재 여유 있음). 본런 전 `df -h /home` 확인. disposable 아티팩트 정리법은 이전 세션 참고.
- **종료 시 인스턴스 stop**(삭제 아님). 자동종료 watcher 패턴: `~/kd_autostop.sh` 참고(이전 KD 런에서 사용).

### 산출물 위치
- 학습 결과 모델: `artifacts/phi4-pruned-depth-chat-v1`(예정)
- 비교 대상: `artifacts/phi4-pruned-depth-distill-merged-scaleup`(scaleup), `artifacts/phi4-pruned-depth-distill-kd-v1`(kd-v1, 이전 세션 산출)
- student 베이스: `artifacts/phi4-pruned-depth-masked`

---

## 한 줄 다음 액션
다음 GPU 세션: **VM start → 이 Mac 새 코드 scp+commit(명시 파일만) → train_sft_multiturn.py 생성 → 전량 코퍼스 빌드 → 파일럿→본런 SFT → KMMLU 가드 + 페어와이즈 평가.** 상세는 Plan 2 Task 6 런북.
