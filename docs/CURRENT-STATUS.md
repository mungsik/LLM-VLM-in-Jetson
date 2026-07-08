# CURRENT STATUS — 다른 컴퓨터용 진입점 (최종 업데이트 2026-07-08)

다른 컴퓨터에서 이 파일 하나만 보면 전체 현황 + 다음 할 일을 알 수 있게 정리.
(repo는 iCloud 동기화. 라이브 인스턴스/결과는 어느 컴퓨터서나 `gcloud`로 접근 — 프로젝트 `polarpulse-dev-mungsik`.)

---

## 🟢 최신 상태 (2026-07-08 — **SFT v2 완료 + 비교평가 끝, VM 정지·과금 0**)

### ✅ SFT v2 재학습 완료 (한국어 자연스러움 개선)

**왜 했나:** v1(phi4-sft)의 한국어가 어색(시제오류·말투혼용). 원인 = SFT 데이터의 번역투(`smol-koreantalk`) + 언더트레이닝(0.42에폭). CPT 베이스(phi4-cpt)는 네이티브 KoWiki라 깨끗 → **SFT만 네이티브로 다시 얹음**.

| 항목 | v1 (phi4-sft) | **v2 (phi4-sft-v2)** |
|---|---|---|
| 데이터 | smol(번역투)+RealQA+KoCommercial 57k | **smol 제거**, RealQA×2 + KoCommercial(20k캡) = **55,503대화, 네이티브 2:1** |
| 학습량 | 1500스텝(≈0.42에폭) | **6,792스텝(2에폭)**, 4h39m |
| LoRA | r=32 | **r=64** (trainable 144.8M=1.44%) |
| 결과 loss | — | **train 0.813 / eval 0.823** (과적합 없음) |
| 베이스 | phi4-cpt | 동일 |

- 산출물: `artifacts/phi4-sft-v2` (LoRA 병합된 풀모델 19.7GB). 데이터: `artifacts/sft_data.jsonl`(55,503). 구데이터 백업 `sft_data.smol.jsonl.bak`.
- 재현: `scripts/train_sft_multiturn.py --student artifacts/phi4-cpt --corpus artifacts/sft_data.jsonl --out artifacts/phi4-sft-v2 --lora --lora-r 64 --epochs 2`. sft_data.yaml에 `max_rows_per_source: 20000` 캡 추가됨.
- 자동정지 스크립트 `/tmp/autostop.sh` 사용했었음(완료 감지→shutdown). 결과 기록 `~/SFT_V2_RESULT.txt`.

### 🔎 base vs v2 side-by-side 평가 (역할극 멀티턴 대화)

- **좋아진 점:** v1 대비 **한국어 자연스러움 확실히 향상**. 번역투·시제오류·말투혼용 크게 줄음. 역할(마을 지킴이) 유지도 됨.
- **남은 문제 2종:**
  1. **반복 퇴화** — "멍식이님과 멍식이님의…" 이름/대명사 반복, 순환 문장. 원인 = ①디코딩 `repetition_penalty=1.0`(프루닝모델이 1.1↑서 환각이라 낮게둠)이 반복 억제 못함 ②KoCommercial 데이터가 장황·반복적(README도 "downstream 품질필터 없음" 명시).
  2. **멀티턴 붕괴** — 논리모순 지적 발화에서 "무슨 일이 있었나요?…무슨 일이 있었나요?" 반복. 원인 = SFT에 **멀티턴 데이터 0개(단일턴만)** → 멀티턴 OOD + 프루닝 용량손실.

### ⏭ 다음 세션 개선안 (아직 미적용)

- **A. 디코딩 (재학습 불필요, 지금 서버로 바로):** `repetition_penalty` 대신 **`frequency_penalty 0.3~0.5` + `presence_penalty 0.3`**(OpenAI식, 프루닝모델서 환각 없이 반복만 억제). temp 0.5. → UI(`phi4-compare-chat.html`)는 현재 `repetition_penalty`만 보냄(line~145 fetch body) → 슬라이더+요청필드 추가 필요.
- **B. 시스템 프롬프트 (무료):** UI에 시스템 프롬프트 입력창 추가. `"너는 송구봉, 가짜연구소 지킴이. 간결하게, 같은 말 반복 말고 답하라"` → 역할일관성+장황함 개선.
- **C. v3 재학습 (근본):** ①build_sft_data에 응답 품질게이트(n-gram 반복률·길이 필터) ②네이티브 멀티턴 `heegyu/korquad-chat-v1` 주입(`<sys>/<usr>/<bot>` 파서 어댑터 신규 필요).
- **D. Q3 양자화** — `phi4-sft-v2` 대상, `uv pip install sentencepiece` 먼저 → Jetson 배포.

### 🧹 정리 이력 (2026-07-08)
- `artifacts/phi4-pruned-depth-masked`(28층/ratio0.30, legacy 미사용) **삭제** → 디스크 20G 회수(현재 여유 84G/291G, 72%).
- 남은 프루닝 베이스 = `prune_sweep/ratio_0.35`(26층/9.89B, **CPT의 실제 뿌리** — 유지).
- v1 `phi4-sft`(19G)는 아직 보존(v2 최종확정 시 정리 후보). `sft_data.smol.jsonl.bak`(122M)도 잔존.

### 서빙 비교 방법 (다시 할 때)
vLLM 2개 순차 기동(동시 기동 시 KV OOM): 8001 base(`microsoft/phi-4`, util 0.38), 8002 `artifacts/phi4-sft-v2`(util 0.42, `--chat-template artifacts/phi4-sft-v2/chat_template.jinja`). SSH `-L 8001/8002` 터널 후 `docs/tools/phi4-compare-chat.html`(이미 8002=phi4-sft-v2로 맞춰둠). **디코딩: rep_pen 1.0 유지, temp 0.3~0.5.**

### 📈 KMMLU 정량 측정 (2026-07-08 오후, 전부 limit 50 동일 하니스 `eval_kmmlu.py`)

| 단계 | 모델 | 크기 | KMMLU |
|---|---|---:|---:|
| 원본 | Phi-4 | 14.7B | **0.356** |
| ① 프루닝만 | ratio_0.35 | 9.89B | 0.351 |
| ② CPT | phi4-cpt | 9.89B | 0.387 |
| ③ SFT v1 | phi4-sft | 9.89B | 0.396 |
| ③ SFT v2 | phi4-sft-v2 | 9.89B | **0.399** |

- **프루닝 거의 무손실**(0.356→0.351) · **CPT가 원본 초월**(→0.387, KMMLU=한국어 벤치라 한국어 주입 효과) · SFT 소폭↑.
- **v2(0.399) > 원본(0.356)** — 33% 작은데 한국어 지식 오히려↑. **v1≈v2(0.396 vs 0.399)** → KMMLU론 v1/v2 구분 안 됨(차이는 자연스러움/멀티턴).
- 주의: limit 50=부분측정(노이즈 有). distill-scaleup의 42.38%는 **full+10.6B(덜 자름)**라 직접 비교 불가. 원본 34.11%(distill문서 full)와 0.356(limit50)은 하니스차이지만 방향 일치(≈34~36%); 40.75는 또 다른 하니스.

### 🎛 디코딩 발견 (2026-07-08) — 반복 ≠ 멀티턴 (별개 문제)
- **반복 붕괴**("멍식이님 x30") = **디코딩 문제**. **freq_penalty 0.4 + presence_penalty 0.3**(rep_pen 1.0 유지)로 **해결 검증됨**. (프루닝모델은 분포가 뾰족→rep_pen 1.1↑ 곱셈이 top토큰 뒤집어 환각. freq/pres는 가산식이라 반복만 억제·환각X. 셋 다 vLLM 지원, freq/pres는 OpenAI표준.) ※UI는 원복해둠(rep_pen만) — 재현하려면 freq/pres 다시 추가.
- **멀티턴 붕괴**(역할·이름 못지킴) = **데이터 문제**(v2 SFT 단일턴만, 멀티턴 0개) → 디코딩으로 안 잡힘. **v3에서 네이티브 멀티턴(`heegyu/korquad-chat-v1`) 주입 필요**.

### ⏱ 단계별 학습시간 (단일 GPU 실측)
프루닝: ratio당 수분(스윕 전체 ~1h) · **CPT: 8h9m**(full-FT 840스텝/110M토큰) · **SFT v1: ~1.5h**(1500스텝) · **SFT v2: 4h39m**(2에폭 6792스텝 r64) · 양자화: 미완(sentencepiece). 총 ≈15h+.

> 아래 07-06 블록 = v1 파이프라인 완료 기록(프루닝·CPT 상세, 그대로 유효).

---

## 🟢 지금 상태 (2026-07-06 — GPU 인스턴스 정지, 과금 0)

### 방향: Phi-4 → **ShortGPT 프루닝 → 한국어 CPT → 대화 SFT → Q3 양자화 → Jetson**
목표: Phi-4를 경량화해 **Jetson Orin Nano 8GB**에서 자연스러운 한국어 대화. niceinfo(고객사)와 무관.

### ✅ 완료: 프루닝 + CPT + SFT (모델 전부 VM 디스크에 저장됨)

| 단계 | 방법 / 데이터 | 결과 |
|---|---|---|
| ① ShortGPT 프루닝 | phi-4 40층 BI측정(보정: KoCommercial+RealQA) | 40→26층, **9.89B** (ratio 0.35) |
| ② CPT (full-FT, ~8h) | KoWiki(wikipedia 20231101.ko 40k) + 영어replay15%(wikitext), **110M토큰** | **한국어 PPL 108→4.45**, 영어 164→19, KMMLU 0.351→0.387 |
| ③ SFT (LoRA, ~1.5h) | smol-koreantalk 13.6k + KoAlpaca-RealQA 14.4k×2 + KoCommercial 14.9k = **57k대화** | 대화 능력(멀티턴은 약함) |

**핵심 성과:** ShortGPT로 반토막 낸 모델을 CPT가 한국어 PPL **24배** 살림 + 영어 유지. 상세 = `docs/superpowers/RESULTS-2026-07-06-phi4-korean-cpt-pipeline.md`.

### ⏳ 남은 할 일
1. **Q3 양자화** (미완) — chain이 `sentencepiece` 없어서 실패했음. **`compression/.venv-vllm`이나 학습venv에 `uv pip install sentencepiece`** 후:
   ```
   cd compression && uv run python ~/llama.cpp/convert_hf_to_gguf.py artifacts/phi4-sft --outfile /tmp/f16.gguf --outtype f16
   ~/llama.cpp/build/bin/llama-quantize /tmp/f16.gguf artifacts/phi4-sft-Q3_K_M.gguf Q3_K_M
   ```
   → `phi4-sft-Q3_K_M.gguf` (~4.8GB, Jetson 배포용)
2. **(선택) SFT 풀 에폭 재학습** — 멀티턴 품질 개선하려면. 지금 SFT는 1500스텝 캡(≈1에폭 42%)이라 가벼움.
3. **Jetson 실측** — GGUF를 실제 Orin Nano 8GB에 올려 fit·tok/s (물리 기기 필요, 원격 불가).

### 서빙 비교 (base vs 우리 모델)
`docs/tools/phi4-compare-chat.html` = 원본 Phi-4(14.7B) vs phi4-sft 2열 한국어 비교. vLLM 2개(8001 base, 8002 sft) 띄우고 SSH 터널 후 브라우저.
- **디코딩 주의:** 프루닝 모델은 **rep_pen 1.0**(1.1 이상 금지), temp 0~0.3. rep_pen 1.15면 "알파인소프트" 같은 환각 남 → 모델 문제 아니라 디코딩 설정.

---

## 🖥️ VM 접근 (어느 컴퓨터서나)

```bash
P=polarpulse-dev-mungsik; Z=asia-east1-a
gcloud compute instances start phi4-blackwell --zone=$Z --project=$P
gcloud compute ssh phi4-blackwell --zone=$Z --project=$P --tunnel-through-iap
# 끝나면 반드시 정지 (SA에 compute 스코프 없어 self-stop 불가):
gcloud compute instances stop phi4-blackwell --zone=$Z --project=$P
```
- repo: VM `/home/mungsik/LLM-VLM-in-Jetson` (소유 mungsik → `sudo -u mungsik`). 브랜치 **`feat/prune-sweep-cpt`**.
- 학습 venv `compression/.venv`, 서빙 `compression/.venv-vllm`(vllm 0.23). uv 사용. **`uv sync` 금지**(vllm prune) — 추가 dep은 `uv pip install`.
- HF 토큰: VM `~/.cache/huggingface/token`에 저장됨(gated RealQA 접근). 로컬 `source-code/.env` HUGGINGFACE_API_KEY.
- 산출물 `compression/artifacts/`: `prune_sweep/ratio_0.35`, `phi4-cpt`, `phi4-sft`(최종), `sft_data.jsonl`, `cpt_data`.

## 📜 문서
- `docs/superpowers/specs/2026-07-05-phi4-korean-chat-jetson-design.md` — 설계
- `docs/superpowers/plans/2026-07-05-*.md` — 5개 플랜(CPT데이터/SFT데이터/평가/프루닝+CPT/양자화)
- `docs/superpowers/RESULTS-2026-07-06-phi4-korean-cpt-pipeline.md` — **실행 결과 상세**

## 🧩 코드 (VM 브랜치 `feat/prune-sweep-cpt`, 커밋됨)
- `compression/src/data/` — CPT 데이터(packing/replay/cpt_corpus)
- `compression/src/distill/sft_sources.py` — SFT 어댑터
- `compression/src/eval/` — 평가(ppl/judge/ifeval/report)
- `compression/src/prune/fit.py` — 8GB fit 판정
- `compression/scripts/` — build_cpt_data / build_sft_data / sweep_prune / train_cpt / train_sft_multiturn / eval_ppl 등

## ⚠️ 메모
- **Pseudo-Lab PR #3**(`shortgpt/` 폴더): ShortGPT 프루닝 코드만 공개 제출됨(리뷰반영 폴더명 변경 완료). CPT/SFT는 미포함.
- 프루닝 스윕 실측: 0.25=11.25B/0.360, 0.35=9.89B/0.351, 0.40=9.21B/0.296, 0.45=8.53B/0.288 (품질절벽 0.35↔0.40, Q4로는 다 8GB 초과→Q3).
- 옛 트랙(NeMo/Minitron/distill chat-v1/kd)은 폐기·산출물 일부 삭제.
