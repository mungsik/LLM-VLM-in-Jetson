# 진행 현황 / 다른 컴퓨터에서 이어가기 (Handoff)

> 마지막 업데이트: 2026-06-16. 현재 브랜치 `recovery-task0-6` (fork에 push됨).

## 한 줄 요약
Phi-4(14B) → Jetson Orin Nano 8GB 경량화. **PR1+PR2(Task 0~9) 구현 완료·커밋·push됨. 단위테스트 12 passed.** 다음 = 실제 Phi-4 프루닝(GPU 서버 integration) → distillation/양자화 PR.

## 문서
- 설계 spec: `docs/superpowers/specs/2026-06-08-phi4-compression-jetson-design.md`
- 구현 plan: `docs/superpowers/plans/2026-06-09-phi4-compression-pr1-pr2.md`

## 완료 상태 (전부 커밋·push)
| Task | 내용 | 상태 |
|---|---|---|
| 0~4 | PR1: scaffold·param_stats·model_loader·eval_kmmlu(KMMLU) | ✅ |
| 5 | prune/calibration.py | ✅ |
| 6 | prune/importance.py (활성값 중요도) | ✅ |
| 7 | prune/structured_prune.py | ✅ **수동 MLP 슬라이싱으로 재작성** (아래 주의) |
| 8 | prune/report.py | ✅ |
| 9 | scripts/run_prune.py + run_helpers.py | ✅ |

단위테스트: `compression/`에서 `.venv/bin/python -m pytest` → **12 passed, 1 deselected(integration)**.

## ⚠️⚠️ 환경 셋업 — iCloud 사고 방지 (필독)
이 프로젝트는 `~/Desktop/PolarPulse`(iCloud 동기화 폴더) 안에 있다. **git 저장소를 iCloud로 두 기기에서 공유하면 `.git`이 충돌본으로 깨진다**(실제로 한 번 깨졌음). 그래서 이 맥에선:
- **`.git`을 동기화 폴더 밖으로 분리**: 실제 git 디렉토리는 `~/.gitdirs/phi4-jetson` 에 있고, 작업폴더의 `.git`은 그곳을 가리키는 **포인터 파일**(`gitdir: ...`)이다. → iCloud가 `.git`을 안 건드려서 안 깨짐 + VSCode도 정상 인식.
- **venv도 재생성 필요**(다른 기기 venv는 경로가 달라 동작 안 함).

**새 컴퓨터에서 이어갈 때 (권장 = 동기화 밖에 clone):**
```bash
git clone git@github.com:mungsik/LLM-VLM-in-Jetson.git   # 또는 https
cd LLM-VLM-in-Jetson && git checkout recovery-task0-6
cd compression && python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest   # 12 passed 확인
```
※ iCloud 폴더 안에서 작업하려면 위처럼 `.git`을 `~/.gitdirs`로 분리하고, 절대 두 기기서 동시 작업 금지.

## 기술 주의사항
- **transformers 5.10.2**: `from_pretrained`는 `dtype=`(torch_dtype 아님). model_loader.py 반영됨.
- **Task 7 = 수동 텐서 슬라이싱(structured)**. torch-pruning auto-trace는 transformers 5.x에서 폭주(무한루프)해서 **안 씀**. MLP intermediate를 활성값 중요도 순으로 직접 슬라이싱(차원 실제 축소 → GGUF 메모리 직결). Llama(분리 gate/up)·Phi-3/4(fused gate_up_proj) 둘 다 지원. head/depth 프루닝은 향후 확장.
- 설치 버전: torch 2.12 / transformers 5.10.2 / torch-pruning 1.6.x / lm-eval 0.4.12 / datasets 5.0.0.

## 진행 현황 (2026-06-17) — 프루닝 방법 비교 완료

**실제 Phi-4 프루닝을 GCP GPU에서 돌려 KMMLU 비교까지 완료.** 상세: `docs/results/2026-06-17-pruning-kmmlu.md`
- 추가 코드(전부 push됨): `prune/depth_prune.py`(ShortGPT), `scripts/run_depth_prune.py`, `scripts/eval_compare.py`, `eval_kmmlu` batch_size, fused gate_up_proj 활성값 수정, CPU 슬라이싱(OOM 수정). 단위테스트 **15 passed**.
- KMMLU(limit=20): 원본 34.1% / **depth-ShortGPT 31.9%(최고)** / width-활성값 18.3% / width-magnitude 7.3%.
- **SliceGPT 미완**: 공식 lib로 Phi-4 로딩까지 성공(phi3_adapter 패치 필요)했으나 슬라이싱 단계서 조용히 크래시 → 내일 디버깅.

### GCP 인스턴스 (현재 둘 다 STOPPED)
- **`phi4-blackwell`** (asia-east1-a, g4-standard-48, **RTX PRO 6000 96GB**) ← 메인. 재시작: `gcloud compute instances start phi4-blackwell --zone=asia-east1-a`. SSH: `gcloud compute ssh phi4-blackwell --zone=asia-east1-a --tunnel-through-iap` (포트22 막히면 IAP 필수).
  - 재시작 후 **NVIDIA 드라이버 모듈 재설치 필요**(커널 업뎃 시): `sudo apt-get install -y linux-modules-nvidia-580-server-open-$(uname -r) && sudo modprobe nvidia`.
  - 레포: `~/LLM-VLM-in-Jetson`(메인 venv `compression/.venv`) / `~/TransformerCompression`(SliceGPT, `.venv-slice`, transformers 4.41+torch cu130).
  - 산출물: `~/LLM-VLM-in-Jetson/compression/artifacts/{phi4-pruned-act,phi4-pruned-smoke,phi4-pruned-depth}` (정지해도 디스크 보존, 삭제하면 소실 → 필요시 GCS).
- `niceinfo-poc-seoul`(Seoul, A100×2 40GB) — 안 씀, STOPPED.
- ⚠️ 작업 끝나면 **반드시 instance stop** (g4 ~$3-4/hr).

### SliceGPT 재개 메모
- `~/TransformerCompression/src/slicegpt/adapters/phi3_adapter.py` line 240,260: `"microsoft/phi-4"` 허용 패치 적용됨(이 패치 안 하면 Phi-4 인식 못함).
- 크래시: 슬라이싱(PCA 회전) 단계서 에러·OOM 없이 죽음. 디버깅: (a) Phi-3-mini로 동작 확인 (b) `--device cpu`로 회전만 (c) torch eigh 등 격리.

## 남은 로드맵
1. SliceGPT 크래시 해결 (또는 Phi-3-mini 데이터포인트)
2. **distillation**(teacher=Phi-4, 한국어) — width 하락 회복 (depth는 거의 불필요)
3. **GGUF 양자화**(imatrix, Q4→IQ3/IQ2) → Jetson Orin Nano 8GB 벤치
4. (선택) 브랜치명 `recovery-task0-6` → `feature/phi4-compression` 통합
5. (housekeeping) `Pseudo-Lab/.old-corrupt-trash.nosync` 삭제 — 이미 삭제 확인됨
