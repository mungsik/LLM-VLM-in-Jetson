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

## 다음 할 일
1. **GPU 서버에서 실제 Phi-4 프루닝**: `.venv/bin/python scripts/run_prune.py --config configs/prune_phi4.yaml` (integration). 한국어 보정셋(KoCommercial/KoAlpaca-RealQA/kowikitext-qa) 다운로드 + CUDA 필요.
2. 다음 PR: **distillation**(teacher=Phi-4, 한국어 데이터) → **GGUF 양자화**(imatrix, Q4→IQ3/IQ2) → Jetson 벤치.
3. (선택) 브랜치명 정리: `recovery-task0-6` → `feature/phi4-compression` 통합.
4. (housekeeping) 숨김 폴더 `Pseudo-Lab/.old-corrupt-trash.nosync`(옛 깨진 저장소) Finder로 휴지통 삭제.
