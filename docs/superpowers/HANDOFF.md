# 진행 현황 / 다른 컴퓨터에서 이어가기 (Handoff)

> 마지막 업데이트: 2026-06-09. 브랜치 `feature/phi4-compression`.

## 한 줄 요약
Phi-4(14B) → Jetson Orin Nano 8GB 경량화 프로젝트. **PR1(Task 0~4) 구현 완료·커밋됨. 다음 시작점 = Task 5 (prune/calibration.py).**

## 문서
- 설계 spec: `docs/superpowers/specs/2026-06-08-phi4-compression-jetson-design.md`
- 구현 plan(Task 0~9, PR1+PR2 범위): `docs/superpowers/plans/2026-06-09-phi4-compression-pr1-pr2.md`
- 실행 방식: superpowers **subagent-driven-development** (Task별 구현 → 스펙 리뷰 → 품질 리뷰).

## 완료 상태 (커밋됨)
| Task | 내용 | 상태 |
|---|---|---|
| 0 | scaffold (deps·configs·dirs) | ✅ `5c24c2d` |
| 1 | tiny GQA Llama 프록시 fixture | ✅ `f16ffbd` |
| 2 | common/param_stats.py | ✅ `1bff946` |
| 3 | common/model_loader.py | ✅ `9e0255f` |
| 4 | common/eval_kmmlu.py (**PR1 끝**) | ✅ `2f9b92b` |
| 5 | prune/calibration.py | ⏭️ **다음 시작점** |
| 6 | prune/importance.py (활성값 중요도) | ⬜ |
| 7 | prune/structured_prune.py (torch-pruning) | ⬜ |
| 8 | prune/report.py | ⬜ |
| 9 | scripts/run_prune.py (PR2 끝) | ⬜ |

현재 단위테스트: **7 passed, 1 deselected**(integration).

## 새 컴퓨터 환경 셋업
```bash
git clone https://github.com/Pseudo-Lab/LLM-VLM-in-Jetson.git
cd LLM-VLM-in-Jetson
git checkout feature/phi4-compression
cd compression
python3 -m venv .venv          # Python 3.10~3.12
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt   # ⚠️ 포그라운드로! (오래 걸림)
.venv/bin/python -m pytest     # 7 passed, 1 deselected 확인
```
테스트는 항상 `compression/` 디렉토리에서 `.venv/bin/python -m pytest` 로 실행.

## ⚠️ 발견된 환경 주의사항 (재발 방지)
- **pip install은 반드시 포그라운드**(또는 백그라운드 후 완료 대기). 백그라운드로 돌리고 turn을 끝내면 설치가 중간에 killed되어 torch가 깨짐(`libtorch_global_deps.dylib` 없음).
- **transformers 5.10.2**: `from_pretrained`는 `dtype=`가 canonical (`torch_dtype=`은 BC용). model_loader.py에 이미 반영됨.
- 설치 검증된 버전: torch 2.12.0 / transformers 5.10.2 / torch-pruning 1.6.1 / lm-eval 0.4.12 / datasets 5.0.0. (이전 컴퓨터는 anaconda python3.11 사용)
- **Task 5 주의(미해결)**: `tokenize_texts`가 `padding="max_length"`를 쓰는데 Llama 토크나이저는 pad_token이 없어 에러남 → 구현 시 pad_token 없으면 eos_token으로 폴백 처리 필요. 테스트는 `hf-internal-testing/llama-tokenizer`(소량 다운로드) 사용.
- **Task 7 주의**: torch-pruning 1.6.1의 attention head 프루닝 인자(`num_heads`/`prune_num_heads`)가 버전에 민감 → 설치된 버전 예제 확인 후 인자명 맞출 것. 핵심 검증 = "파라미터 감소 + forward 정상".

## 다음 할 일
1. 환경 셋업 후 `git log --oneline` 으로 커밋 확인.
2. plan의 **Task 5**부터 subagent-driven-development로 재개 (또는 직접 TDD).
3. PR2(Task 5~9) 완료 후 GPU 서버에서 실제 Phi-4 integration 스모크.
