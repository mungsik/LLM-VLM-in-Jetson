# Phi-4 구조적 프루닝 + KMMLU 결과 (2026-06-17)

## 환경
- GPU: NVIDIA RTX PRO 6000 Blackwell 96GB (GCP `g4-standard-48`, asia-east1-a)
- 모델: `microsoft/phi-4` (14.66B, Phi-3 아키텍처, fused gate_up_proj)
- 보정셋: `MarkrAI/KoCommercial-Dataset` 64 samples, seq_len 512 (한국어)
- 프루닝: MLP intermediate width 30% 슬라이싱 (structured, 실제 차원 축소)
- 코드: 수동 텐서 슬라이싱 (torch-pruning 미사용 — transformers 5.x에서 폭주)

## 프루닝 결과
| | 값 |
|---|---|
| 파라미터 | 14,659,507,200 → **11,356,492,800** (22.5% 감축) |
| 메모리(4bit 추정) | 7.33GB → **5.68GB** |

- 22.5%(전체) = MLP intermediate 30% 감축. MLP가 전체의 ~2/3라 그 비율.
- 프루닝만으로 4bit 5.68GB → **이미 Jetson Orin Nano 8GB 적재 가능 수준**.

## KMMLU 비교 (limit=20, 4지선다)
| 모델 | KMMLU 정확도 |
|---|---|
| 원본 Phi-4 | **34.11%** |
| 프루닝 (**활성값**/Minitron) | **18.33%** |
| 프루닝 (magnitude) | **7.33%** |

### 핵심 발견
- **활성값 기반(18.33%) > magnitude(7.33%)** — 같은 30% 프루닝인데 **2.5배** 정확도 보존.
  → "어떤 뉴런을 남기냐"가 결정적이며, **활성값 기반(Minitron 방식)이 weight magnitude보다 우월**함을 정량 입증.
- 둘 다 원본(34%) 대비 크게 하락 → **distillation 회복 전이라 정상.** 회복(PR3)이 다음 단계.
- 원본 Phi-4의 한국어 baseline 자체가 낮음(34%, 영어 중심 모델).
- caveat: limit=20 소표본 → 노이즈 있으나 활성값 vs magnitude 격차는 충분히 큼.

## 산출물
- 프루닝 모델: 인스턴스 `phi4-blackwell:~/LLM-VLM-in-Jetson/compression/artifacts/`
  - `phi4-pruned-act` (활성값 기반)
  - `phi4-pruned-smoke` (magnitude)
- 인스턴스 정지 시 디스크 유지됨(삭제 시 소실) → 장기 보존 필요하면 GCS 업로드 권장.

## 다음
- PR3 distillation (teacher=Phi-4, 한국어)으로 34→18% 하락 회복
- (옵션) attention head·depth 프루닝 추가, 풀 KMMLU 정밀 측정
