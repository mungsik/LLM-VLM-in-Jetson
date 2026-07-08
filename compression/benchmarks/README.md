# benchmarks/ — 기성 GGUF 모델 양자화 기준점 실험

**프루닝 파이프라인과 별개**의 보조 실험. "14B를 프루닝" 대신 "작은 기성 모델을 양자화"하면
Jetson에서 어떤지 비교하기 위한 기준점(baseline)이다. 대상: `unsloth/Phi-4-mini-instruct-GGUF` (3.8B).

## 스크립트
| 파일 | 용도 |
|---|---|
| `eval_gguf.py` | GGUF 모델을 transformers `gguf_file` 로더로 KMMLU 평가 (문항별 예측 저장) |
| `eval_task.py` | 임의 lm-eval 태스크(HRM8K 등)를 HF/GGUF로 평가 |
| `eval_ppl.py` | transformers 로더로 한국어 PPL (⚠️ 아래 주의 — Phi-4-mini는 부정확) |
| `run_quant_sweep.sh` | 8종 quant KMMLU 일괄 |
| `run_ppl_sweep.sh` | 8종 quant PPL 일괄 (transformers 로더) |
| `run_llamacpp_ppl.sh` | **8종 quant PPL (llama.cpp llama-perplexity — 정확)** |
| `run_ko_ifeval_sweep.sh` | Ko-IFEval/IFEval류 생성형 지시 준수 평가 (llama.cpp server + lm-eval GGUF) |
| `summarize_lm_eval_json.py` | lm-eval 출력 JSON을 `docs/results/*.jsonl` 요약 형식으로 변환 |

## ⚠️ 중요한 교훈 (transformers GGUF 로더의 함정)
transformers 의 `gguf_file` 로더는 **Phi-4-mini 의 LongRoPE 를 잘못 변환**한다
(rope_type을 default/theta=10000으로 설정). 짧은 입력(KMMLU 문제)은 그럭저럭 되지만
**긴 컨텍스트(PPL 측정)에서 망가져** PPL이 수백만으로 폭주한다.
→ **GGUF 의 PPL/긴 컨텍스트 평가는 반드시 llama.cpp(`llama-perplexity`)로 해야 한다.**
   llama.cpp는 Jetson 실제 런타임이기도 해서 대표성도 높다.

## 핵심 결과
- **KMMLU(limit=20):** 양자화 무관 거의 평평(29~30%). 단, 4지선다 logprob + 바닥점수라 양자화 손실을 못 잡는 지표.
- **PPL(llama.cpp, 한국어 wiki):** 진짜 손실이 보임 →
  BF16 12.19 / Q4_K_M 12.75(+5%) / Q3_K_M 14.71(+21%) / Q2_K 63.6(붕괴).
  **Q4_K_M(2.49GB)가 무손실 sweet spot, Q3 이하 비권장.**
- **Ko-IFEval:** 현재 레포 결과에는 아직 없음. `run_ko_ifeval_sweep.sh`가
  `docs/results/ko_ifeval.jsonl`을 만들면 `build_quant_sweep_xlsx.py`가
  엑셀의 `Ko-IFEval strict/loose` 컬럼과 `Ko-IFEval` 시트에 자동 병합한다.

## Ko-IFEval 실행
현재 설치된 기본 `lm-eval`에는 `ifeval`은 있지만 한국어 `ko_ifeval` 태스크가 기본 포함되어
있지 않을 수 있다. 한국어 custom task를 등록한 환경에서는 기본값 그대로 실행한다.

```bash
bash benchmarks/run_ko_ifeval_sweep.sh
```

영문 IFEval로 파이프라인만 검증하려면:

```bash
TASK=ifeval LIMIT=20 bash benchmarks/run_ko_ifeval_sweep.sh
```

상세 결과: [`../../docs/results/`](../../docs/results/)
