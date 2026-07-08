# Phi-4-mini GGUF 양자화 PPL (llama.cpp llama-perplexity)

측정: 2026-06-22, GCP phi4-blackwell(RTX PRO 6000), llama.cpp CUDA(sm_120) 빌드.
코퍼스: 한국어 wikipedia 80문서, ctx=512. 낮을수록 좋음.

| quant | 실효 bpw | 크기 | PPL | BF16 대비 |
|---|---|---|---|---|
| BF16 | 16.02 | 7.68GB | 12.19 | 기준 |
| Q8_0 | 8.52 | 4.08GB | 12.20 | +0.1% |
| Q6_K | 6.58 | 3.16GB | 12.40 | +1.7% |
| Q5_K_M | 5.94 | 2.85GB | 12.51 | +2.6% |
| Q4_K_M | 5.20 | 2.49GB | 12.75 | +4.6% |
| Q3_K_M | 4.42 | 2.12GB | 14.71 | +20.7% |
| Q2_K_L | 3.51 | 1.68GB | 63.61 | +422% |
| Q2_K | 3.51 | 1.68GB | 63.61 | +422% |

**결론:** Q4_K_M(2.49GB)이 무손실 sweet spot. Q3부터 열화, Q2 붕괴.
**교훈:** transformers gguf_file 로더는 Phi-4-mini LongRoPE를 오변환(rope theta=10000)해
긴 컨텍스트 PPL이 폭주(수백만). GGUF 평가는 반드시 llama.cpp로.
