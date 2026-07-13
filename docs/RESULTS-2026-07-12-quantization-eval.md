# Phi-4 경량화 — 양자화 & 평가 결과 (2026-07-12)

Jetson Orin Nano 8GB 배포를 위한 Phi-4 경량화 파이프라인의 **양자화·평가 단계** 기록.
파이프라인: `Phi-4(14.7B) → ShortGPT 프루닝(9.89B) → 한국어 CPT → 대화 SFT → 양자화 → Jetson`

## 1. 최종 SFT 모델 = v2

- **v2 확정**: 짧은 일상 대화의 자연스러움 우선 (목표 = Jetson 일반 한국어 대화).
- v3(korquad 멀티턴 데이터 추가)는 긴 맥락 유지·환각 억제엔 강했으나, korquad의 문서체("~니까요/~습니다")가 섞여 **짧은 캐주얼 대화가 어색**해져 미채택.
- 멀티턴 붕괴는 상당 부분 **디코딩 문제**였음(freq_penalty 0.4 + presence_penalty 0.3로 해결).

## 2. 양자화 4종 비교 (대상: phi4-sft-v2, 9.89B)

| 방법 | 크기 | PPL | KMMLU | 배포처 |
|---|---|---|---|---|
| f16 (원본) | 19.78 GB | 2.804* | **0.3987** | — |
| **GGUF Q3_K_M (imatrix)** ⭐ | **5.02 GB** | 2.976* | 0.3773 (−2.1%p) | **Jetson** |
| GGUF Q4_K_M (imatrix) | 6.14 GB | 2.838* | 0.3858 (−1.3%p) | Jetson(빡빡) |
| GPTQ W4A16 (llm-compressor) | 6.2 GB | 2.884** | 0.3853 (−1.3%p) | 서버 GPU |
| AWQ W4A16 (llm-compressor) | 6.3 GB | 2.996** | 0.3769 (−2.2%p) | 서버 GPU |

<sub>PPL: *llama.cpp, **vLLM (측정계 다름, 열화%만 비교). KMMLU: lm-eval-harness hf 백엔드 limit50 통일.</sub>

### 핵심 결론
- **최종 배포본 = GGUF Q3_K_M-imat (5.02GB)** — Jetson 8GB용. 지식 손실 −2.1%p, PPL +6.1%.
- **4bit 양자화는 지식(KMMLU)을 거의 안 깎음** (≤2.2%p). PPL 순위 = KMMLU 순위 완전 일치.
- 서버 GPU 배포라면 **GPTQ W4A16(−1.3%p)**가 최고 효율.
- 최종 GGUF Q3 파라미터 실측: **9,888,343,040개(9.89B)**, 텐서 159, 레이어 26. 양자화는 개수 불변(프루닝에서 확정), Q3_K_M 실측 ~4.06 bit/param.

## 3. 함정과 해결 (재현 시 주의)

1. **GGUF 변환 실패** (`tokenizer.model` 없음): 학습 산출물의 `tokenizer_config.json`이 `tokenizer_class="TokenizersBackend"`(원본 Phi-4는 `GPT2Tokenizer`)라 llama.cpp가 SPM 경로로 샘. → **변환용 복사본에서 `tokenizer_class`만 `GPT2Tokenizer`로 수정**.
2. **imatrix 바이너리 없음**: `cmake --build build --target llama-imatrix`로 별도 빌드.
3. **AutoAWQ +41% PPL 비정상**: deprecated AutoAWQ가 torch2.11/transformers5 스택에서 스케일·클리핑 오작동. gptqmodel은 설치 자체 실패(pcre/Blackwell 미성숙). → **llm-compressor(vLLM 공식)로 통일**하니 정상. (Codex GPT-5.6 진단)
4. **GGUF KMMLU 측정 불가** (vLLM 직접 로드 2중 실패): transformers 5.12.1 `is_gguf_available()`가 gguf 0.19.0을 'N/A'로 오인(import명→배포명 역매핑 실패) + vLLM이 hf_config_path 무시. → **lm-eval `hf` 백엔드 + `gguf_file`** + 스크립트 상단 gguf 버전 몽키패치. (Codex 진단)

## 4. 스크립트 (compression/scripts/quant/)

- `llmc_quant.py` — llm-compressor로 GPTQ/AWQ W4A16 양자화 (인자 gptq|awq)
- `kmmlu_gguf_hf.py` — GGUF KMMLU (hf 백엔드 + gguf_file + gguf 버그 패치)
- `kmmlu_one.py` — HF/compressed-tensors 모델 KMMLU
- `vllm_ppl.py` — vLLM completions echo+logprobs 기반 PPL
- `mt_test2.py` — 멀티턴 품질 비교 테스트

## 5. 서빙 비교 UI (docs/tools/)

- `phi4-vs-base.html` — 우리 경량화본(GGUF Q3, llama-server) vs 원본 Phi-4(vLLM) 2열 비교
- `phi4-compare-chat.html` — v2 vs v3 비교 (디코딩 penalty 손잡이)
- GGUF는 vLLM 직접 서빙 불가 → `llama-server -m Q3.gguf --port 8005 -ngl 99 -c 2048 --jinja`

## 6. 다음 단계

- **Jetson Orin Nano 8GB 실측**(물리 기기): `phi4-sft-v2-Q3_K_M-imat.gguf`(5.02GB) 반입 → llama.cpp/ollama 로드 + 속도/메모리 측정.
