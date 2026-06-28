# kd-v1 평가 결과 — full-FT logit-KD (2026-06-28)

## 한 줄 결론
학습은 정상 완료·지식은 보존됐지만, **full-FT logit-KD가 scaleup(1-epoch LoRA) 대비 체감 채팅 품질을 뚜렷이 개선하지는 못했다.** 두 모델이 같은 약점(환각·코드스위칭)을 공유 → 다음 레버는 학습강도가 아니라 **데이터(Approach B)**.

## 학습/지식
- train 2 epoch(3,120 step) 정상 완료, 후반 step-loss ~0.66, 산출물 `artifacts/phi4-pruned-depth-distill-kd-v1`.
- **KMMLU(500 subset) = 41.12%** — 학습 전 masked base 34.92% 대비 향상, 옛 scaleup 42.38%와 동급 → **지식 회귀 없음**(가드 통과).

## 디코딩 함정 (중요)
- **greedy(temp=0, no rep penalty)**: kd-v1·scaleup **둘 다 파국적 반복**으로 붕괴("빛빛빛…", "조선 왕조의 첫 번째 왕…"×40). → 반복은 순수 디코딩 artifact.
- **temp=0.7 + repetition_penalty=1.15**: 양쪽 다 반복 소멸, 유창. → 반복은 서빙단으로 해결됨(사용자 예상대로).

## kd-v1 vs scaleup (sane decoding) 정성 비교
- 계산/추론(p06 평균700원, p11 키 C)·사실vs의견(p07): 둘 다 정확, 동급.
- **사실성**: 양구 명소(p02)·한글 창제(p15) **둘 다 심한 환각** + "모른다" 미준수. 동일 약점.
- **코드스위칭**: kd-v1이 영어 단어 혼입(Mutable/Limited/subjective)·gibberish 꼬리(p03,p14)가 scaleup보다 다소 많음.
- **지시준수**: 5개목록(p03)·표(p14) 양쪽 형식 일탈.

## 해석
KMMLU(지식)는 보존했으나 체감 채팅은 wash(일부 kd-v1 약간 열위). 두 모델이 동일 약점을 공유 → 원인은 **학습강도가 아니라 teacher 데이터**(단일턴 지식QA + Phi-4 자체의 코드스위칭·사실성 한계). 학습방식 레버는 거의 소진.

## 다음 옵션
1. **Approach B (권장)**: teacher 데이터 재생성 — 사실성 grounding, 코드스위칭 억제, 지시 다양화 후 재distill.
2. **서빙단 즉시 적용**: rep_penalty 1.15 + temp 0.7을 비교 UI/운영 기본값으로(반복 즉시 해결).
3. **kd-v1 보류 채택**: 지식QA/RAG 용도면 KMMLU 41% 강점으로 충분.

상세 출력: `2026-06-28-kd-v1-vs-scaleup-probes.md`(greedy), `...-sane-decoding.md`(rep penalty).
