"""Ko-IFEval: 소규모 한국어 지시 프롬프트셋 + 러너 (검증기 재사용)."""
from __future__ import annotations

from src.distill.ifeval_verify import (
    verify_ending,
    verify_forbidden,
    verify_json_keys,
    verify_list_count,
)

# 각 항목: prompt(모델 입력), verify(답 검증), gold(오라클 테스트용 정답 예시)
IFEVAL_PROMPTS: list[dict] = [
    {
        "prompt": "사과의 장점을 정확히 3개, 번호 목록으로만 답하세요.",
        "verify": lambda a: verify_list_count(a, 3),
        "gold": "1. 비타민이 풍부하다\n2. 포만감을 준다\n3. 보관이 쉽다",
    },
    {
        "prompt": '이름과 나이를 "name","age" 키를 가진 JSON으로만 답하세요.',
        "verify": lambda a: verify_json_keys(a, ["name", "age"]),
        "gold": '{"name": "홍길동", "age": 30}',
    },
    {
        "prompt": "여행의 좋은 점을 설명하되 '돈'이라는 단어는 절대 쓰지 마세요.",
        "verify": lambda a: verify_forbidden(a, ["돈"]),
        "gold": "여행은 새로운 경험과 견문을 넓혀 줍니다.",
    },
    {
        "prompt": "짧게 답하고 반드시 '끝.'으로 문장을 마치세요.",
        "verify": lambda a: verify_ending(a, "끝."),
        "gold": "오늘 할 일을 마쳤습니다. 끝.",
    },
]


def run_ifeval(generate_fn, prompts: list[dict] = IFEVAL_PROMPTS) -> dict:
    passed = 0
    for p in prompts:
        ans = generate_fn(p["prompt"])
        # 빈 답은 어떤 지시도 통과 못 함(금칙어 검증이 vacuously True 되는 것 방지).
        if ans and ans.strip() and p["verify"](ans):
            passed += 1
    n = len(prompts)
    return {"pass_rate": passed / n if n else 0.0, "n": n, "passed": passed}
