from src.eval.ifeval_run import IFEVAL_PROMPTS, run_ifeval


def test_prompts_have_prompt_and_verify():
    assert len(IFEVAL_PROMPTS) >= 4
    for p in IFEVAL_PROMPTS:
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert callable(p["verify"])


def test_run_all_pass_with_oracle():
    # oracle generate_fn: 각 프롬프트의 정답을 그대로 돌려주면 pass_rate=1.0
    def oracle(prompt):
        return next(p["gold"] for p in IFEVAL_PROMPTS if p["prompt"] == prompt)

    res = run_ifeval(oracle)
    assert res["pass_rate"] == 1.0
    assert res["n"] == len(IFEVAL_PROMPTS)


def test_run_all_fail_with_empty():
    res = run_ifeval(lambda prompt: "")
    assert res["passed"] == 0
    assert res["pass_rate"] == 0.0
