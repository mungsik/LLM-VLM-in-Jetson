import torch
from src.common.param_stats import count_parameters
from src.prune.structured_prune import prune_width


def test_prune_width_reduces_params_and_runs(tiny_model, example_inputs):
    before = count_parameters(tiny_model)
    pruned, info = prune_width(tiny_model, example_inputs, ratio=0.25)
    after = count_parameters(pruned)

    # 파라미터가 실제로 감소
    assert after < before
    assert info["params_before"] == before
    assert info["params_after"] == after
    # forward가 여전히 동작하고 vocab 차원 유지
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128
