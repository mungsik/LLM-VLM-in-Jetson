from src.common.param_stats import count_parameters
from src.prune.depth_prune import compute_block_influence, prune_depth


def test_block_influence_shape(tiny_model, example_inputs):
    bi = compute_block_influence(tiny_model, [example_inputs])
    assert bi.shape[0] == tiny_model.config.num_hidden_layers  # 2


def test_prune_depth_removes_layers_and_runs(tiny_model, example_inputs):
    bi = compute_block_influence(tiny_model, [example_inputs])
    before = count_parameters(tiny_model)
    pruned, info = prune_depth(tiny_model, ratio=0.5, bi_scores=bi)

    # 레이어가 실제로 줄고(2 -> 1) 파라미터 감소
    assert pruned.config.num_hidden_layers == 1
    assert len(pruned.model.layers) == 1
    assert count_parameters(pruned) < before
    # forward 정상 + vocab 차원 유지
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128
