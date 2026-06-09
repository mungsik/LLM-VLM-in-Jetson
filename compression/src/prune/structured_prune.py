"""torch-pruning 기반 구조적 width 프루닝 (GQA 존중)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch_pruning as tp

from src.common.param_stats import count_parameters

# torch-pruning이 내부적으로 쓰는 prune 함수 핸들 (출력 채널 프루닝 식별용)
from torch_pruning.pruner import function as _tp_function


class _PrecomputedImportance(tp.importance.MagnitudeImportance):
    """활성값 등으로 미리 계산한 '출력 채널' 점수를 torch-pruning에 공급.

    MagnitudeImportance를 상속해 그룹 reduce/normalize 머신러리는 그대로 쓰고,
    점수가 있는 Linear의 출력채널 프루닝에 한해 weight magnitude 대신 precomputed
    점수를 사용한다. 나머지(입력채널·미등록 레이어)는 magnitude로 폴백.
    """

    _OUT_FNS = (
        _tp_function.prune_linear_out_channels,
        _tp_function.prune_conv_out_channels,
    )

    def __init__(self, scores_by_module: dict[nn.Module, torch.Tensor], **kwargs):
        super().__init__(**kwargs)
        self._scores = scores_by_module

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            scores = self._scores.get(layer)
            if (
                scores is not None
                and prune_fn in self._OUT_FNS
                and scores.shape[0] == getattr(layer, "out_features", -1)
            ):
                local_imp = scores.to(layer.weight.device)[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)
            elif isinstance(layer, nn.Linear) and prune_fn in self._OUT_FNS:
                w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)
        if len(group_imp) == 0:
            # 입력채널 등만 있는 그룹 → 부모(magnitude) 로직에 위임
            return super().__call__(group)
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


def _attention_head_map(model: nn.Module) -> dict[nn.Module, int]:
    """attention q/k/v proj 모듈 → head 수 매핑 (GQA 처리용).

    Llama식 분리 q/k/v_proj는 매핑하고, Phi-3/Phi-4식 fused qkv_proj는
    head 단위 매핑 대상이 아니므로 건너뛴다(존재하는 속성만 매핑).
    """
    head_map: dict[nn.Module, int] = {}
    cfg = model.config
    for layer in model.model.layers:
        attn = layer.self_attn
        q, k, v = (getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj"))
        if q is not None:
            head_map[q] = cfg.num_attention_heads
        if k is not None:
            head_map[k] = cfg.num_key_value_heads
        if v is not None:
            head_map[v] = cfg.num_key_value_heads
    return head_map


def prune_width(
    model: nn.Module,
    example_inputs: torch.Tensor,
    ratio: float,
    importance_scores: dict[nn.Module, torch.Tensor] | None = None,
):
    """폭(width) 구조적 프루닝을 수행하고 (pruned_model, info) 반환."""
    params_before = count_parameters(model)

    importance = (
        _PrecomputedImportance(importance_scores)
        if importance_scores
        else tp.importance.MagnitudeImportance(p=2)
    )
    ignored = [model.lm_head, model.model.embed_tokens]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=ignored,
        num_heads=_attention_head_map(model),
        prune_num_heads=True,
        prune_head_dims=False,
        global_pruning=False,
        # HF CausalLM은 ModelOutput(tuple)을 반환 → 의존성 추적용 logits 텐서 추출
        output_transform=lambda out: out.logits,
    )
    pruner.step()

    params_after = count_parameters(model)
    info = {
        "params_before": params_before,
        "params_after": params_after,
        "ratio_actual": 1.0 - params_after / params_before,
        "ratio_target": ratio,
    }
    return model, info
