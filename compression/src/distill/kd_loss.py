"""Top-k logit-level KD loss (+ CE) for sequence-level distillation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_topk_loss(
    student_logits: torch.Tensor,   # [B, L, V]
    pos: torch.Tensor,              # [B, P] long, label 토큰 인덱스 j (>=1)
    pos_mask: torch.Tensor,         # [B, P], 1 valid / 0 padding
    topk_idx: torch.Tensor,         # [B, P, k] long
    topk_logit: torch.Tensor,       # [B, P, k] float (소프트닝 전 teacher logit)
    hard_labels: torch.Tensor,      # [B, P] long (= input_ids[pos]); padding은 -100 가능
    temperature: float = 2.0,
    alpha: float = 0.9,
) -> dict:
    B, P, k = topk_idx.shape
    V = student_logits.size(-1)
    pred_pos = (pos - 1).clamp(min=0)                       # [B, P] 예측 위치 i=j-1
    idx = pred_pos.unsqueeze(-1).expand(B, P, V)            # [B, P, V]
    pred_logits = student_logits.gather(1, idx)            # [B, P, V]

    # --- KD: top-k 위에서 KL ---
    t_prob = F.softmax(topk_logit / temperature, dim=-1)    # [B, P, k] (top-k 정규화)
    s_logprob_full = F.log_softmax(pred_logits / temperature, dim=-1)  # [B, P, V]
    s_logprob_topk = s_logprob_full.gather(2, topk_idx)     # [B, P, k]
    kl = (t_prob * (torch.log(t_prob + 1e-9) - s_logprob_topk)).sum(-1)  # [B, P]
    mask = pos_mask.float()
    denom = mask.sum().clamp(min=1.0)
    loss_kd = (temperature ** 2) * (kl * mask).sum() / denom

    # --- CE: 정답 토큰 hard target ---
    ce_tok = F.cross_entropy(
        pred_logits.reshape(B * P, V),
        hard_labels.reshape(B * P),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, P)
    loss_ce = (ce_tok * mask).sum() / denom

    loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
    return {"loss": loss, "loss_kd": loss_kd.detach(), "loss_ce": loss_ce.detach()}
