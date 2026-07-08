import torch
from src.distill.kd_loss import kd_topk_loss


def _toy():
    # B=1, L=3, V=5, P=1, k=2. label 위치 j=2 → 예측 위치 i=1.
    student_logits = torch.zeros(1, 3, 5)
    pos = torch.tensor([[2]])
    pos_mask = torch.tensor([[1.0]])
    topk_idx = torch.tensor([[[3, 1]]])        # teacher top-k 토큰 id
    topk_logit = torch.tensor([[[2.0, 0.0]]])  # teacher가 3을 선호
    hard_labels = torch.tensor([[3]])
    return student_logits, pos, pos_mask, topk_idx, topk_logit, hard_labels


def test_returns_finite_scalar_with_grad():
    s, pos, m, ti, tl, hl = _toy()
    s.requires_grad_(True)
    out = kd_topk_loss(s, pos, m, ti, tl, hl, temperature=2.0, alpha=0.9)
    assert set(out) == {"loss", "loss_kd", "loss_ce"}
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_perfect_match_gives_near_zero_kd():
    # student이 teacher top-k 분포를 그대로 재현하면 KD ≈ 0
    s, pos, m, ti, tl, hl = _toy()
    # 예측 위치 i=1 의 logit을 teacher와 일치시킴 (idx 3→2.0, idx 1→0.0, 나머지 매우 작게)
    s[0, 1, :] = -30.0
    s[0, 1, 3] = 2.0
    s[0, 1, 1] = 0.0
    out = kd_topk_loss(s, pos, m, ti, tl, hl, temperature=2.0, alpha=1.0)
    assert out["loss_kd"].item() < 1e-3


def test_padding_position_is_ignored():
    s, pos, m, ti, tl, hl = _toy()
    # pos를 2개로 늘리고 두 번째는 padding(mask=0) — 결과가 P=1 단독과 동일해야
    pos2 = torch.tensor([[2, 0]])
    m2 = torch.tensor([[1.0, 0.0]])
    ti2 = torch.tensor([[[3, 1], [0, 0]]])
    tl2 = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    hl2 = torch.tensor([[3, -100]])
    a = kd_topk_loss(s, pos, m, ti, tl, hl)["loss"]
    b = kd_topk_loss(s, pos2, m2, ti2, tl2, hl2)["loss"]
    assert torch.allclose(a, b, atol=1e-5)
