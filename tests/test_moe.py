"""Tests for MoEFFN: vectorized dispatch correctness + aux-loss-free balancing."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from open_mythos.main import MoEFFN, MythosConfig


def _cfg(**overrides) -> MythosConfig:
    base = dict(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=32,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
    )
    base.update(overrides)
    return MythosConfig(**base)


def _reference_dispatch(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """The original O(topk * n_experts) nested-loop dispatch, as a reference."""
    B, T, D = x.shape
    flat = x.view(B * T, D)
    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
    ts = scores.gather(-1, topk_idx)
    ts = ts / ts.sum(-1, keepdim=True)
    out = torch.zeros_like(flat)
    for i in range(moe.topk):
        eids = topk_idx[:, i]
        w = ts[:, i].unsqueeze(-1)
        for eid in range(moe.n_experts):
            mask = eids == eid
            if mask.any():
                out[mask] += w[mask] * moe.routed_experts[eid](flat[mask])
    for shared in moe.shared_experts:
        out = out + shared(flat)
    return out.view(B, T, D)


def test_vectorized_dispatch_matches_reference():
    torch.manual_seed(0)
    moe = MoEFFN(_cfg()).eval()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        got = moe(x)
        ref = _reference_dispatch(moe, x)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item()


def test_gradients_flow_and_finite():
    torch.manual_seed(0)
    moe = MoEFFN(_cfg())
    x = torch.randn(2, 5, 64, requires_grad=True)
    moe(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_topk_weights_are_normalized():
    torch.manual_seed(0)
    moe = MoEFFN(_cfg())
    flat = torch.randn(10, 64)
    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
    w = scores.gather(-1, idx)
    w = w / w.sum(-1, keepdim=True)
    assert torch.allclose(w.sum(-1), torch.ones(10), atol=1e-6)


def _load_cv(moe: MoEFFN, x: torch.Tensor) -> float:
    with torch.no_grad():
        flat = x.reshape(-1, x.shape[-1])
        logits = moe.router(flat)
        _, idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=moe.n_experts).float()
        load = counts / counts.sum()
    return (load.std() / load.mean()).item()


def test_bias_update_reduces_imbalance():
    torch.manual_seed(1)
    moe = MoEFFN(_cfg(n_experts=16, moe_bias_update_speed=1e-2))
    # Collapse the router so a couple of experts dominate initially.
    with torch.no_grad():
        moe.router.weight.mul_(0.1)
        moe.router.weight[0] += 3.0
        moe.router.weight[1] += 2.5
    probe = torch.randn(8, 32, 64)
    cv_before = _load_cv(moe, probe)
    moe.train()
    for _ in range(300):
        moe(torch.randn(8, 32, 64))
    cv_after = _load_cv(moe, probe)
    assert cv_after < cv_before  # load became more uniform


def test_update_disabled_and_eval_freeze_bias():
    torch.manual_seed(0)
    # speed=0 disables the update entirely
    moe = MoEFFN(_cfg(moe_bias_update_speed=0.0)).train()
    before = moe.router_bias.clone()
    moe(torch.randn(4, 8, 64))
    assert torch.equal(before, moe.router_bias)

    # eval() also freezes it even when speed > 0
    moe2 = MoEFFN(_cfg(moe_bias_update_speed=1e-2)).eval()
    before2 = moe2.router_bias.clone()
    moe2(torch.randn(4, 8, 64))
    assert torch.equal(before2, moe2.router_bias)
