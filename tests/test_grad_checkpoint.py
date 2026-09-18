"""Gradient checkpointing on the recurrent loop: exact-equivalence + memory."""
from __future__ import annotations

import copy

import torch

from open_mythos.main import MythosConfig, RecurrentBlock, precompute_rope_freqs


def _cfg(**overrides) -> MythosConfig:
    base = dict(
        vocab_size=100,
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=4,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        act_threshold=0.99,
        dropout=0.0,  # keep the two runs deterministic
    )
    base.update(overrides)
    return MythosConfig(**base)


def _run(block: RecurrentBlock, h, e, freqs, n_loops):
    block.train()
    out = block(h.clone(), e.clone(), freqs, None, n_loops, None)
    out.sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in block.named_parameters()}
    return out.detach(), grads


def test_checkpointing_matches_non_checkpointed():
    torch.manual_seed(0)
    block = RecurrentBlock(_cfg())
    T, n_loops = 6, 4
    freqs = precompute_rope_freqs(block.cfg.dim // block.cfg.n_heads, 16)[:T]
    h = torch.randn(2, T, block.cfg.dim)
    e = torch.randn(2, T, block.cfg.dim)

    # Two identical blocks; one with checkpointing on.
    block_ck = copy.deepcopy(block)
    block.cfg.grad_checkpoint = False
    block_ck.cfg.grad_checkpoint = True

    out, grads = _run(block, h, e, freqs, n_loops)
    out_ck, grads_ck = _run(block_ck, h, e, freqs, n_loops)

    assert torch.allclose(out, out_ck, atol=1e-6), (out - out_ck).abs().max()
    for name in grads:
        g, gck = grads[name], grads_ck[name]
        assert torch.allclose(g, gck, atol=1e-5), (name, (g - gck).abs().max())


def test_checkpointing_reduces_activation_memory():
    # CUDA-only measurement; skip cleanly on CPU-only CI.
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    T, n_loops = 32, 8
    dev = "cuda"

    def peak(grad_ckpt: bool) -> int:
        block = RecurrentBlock(_cfg(dim=128, max_loop_iters=n_loops)).to(dev).train()
        block.cfg.grad_checkpoint = grad_ckpt
        freqs = precompute_rope_freqs(block.cfg.dim // block.cfg.n_heads, 64)[:T].to(dev)
        h = torch.randn(4, T, block.cfg.dim, device=dev)
        e = torch.randn(4, T, block.cfg.dim, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
        block(h, e, freqs, None, n_loops, None).sum().backward()
        return torch.cuda.max_memory_allocated(dev)

    assert peak(True) < peak(False)


def test_no_checkpoint_path_in_eval(monkeypatch):
    # eval() must never checkpoint even if the flag is set (no_grad generation).
    import open_mythos.main as mm

    called = {"n": 0}
    real = mm.checkpoint

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mm, "checkpoint", spy)
    block = RecurrentBlock(_cfg(grad_checkpoint=True)).eval()
    T = 4
    freqs = precompute_rope_freqs(block.cfg.dim // block.cfg.n_heads, 16)[:T]
    h = torch.randn(1, T, block.cfg.dim)
    e = torch.randn(1, T, block.cfg.dim)
    with torch.no_grad():
        block(h, e, freqs, None, 3, None)
    assert called["n"] == 0
