"""Hermetic CPU tests for the code fine-tuning script (no GPU, no downloads)."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch

from open_mythos.main import MythosConfig, OpenMythos

_FT_PATH = Path(__file__).resolve().parents[1] / "training" / "finetune_code.py"
_spec = importlib.util.spec_from_file_location("finetune_code", _FT_PATH)
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def _cfg(**overrides) -> MythosConfig:
    base = dict(
        vocab_size=64, dim=32, n_heads=4, n_kv_heads=2, max_seq_len=32,
        max_loop_iters=3, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4,
    )
    base.update(overrides)
    return MythosConfig(**base)


def _synth(vocab: int, mb: int, seq: int):
    while True:
        x = torch.randint(0, vocab, (mb, seq))
        yield x, x.clone()


def test_train_loop_runs_and_checkpoints():
    torch.manual_seed(0)
    cfg = _cfg()
    model = OpenMythos(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with tempfile.TemporaryDirectory() as d:
        final = ft.train_loop(
            model, opt, _synth(cfg.vocab_size, 2, 16), cfg=cfg,
            vocab_size=cfg.vocab_size, device="cpu", total_steps=4, grad_accum=2,
            warmup_steps=1, max_lr=3e-4, log_every=0, ckpt_every=2, ckpt_dir=d,
        )
        assert final == 4
        assert ft._list_ckpts(d)  # at least one checkpoint written


def test_checkpoint_resume_roundtrip():
    torch.manual_seed(0)
    cfg = _cfg()
    m1 = OpenMythos(cfg)
    o1 = torch.optim.AdamW(m1.parameters(), lr=3e-4)
    with tempfile.TemporaryDirectory() as d:
        ft.train_loop(
            m1, o1, _synth(cfg.vocab_size, 2, 16), cfg=cfg,
            vocab_size=cfg.vocab_size, device="cpu", total_steps=2, grad_accum=1,
            warmup_steps=1, max_lr=3e-4, log_every=0, ckpt_every=2, ckpt_dir=d,
        )
        m2 = OpenMythos(cfg)
        o2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
        step = ft.load_checkpoint(m2, o2, ft._list_ckpts(d)[-1], ddp=False)
        assert step == 2
        assert torch.equal(
            m1.state_dict()["embed.weight"], m2.state_dict()["embed.weight"]
        )


def test_load_base_checkpoint_weights_only():
    torch.manual_seed(0)
    cfg = _cfg()
    m1 = OpenMythos(cfg)
    o1 = torch.optim.AdamW(m1.parameters(), lr=3e-4)
    with tempfile.TemporaryDirectory() as d:
        ft.save_checkpoint(m1, o1, 5, cfg, cfg.vocab_size, d, ddp=False, master=True)
        m2, cfg2, vocab = ft.load_base_checkpoint(ft._list_ckpts(d)[-1])
        assert vocab == cfg.vocab_size and cfg2.dim == cfg.dim
        assert torch.equal(
            m1.state_dict()["embed.weight"], m2.state_dict()["embed.weight"]
        )


def test_get_lr_warmup_then_decay():
    lrs = [ft.get_lr(s, 2, 10, 1e-3, 1e-4) for s in range(12)]
    assert lrs[0] < lrs[1] <= max(lrs)
    assert abs(lrs[-1] - 1e-4) < 1e-9


def test_parse_args_defaults():
    a = ft.parse_args([])
    assert a.lr == 3e-5
    assert a.grad_checkpoint is False  # opt-in, off by default
    assert a.dataset  # a default code dataset is set
