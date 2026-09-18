"""Hermetic tests for tokenizer/vocab helpers (no network downloads)."""
from __future__ import annotations

import open_mythos
from open_mythos import tokenizer as tk
from open_mythos.main import MythosConfig


class _FakeTokenizer:
    """Stand-in for MythosTokenizer that needs no network."""

    def __init__(self, model_id: str = "fake"):
        self.model_id = model_id

    @property
    def vocab_size(self) -> int:
        return 12345


def test_all_public_exports_resolve():
    # __all__ previously listed load_tokenizer/get_vocab_size, which did not
    # exist — `from open_mythos import *` would then fail. Guard against it.
    for name in open_mythos.__all__:
        assert hasattr(open_mythos, name), f"missing export: {name}"


def test_get_vocab_size_uses_tokenizer(monkeypatch):
    monkeypatch.setattr(tk, "MythosTokenizer", _FakeTokenizer)
    assert tk.get_vocab_size() == 12345


def test_configure_vocab_size_syncs_and_returns_cfg(monkeypatch):
    monkeypatch.setattr(tk, "MythosTokenizer", _FakeTokenizer)
    cfg = MythosConfig(vocab_size=32000)
    returned = tk.configure_vocab_size(cfg)
    assert returned is cfg
    assert cfg.vocab_size == 12345  # now matches the tokenizer, not the default


def test_load_tokenizer_constructs(monkeypatch):
    monkeypatch.setattr(tk, "MythosTokenizer", _FakeTokenizer)
    tok = tk.load_tokenizer("some/model")
    assert isinstance(tok, _FakeTokenizer) and tok.model_id == "some/model"
