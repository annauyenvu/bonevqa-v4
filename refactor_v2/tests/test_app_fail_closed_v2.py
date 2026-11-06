import pytest

from app import inference


def test_missing_checkpoint_fails_closed(monkeypatch):
    monkeypatch.setenv("BONEVQA_CKPT", "/definitely/missing/checkpoint.pt")
    monkeypatch.delenv("BONEVQA_ALLOW_MOCK", raising=False)
    inference._ENGINE = None
    with pytest.raises(RuntimeError, match="BONEVQA_CKPT"):
        inference.get_engine()


def test_mock_requires_explicit_development_flag(monkeypatch):
    monkeypatch.setenv("BONEVQA_CKPT", "/definitely/missing/checkpoint.pt")
    monkeypatch.setenv("BONEVQA_ALLOW_MOCK", "1")
    inference._ENGINE = None
    assert isinstance(inference.get_engine(), inference.MockEngine)
    inference._ENGINE = None

