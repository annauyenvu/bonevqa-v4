import pytest
import torch

from src.checkpointing import checkpoint_manifest, compatible_state, validate_manifest


def test_wrong_shape_is_not_silently_skipped():
    expected = {"head.weight": torch.zeros(2, 3)}
    incoming = {"head.weight": torch.zeros(3, 3)}
    with pytest.raises(RuntimeError, match="wrong_shape"):
        compatible_state(expected, incoming)


def test_manifest_binds_vocab_and_architecture():
    cfg = {"d_model": 8}
    vocab = ["yes", "no"]
    state = {"head.weight": torch.zeros(2, 8)}
    payload = {"manifest": checkpoint_manifest(cfg, vocab, state)}
    validate_manifest(payload, cfg, vocab)
    with pytest.raises(RuntimeError, match="answer_vocab_hash"):
        validate_manifest(payload, cfg, ["no", "yes"])

