import torch

from src.models.losses import single_label_cross_entropy


def test_single_label_cross_entropy_matches_torch_ce():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]])
    target = torch.tensor([0, 1])
    valid = torch.tensor([True, True])
    expected = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(single_label_cross_entropy(logits, target, valid), expected)


def test_single_label_cross_entropy_applies_sample_weight_only_to_valid_rows():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    target = torch.tensor([1, 1, 0])
    valid = torch.tensor([True, True, False])
    weights = torch.tensor([1.0, 4.0, 100.0])
    per = torch.nn.functional.cross_entropy(logits[:2], target[:2], reduction="none")
    expected = (per[0] + 4.0 * per[1]) / 5.0
    assert torch.allclose(single_label_cross_entropy(logits, target, valid, weights), expected)
