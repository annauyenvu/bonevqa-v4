import torch

from src.models.heads import FracturePresenceHead


def test_fracture_head_is_binary_and_trainable():
    head = FracturePresenceHead(16, dropout=0.0)
    fused = torch.randn(5, 16, requires_grad=True)
    logits = head(fused)
    assert logits.shape == (5,)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]), pos_weight=torch.tensor(1.5)
    ).backward()
    assert all(p.grad is not None for p in head.parameters())
