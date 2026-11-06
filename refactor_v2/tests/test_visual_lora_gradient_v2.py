import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.encoders import VisualEncoder


class _CheckpointedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4, bias=False)
        self.frozen.weight.requires_grad_(False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        self.enabled = False

    def set_grad_checkpointing(self, enabled=True):
        self.enabled = enabled

    def _block(self, x):
        return self.frozen(x) + self.lora_B(self.lora_A(x))

    def forward_features(self, x):
        flat = x.flatten(2).transpose(1, 2)
        return checkpoint(self._block, flat, use_reentrant=True) if self.enabled else self._block(flat)


def test_checkpointed_visual_adapter_receives_gradient():
    encoder = VisualEncoder.__new__(VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.kind = "open_clip"
    encoder.grad_checkpoint = True
    encoder.training = True
    encoder.trunk = _CheckpointedTrunk()

    pixels = torch.randn(2, 4, 1, 1)
    tokens = encoder._forward_tokens(pixels)
    tokens.square().mean().backward()

    assert pixels.requires_grad is False
    assert encoder.trunk.lora_A.weight.grad is not None
    assert encoder.trunk.lora_B.weight.grad is not None
    assert torch.count_nonzero(encoder.trunk.lora_A.weight.grad).item() > 0
    assert torch.count_nonzero(encoder.trunk.lora_B.weight.grad).item() > 0
