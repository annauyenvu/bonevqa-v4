"""MPS-vs-CPU gradient regression tests (skipped without MPS).

Two torch 2.6 MPS bugs passed every loss-level check (2026-10-30): layer_norm backward gives wrong
weight/bias grads when its input does not require grad, and adaptive_avg_pool2d is wrong on the
strided patch-grid view used for local crop features.  These tests compare every parameter gradient.
"""
import pytest
import torch
import torch.nn.functional as F

import src.models.bonevqa as bonevqa
import test_v5_model as T

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def test_layer_norm_grads_with_frozen_input():
    import experiments.v5.model_v5  # noqa: F401  (installs the MPS layer_norm fix)

    torch.manual_seed(0)
    x, g = torch.randn(125, 768), torch.randn(4, 125, 768)
    out = {}
    for dev in ("cpu", "mps"):
        ln = torch.nn.LayerNorm(768).to(dev)
        ln(x.to(dev).unsqueeze(0).expand(4, -1, -1)).backward(g.to(dev))
        out[dev] = (ln.weight.grad.cpu(), ln.bias.grad.cpu())
    for a, b in zip(out["cpu"], out["mps"]):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_patch_grid_pooling_matches_cpu():
    torch.manual_seed(0)
    raw = torch.randn(5, 197, 768)

    def pool(dev):
        grid = raw.to(dev)[:, 1:].transpose(1, 2).reshape(5, 768, 14, 14)
        if grid.device.type == "mps":
            grid = grid.contiguous()  # same guard as model_v5._encode_core
        return F.adaptive_avg_pool2d(grid, (2, 2)).cpu()

    assert torch.allclose(pool("cpu"), pool("mps"), atol=1e-5)


@pytest.mark.parametrize("over", [
    {},
    {"v5_local": True},
    {"v5_local": True, "v5_hag": "replace", "v5_local_view": "circle"},
    {"v5_local": True, "v5_latent": True, "v5_pf": "graph"},
])
def test_v5_parameter_grads_match_cpu(monkeypatch, over):
    from experiments.v5.model_v5 import BoneVQAv5
    from experiments.v5.qualify_mps_grads_v5 import compare

    monkeypatch.setattr(bonevqa, "VisualEncoder", T.FakeVisual)
    monkeypatch.setattr(bonevqa, "TextEncoder", T.FakeText)
    if over.get("v5_pf"):
        over = {**over, "v5_kg": T.kg()}
    grads = {}
    for dev in ("cpu", "mps"):
        m = T.build(BoneVQAv5, dropout=0.0, **over)
        m.to(dev)
        m.train()
        out = m(T.batch(), stage=3)
        out["loss"].backward()
        grads[dev] = ({n: p.grad.detach().cpu() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None},
                      [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None])
    result = compare(grads["cpu"][0], grads["mps"][0], grads["cpu"][1], grads["mps"][1])
    assert result["passed"], result["problems"]
