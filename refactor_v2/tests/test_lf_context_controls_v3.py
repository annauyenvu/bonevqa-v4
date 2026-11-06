import pytest
import torch

from src.models.fusion import MultimodalFusion
from src.context_controls import lf_context_mode


def test_training_only_shuffled_control_has_per_image_eval_rule():
    cfg = {"latent_context_control": "shuffled", "latent_context_control_eval": "conditioned"}
    assert lf_context_mode(cfg, True) == "shuffled"
    assert lf_context_mode(cfg, False) == "conditioned"
    cfg["latent_context_control_eval"] = "shuffled"
    with pytest.raises(ValueError, match="batch donors"):
        lf_context_mode(cfg, False)


def inputs():
    torch.manual_seed(8)
    fusion = MultimodalFusion(8, 2, 0.0, use_latent=True).eval()
    fusion.gamma_latent.data.fill_(0.1)
    image, text = torch.randn(4, 5, 8), torch.randn(4, 3, 8)
    prompt = torch.randn(1, 2, 8).expand(4, -1, -1)
    return fusion, image, text, prompt


def test_unconditioned_context_removes_row_semantics_not_capacity():
    fusion, image, text, prompt = inputs()
    real = fusion(image, text, prompt)
    control = fusion(image, text, prompt, latent_context_control="unconditioned")
    assert torch.equal(real["f_i"], control["f_i"])
    assert torch.equal(real["f_l"], control["f_l"])
    assert torch.equal(real["f_fi"], control["f_fi"])
    assert torch.equal(real["f_fl"], control["f_fl"])
    assert torch.allclose(control["latent_delta"], control["latent_delta"][:1].expand(4, -1), atol=1e-7)
    assert not torch.allclose(real["latent_delta"], control["latent_delta"])


def test_context_controls_cannot_change_baseline_when_gate_is_zero():
    fusion, image, text, prompt = inputs()
    fusion.gamma_latent.data.zero_()
    baseline = fusion(image, text, None)["x_f"]
    for mode in ("conditioned", "unconditioned", "shuffled"):
        assert torch.equal(baseline, fusion(image, text, prompt, latent_context_control=mode)["x_f"])


def test_shuffled_context_rejects_singleton_instead_of_identity_placebo():
    fusion, image, text, prompt = inputs()
    with pytest.raises(ValueError, match="at least two"):
        fusion(image[:1], text[:1], prompt[:1], latent_context_control="shuffled")


def test_scientific_training_cannot_silently_use_same_image_context_donors():
    from src.train import validate_training_runtime
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        validate_training_runtime({"latent_context_control": "shuffled"}, "cpu")


@pytest.mark.parametrize("violation", ["scope", "epoch", "sampler"])
def test_bounded_component_smoke_rejects_unscoped_or_unbounded_configs(tmp_path, violation):
    import json
    from src.train import validate_component_smoke_config
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"scope": "development" if violation == "scope" else "technical_smoke_train_only"}))
    cfg = {"data": {"loader_kwargs": {"seal_path": str(seal)}}, "train": {
        "epochs": 3 if violation == "epoch" else 2,
        "component_batching": None if violation == "sampler" else {"manifest": "bound"}}}
    with pytest.raises(RuntimeError, match="Technical component smoke"):
        validate_component_smoke_config(cfg)
