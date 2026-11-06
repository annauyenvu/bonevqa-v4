import torch

from src.models.fusion import MultimodalFusion
from src.models.latent_prompt import LatentPromptGenerator
from src.models.lens_fusion import LensFusion
from src.models.losses import endpoint_masks, multi_positive_decoupled_loss, multi_positive_info_nce_loss
from src.models.routing import ResidualPromptMixer


def test_v_l_residual_is_baseline_at_initialization_and_reachable():
    torch.manual_seed(1)
    mixer = ResidualPromptMixer(8, use_visual=True, use_lens=True, zero_preserving=True)
    head = torch.nn.Linear(8, 2)
    base, visual, lens = (torch.randn(4, 5, 8) for _ in range(3))
    out = mixer(base, visual, lens)
    assert torch.equal(out, base)
    loss = head(out.mean(1)).square().mean()
    loss.backward()
    assert mixer.gamma_visual.grad is not None and mixer.gamma_visual.grad.abs() > 0
    assert mixer.gamma_lens.grad is not None and mixer.gamma_lens.grad.abs() > 0


def test_p_is_additive_and_input_conditioned():
    torch.manual_seed(2)
    latent = LatentPromptGenerator(8, 3, 2, 0.0)
    latent.set_answer_bank(torch.randn(7, 8))
    context = torch.randn(2, 4, 8)
    prompts, _ = latent(context, torch.ones(2, 4, dtype=torch.bool))
    # LaPA generation queries the shared answer bank.  Sample conditioning is
    # deliberately deferred to the ordered language -> image -> multimodal
    # fusion block.
    assert torch.allclose(prompts[0], prompts[1])
    fusion = MultimodalFusion(8, 2, 0.0, use_latent=True)
    image, text = torch.randn(2, 5, 8), torch.randn(2, 4, 8)
    baseline = fusion(image, text, None)["x_f"]
    prompted_out = fusion(image, text, prompts)
    prompted = prompted_out["x_f"]
    assert torch.allclose(prompted, baseline, atol=1e-6)
    prompted.square().mean().backward()
    assert fusion.gamma_latent.grad is not None and fusion.gamma_latent.grad.abs() > 0

    with torch.no_grad():
        fusion.gamma_latent.fill_(1.0)
        conditioned = fusion(image, text, prompts)["latent_delta"]
    assert not torch.allclose(conditioned[0], conditioned[1])


def test_semantic_deltas_are_zero_for_capacity_matched_views():
    torch.manual_seed(3)
    lens = LensFusion(d_model=8, n_views=3, n_layers=1, n_heads=2,
                      dropout=0.0, grid=2, shuffle_r=1)
    original = torch.randn(2, 5, 8)
    duplicate = original.unsqueeze(1).expand(-1, 3, -1, -1)
    paired = torch.cat([duplicate, duplicate], dim=0)
    out = lens(paired)
    delta = out[:2] - out[2:]
    assert torch.equal(delta, torch.zeros_like(delta))


def test_prompt_mixer_zero_delta_is_exactly_no_prompt_even_with_open_gates():
    mixer = ResidualPromptMixer(8, use_visual=True, use_lens=True, zero_preserving=True)
    with torch.no_grad():
        mixer.gamma_visual.fill_(2.0)
        mixer.gamma_lens.fill_(-3.0)
    base = torch.randn(3, 5, 8)
    zero_v = torch.zeros(3, 2, 8)
    zero_l = torch.zeros(3, 4, 8)
    assert torch.equal(mixer(base, zero_v, zero_l), base)


def test_losses_route_closed_and_open_to_disjoint_endpoints():
    target = torch.tensor([0, 1, 2, -1])
    masks = endpoint_masks(["CLOSED", "OPEN", "CLOSED", "OPEN"], target)
    assert masks["closed"].tolist() == [True, False, True, False]
    assert masks["open"].tolist() == [False, True, False, True]


def test_repeated_image_rows_are_multi_positives():
    image = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    text = image.clone()
    loss = multi_positive_info_nce_loss(image, text, ["same", "same", "other"])
    assert torch.isfinite(loss)
    assert loss >= 0
    dcl = multi_positive_decoupled_loss(image, text, ["same", "same", "other"])
    assert torch.isfinite(dcl)
