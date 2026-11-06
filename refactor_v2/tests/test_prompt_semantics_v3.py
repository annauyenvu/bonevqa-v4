import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.data.prompts import build_lens, textual_prompt
from src.models.bonevqa import BoneVQAModel, _build_with_isolated_rng
from src.models.encoders import TextEncoder
from src.models.latent_prompt import LatentPromptGenerator


class _DropoutFrozenText(TextEncoder):
    """Small test double exercising TextEncoder.encode_frozen state handling."""

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Dropout(p=0.9)
        self.proj_head = None
        self.proj_tokens = nn.Identity()

    def forward(self, texts, device):
        x = torch.ones(len(texts), 3, 4, device=device)
        return {"tokens": self.model(x), "mask": torch.ones(len(texts), 3, dtype=torch.bool, device=device)}


def test_frozen_text_memory_is_deterministic_and_restores_training_state():
    encoder = _DropoutFrozenText().train()
    first = encoder.encode_frozen(["yes", "no"], torch.device("cpu"))["tokens"]
    second = encoder.encode_frozen(["yes", "no"], torch.device("cpu"))["tokens"]
    assert torch.equal(first, second)
    assert encoder.model.training is True


def test_optional_module_construction_does_not_advance_shared_rng():
    torch.manual_seed(17)
    expected = torch.randn(5)

    torch.manual_seed(17)
    _build_with_isolated_rng(lambda: nn.Sequential(nn.Linear(11, 13), nn.Linear(13, 7)), 999)
    actual = torch.randn(5)

    assert torch.equal(actual, expected)


def test_latent_prompt_roundtrip_preserves_output_after_bank_rebuild():
    torch.manual_seed(5)
    source = LatentPromptGenerator(8, 3, 2, 0.0, prior_labels=[]).eval()
    bank = torch.randn(6, 8)
    context = torch.randn(2, 4, 8)
    source.set_answer_bank(bank)
    expected, _ = source(context)

    restored = LatentPromptGenerator(8, 3, 2, 0.0, prior_labels=[]).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    restored.set_answer_bank(bank.clone())
    actual, _ = restored(context)

    assert torch.equal(actual, expected)


def test_answer_bank_changes_latent_prompt_but_context_does_not_precondition_it():
    torch.manual_seed(7)
    latent = LatentPromptGenerator(8, 3, 2, 0.0, prior_labels=[]).eval()
    context_a = torch.randn(2, 4, 8)
    context_b = torch.randn(2, 4, 8)
    latent.set_answer_bank(torch.randn(6, 8))
    prompt_a, _ = latent(context_a)
    prompt_b, _ = latent(context_b)
    assert torch.equal(prompt_a, prompt_b)

    latent.set_answer_bank(torch.randn(6, 8))
    prompt_other_bank, _ = latent(context_a)
    assert not torch.allclose(prompt_a, prompt_other_bank)


def test_lens_sampling_is_controlled_by_global_seed():
    image = Image.new("RGB", (8, 8), (120, 120, 120))
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1
    np.random.seed(123)
    first = np.asarray(build_lens(image, mask, mode="unicolor"))
    np.random.seed(123)
    second = np.asarray(build_lens(image, mask, mode="unicolor"))
    assert np.array_equal(first, second)


def test_textual_controls_do_not_smuggle_modality_or_region():
    question = "Is there a fracture?"
    clinical = textual_prompt(question, region=None, style="clinical")
    irrelevant = textual_prompt(question, region=None, style="irrelevant")
    paraphrase = textual_prompt(question, region=None, style="clinical_paraphrase")
    assert "X-ray" in clinical
    assert "X-ray" in paraphrase
    assert "X-ray" not in irrelevant and "bone" not in irrelevant.lower()
    assert question in clinical and question in irrelevant and question in paraphrase
    assert "Question:" not in clinical
    formatted = BoneVQAModel._format_llm_prompt(clinical)
    assert formatted.count("Question:") == 1
    assert formatted == f"Question: {clinical} Answer:"


def test_matched_textual_control_preserves_question_without_clinical_claims():
    question = "Is there a fracture in this image?"
    control = textual_prompt(question, region=None, style="irrelevant_matched")
    header = control[:-len(question)]
    assert control.endswith(question)
    assert "bone" not in header.lower() and "x-ray" not in header.lower()
    assert "fracture" not in header.lower()
    assert control.count(question) == 1

