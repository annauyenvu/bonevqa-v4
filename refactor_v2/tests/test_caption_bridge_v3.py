from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.models.bonevqa as module
from scripts.run_full_model_overfit import (
    TestTextEncoder as TinyText,
    TestVisualEncoder as TinyVisual,
    batch_32,
)


def make_model(monkeypatch, weight):
    monkeypatch.setattr(module, "VisualEncoder", TinyVisual)
    monkeypatch.setattr(module, "TextEncoder", TinyText)
    cfg = dict(d_model=16, n_heads=2, dropout=0.0, image_size=32,
               visual_lora_rank=0, grad_checkpoint=False, n_query=4,
               qformer_layers=1, contrastive_dim=8, use_llm=False,
               use_visual_prompt=False, use_lens=False, use_latent_prompt=False,
               use_textual_prompt=False, use_fracture_head=False,
               stage1_caption_loss_weight=weight, sam={"enabled": False}, prior_labels=[])
    return module.BoneVQAModel(cfg, ["yes", "no"], device="cpu")


def caption_batch():
    batch = {key: value[:2] for key, value in batch_32().items()}
    batch["answer_types"] = ["OPEN", "OPEN"]
    batch["answers"] = ["hand radiograph", "wrist radiograph"]
    return batch


def test_legacy_stage1_keeps_language_loss_disabled(monkeypatch):
    model = make_model(monkeypatch, 0.0)
    assert model(caption_batch(), stage=1)["loss_lm"].item() == 0.0


def test_answer_loss_excludes_auxiliary_alignment(monkeypatch):
    model = make_model(monkeypatch, 0.0)
    batch = {key: value[:2] for key, value in batch_32().items()}
    output = model(batch, stage=3)
    assert output["loss_answer"].requires_grad
    assert torch.equal(output["loss_answer"].detach(), output["loss_cls"])


def test_caption_mode_rejects_missing_llm(monkeypatch):
    model = make_model(monkeypatch, 1.0)
    with pytest.raises(ValueError, match="requires an LLM"):
        model(caption_batch(), stage=1)


def test_disabled_sealed_split_cannot_fall_back_to_other_dataset():
    from src.train import build_split
    with pytest.raises(ValueError, match="Missing sealed manifest"):
        build_split(lambda *args, **kwargs: pytest.fail("loader must not run"),
                    "roco", "test", None, True,
                    {"sealed_records_by_split": {"test": None}})


def test_epoch_seed_replays_cpu_dropout_and_shuffle(monkeypatch):
    from src.train import seed_training_epoch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    state = torch.get_rng_state()
    try:
        assert seed_training_epoch(42, 1) == 43
        first_order = torch.randperm(8)
        first_dropout = torch.nn.functional.dropout(torch.ones(256), p=0.1, training=True)
        seed_training_epoch(42, 1)
        assert torch.equal(first_order, torch.randperm(8))
        assert torch.equal(first_dropout, torch.nn.functional.dropout(torch.ones(256), p=0.1, training=True))
    finally:
        torch.set_rng_state(state)


def test_mps_v3_rejects_unverified_checkpoint_recomputation():
    from src.train import validate_training_runtime
    with pytest.raises(RuntimeError, match="gradient parity"):
        validate_training_runtime({"prompt_protocol_version": 3, "grad_checkpoint": True,
                                   "llm_grad_checkpoint": False}, "mps")
    validate_training_runtime({"prompt_protocol_version": 3, "grad_checkpoint": False,
                               "llm_grad_checkpoint": False}, "mps")
    validate_training_runtime({"prompt_protocol_version": 3}, "cpu")


def test_caption_loss_reaches_qformer_and_prefix(monkeypatch):
    model = make_model(monkeypatch, 1.0)
    class LanguageStub(nn.Module):
        def forward(self, inputs_embeds, **kwargs):
            return SimpleNamespace(loss=inputs_embeds.square().mean())
    model.llm = LanguageStub()
    model.prefix_proj = nn.Linear(16, 16)
    # Isolate integration routing; real pretrained LM is separately checked
    # by verify_caption_bridge_v3.py on sealed real-data captions.
    monkeypatch.setattr(model, "_build_llm_inputs", lambda prefix, questions, answers:
                        (prefix, None, None))
    model(caption_batch(), stage=1)["loss"].backward()
    for part in (model.qformer, model.prefix_proj):
        assert any(p.grad is not None and p.grad.isfinite().all()
                   and p.grad.abs().sum() > 0 for p in part.parameters())
