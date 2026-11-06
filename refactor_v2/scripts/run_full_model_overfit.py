"""Overfit 32 real BoneVQAModel forward paths with lightweight test encoders.

This isolates integration/routing correctness from pretrained model downloads.
It exercises the production BoneVQAModel, losses, fusion, masks and V/L/P gates.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import src.models.bonevqa as model_module


class TestVisualEncoder(nn.Module):
    def __init__(self, backbone, d_model, lora_rank, image_size=32, **kwargs):
        super().__init__()
        self.image_size, self.d_model = image_size, d_model
        self.kind, self.shared_clip = "test", None
        self.patch = nn.Linear(3, d_model)

    def preprocess(self, images, device):
        rows = []
        for image in images:
            arr = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            rows.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(rows).to(device)

    def forward(self, pixels):
        patches = F.adaptive_avg_pool2d(pixels, (2, 2)).flatten(2).transpose(1, 2)
        patches = self.patch(patches)
        cls = patches.mean(1, keepdim=True)
        tokens = torch.cat([cls, patches], dim=1)
        return {"tokens": tokens, "embed": F.normalize(cls[:, 0], dim=-1)}


class TestTextEncoder(nn.Module):
    def __init__(self, visual_encoder=None, d_model=16, max_len=8, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(64, d_model)

    def forward(self, texts, device):
        ids = torch.tensor([[sum(map(ord, text)) % 64, len(text) % 64] for text in texts], device=device)
        tokens = self.embedding(ids)
        return {"tokens": tokens, "mask": torch.ones_like(ids, dtype=torch.bool),
                "embed": F.normalize(tokens.mean(1), dim=-1)}

    def encode_frozen(self, texts, device):
        return self.forward(texts, device)


def batch_32():
    images, masks, answers, ids = [], [], [], []
    for index in range(32):
        positive = index % 2 == 1
        value = 220 if positive else 25
        arr = np.full((32, 32, 3), value, dtype=np.uint8)
        arr[:, index % 16:index % 16 + 3] = 255 - value
        images.append(Image.fromarray(arr))
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[:, 8:24, 8:24] = 1
        masks.append(mask)
        answers.append("yes" if positive else "no")
        ids.append(f"overfit-{index}")
    return {"images": images, "questions": ["Is there a fracture in this image?"] * 32,
            "answers": answers, "answer_types": ["CLOSED"] * 32, "regions": ["hand"] * 32,
            "masks": masks, "boxes": [None] * 32, "image_ids": ids}


def run_config(name, flags):
    torch.manual_seed(17)
    cfg = {
        "d_model": 16, "n_heads": 2, "dropout": 0.0, "image_size": 32,
        "visual_lora_rank": 0, "grad_checkpoint": False, "n_query": 4,
        "qformer_layers": 1, "n_latent": 4, "lens_layers": 1, "lens_shuffle_r": 2,
        "contrastive_dim": 8, "use_llm": False, "use_textual_prompt": False,
        "use_visual_prompt": flags[0], "use_lens": flags[1], "use_latent_prompt": flags[2],
        "sam": {"enabled": False, "model_id": "unused", "grid": 2},
        "prior_labels": [], "eta": 0.01, "lam": 0.0, "type_loss_weight": 0.0,
    }
    model = model_module.BoneVQAModel(cfg, ["yes", "no"], device="cpu")
    data = batch_32()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    initial = None
    first_grads = {}
    for step in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(data, stage=3)
        if initial is None:
            initial = float(output["loss"])
        output["loss"].backward()
        if step == 0:
            for key, parameter in (("V", model.prompt_mixer.gamma_visual), ("L", model.prompt_mixer.gamma_lens),
                                   ("P", model.fusion.gamma_latent)):
                if parameter is not None:
                    first_grads[key] = float(parameter.grad.abs())
        optimizer.step()
    model.eval()
    with torch.no_grad():
        core = model._encode_core(data["images"], data["questions"], data["regions"], data["masks"], "contour", False)
        prediction = core["logits_closed"].argmax(1)
        target = torch.tensor([model.answer_to_idx[a] for a in data["answers"]])
        accuracy = float((prediction == target).float().mean())
        final = float(model(data, stage=3)["loss"])
    required = [key for key, enabled in zip(("V", "L", "P"), flags) if enabled]
    passed = accuracy >= 0.98 and final < initial * 0.25 and all(first_grads.get(k, 0) > 0 for k in required)
    return {"name": name, "initial_loss": initial, "final_loss": final, "accuracy": accuracy,
            "first_step_gate_gradients": first_grads, "passed": passed}


def main():
    model_module.VisualEncoder = TestVisualEncoder
    model_module.TextEncoder = TestTextEncoder
    configs = {"None": (False, False, False), "V": (True, False, False), "L": (False, True, False),
               "P": (False, False, True), "VLP": (True, True, True)}
    results = [run_config(name, flags) for name, flags in configs.items()]
    report = {"kind": "full BoneVQAModel integration overfit with lightweight encoders", "samples": 32,
              "all_passed": all(row["passed"] for row in results), "results": results}
    path = Path("refactor_v2/reports/full_model_overfit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
