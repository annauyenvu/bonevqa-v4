"""Fast 64-sample gradient and overfit gate for repaired V/L/P routing."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import MultimodalFusion
from src.models.latent_prompt import LatentPromptGenerator
from src.models.routing import ResidualPromptMixer


class TinyEndpointModel(nn.Module):
    def __init__(self, d_model, use_v, use_l, use_p):
        super().__init__()
        self.use_v, self.use_l, self.use_p = use_v, use_l, use_p
        self.mixer = ResidualPromptMixer(d_model, use_v, use_l)
        self.latent = LatentPromptGenerator(d_model, 4, 2, 0.0) if use_p else None
        self.fusion = MultimodalFusion(d_model, 2, 0.0, use_latent=use_p)
        self.head = nn.Linear(d_model, 2)

    def forward(self, base, visual, lens, text):
        image = self.mixer(base, visual if self.use_v else None, lens if self.use_l else None)
        prompt = None
        if self.latent is not None:
            context = torch.cat([image, text], dim=1)
            prompt, _ = self.latent(context, torch.ones(context.shape[:2], dtype=torch.bool))
        fused = self.fusion(image, text, prompt)["x_f"]
        return self.head(fused)


def run_one(name, flags, seed, samples, steps):
    torch.manual_seed(seed)
    d = 16
    signal = torch.randn(samples, 1)
    noise = lambda n: torch.randn(samples, n, d) * 0.25
    direction = torch.randn(d)
    base = noise(5) + signal[:, :, None] * direction
    visual = noise(4) + signal[:, :, None] * direction.roll(1)
    lens = noise(6) + signal[:, :, None] * direction.roll(2)
    text = noise(3) + signal[:, :, None] * direction.roll(3)
    target = (signal[:, 0] > 0).long()
    model = TinyEndpointModel(d, *flags)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)

    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(base, visual, lens, text), target))
    first_gradients = {}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(base, visual, lens, text)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        if step == 0:
            for parameter_name in ("gamma_visual", "gamma_lens"):
                parameter = getattr(model.mixer, parameter_name)
                if parameter is not None:
                    first_gradients[parameter_name] = float(parameter.grad.abs())
            if model.latent is not None:
                first_gradients["gamma_latent"] = float(model.fusion.gamma_latent.grad.abs())
        optimizer.step()
    with torch.no_grad():
        logits = model(base, visual, lens, text)
        final_loss = float(F.cross_entropy(logits, target))
        accuracy = float((logits.argmax(1) == target).float().mean())
    gates = model.mixer.gate_values()
    gates["latent"] = None if model.latent is None else float(model.fusion.gamma_latent.detach())
    passed = accuracy >= 0.98 and final_loss < initial_loss * 0.2
    if flags[0]:
        passed = passed and first_gradients.get("gamma_visual", 0.0) > 0
    if flags[1]:
        passed = passed and first_gradients.get("gamma_lens", 0.0) > 0
    if flags[2]:
        passed = passed and first_gradients.get("gamma_latent", 0.0) > 0
    return {
        "name": name,
        "samples": samples,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accuracy": accuracy,
        "first_step_gate_gradients": first_gradients,
        "learned_gates": gates,
        "passed": bool(passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=Path("refactor_v2/reports/step3_gate.json"))
    args = parser.parse_args()
    configs = {
        "None": (False, False, False),
        "V": (True, False, False),
        "L": (False, True, False),
        "P": (False, False, True),
        "VLP": (True, True, True),
    }
    results = [run_one(name, flags, args.seed, args.samples, args.steps) for name, flags in configs.items()]
    report = {"all_passed": all(row["passed"] for row in results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
