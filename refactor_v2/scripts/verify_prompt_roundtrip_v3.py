"""Verify deterministic full-model checkpoint round-trip for prompt v3.

This is intentionally an evaluation-only gate.  It reloads the same sealed
checkpoint twice in independent model instances, rebuilds the training-answer
bank, and compares latent/fused/endpoint tensors on the same real-data batch.
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.utils import get_device
from src.train import build_split


@torch.no_grad()
def evaluate_once(checkpoint: Path, batch: dict, device: str):
    model = BoneVQAModel.load(str(checkpoint), device=device)
    model.eval()
    nb = normalize_batch(batch)
    masks = model._prepare_masks(
        nb["images"], nb.get("masks"), nb.get("boxes"), nb.get("image_ids")
    )
    core = model._encode_core(
        nb["images"], nb["questions"], nb.get("regions") or [None] * len(nb["images"]),
        masks, model.cfg["prompt_kind"], False,
    )
    tensors = {
        "answer_bank": model.latent.answer_bank.detach().cpu(),
        "x_lp": core["x_lp"].detach().cpu(),
        "latent_delta": core["fus"]["latent_delta"].detach().cpu(),
        "x_f": core["fus"]["x_f"].detach().cpu(),
        "logits_closed": core["logits_closed"].detach().cpu(),
        "fracture_logit": core["fracture_logit"].detach().cpu(),
    }
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("latent.") or key.startswith("fusion.lp_")
        or key == "fusion.gamma_latent"
    }
    del core, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return tensors, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path,
                        default=Path("refactor_v2/reports/prompt_roundtrip_v3.json"))
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if not cfg.get("use_latent_prompt", False):
        raise RuntimeError("Round-trip gate này yêu cầu checkpoint P.")
    data_cfg = cfg.get("data", {})
    # Config embedded by BoneVQAModel.save contains model-only keys.  Training
    # data provenance is read from the run's saved config.json when available.
    run_cfg_path = args.checkpoint.parent / "config.json"
    if run_cfg_path.exists():
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        data_cfg = run_cfg["data"]
    dataset = build_split(
        build_dataset,
        data_cfg.get("dataset", "fracatlas_component_clean"),
        data_cfg.get("val_split", "model_val"), None, True,
        data_cfg.get("loader_kwargs") or {},
    )
    samples = [dataset[i] for i in range(min(args.samples, len(dataset)))]
    batch = collate_fn(samples)
    device = str(get_device())

    tensors_a, state_a = evaluate_once(args.checkpoint, batch, device)
    tensors_b, state_b = evaluate_once(args.checkpoint, batch, device)
    tensor_diff = {
        key: float((tensors_a[key] - tensors_b[key]).abs().max())
        for key in tensors_a
    }
    state_diff = {
        key: float((state_a[key] - state_b[key]).abs().max())
        for key in state_a
    }
    passed = (
        state_a.keys() == state_b.keys()
        and all(value <= args.atol for value in tensor_diff.values())
        and all(value == 0.0 for value in state_diff.values())
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "samples": len(samples),
        "atol": args.atol,
        "max_abs_output_differences": tensor_diff,
        "max_abs_state_differences": state_diff,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
