"""Real-model forward/backward reproducibility, with no optimizer updates."""
import json
from pathlib import Path

import torch

from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.train import build_split, seed_training_epoch


def main():
    folder = Path("refactor_v2/checkpoints/pilot_roco_epochseed_continuous_v3")
    cfg = json.loads((folder / "config.json").read_text())
    dataset = build_split(build_dataset, "roco", "model_val", 4, True, cfg["data"]["loader_kwargs"])
    batch = collate_fn([dataset[0], dataset[1]])
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS diagnostic cannot fall back to CPU")
    model = BoneVQAModel.load(str(folder / "last.pt"), device="mps")
    parameters = dict(model.named_parameters())
    selected = ["qformer.query", "prefix_proj.0.weight"]
    for prefix in ("visual.", "llm."):
        selected.append(next(k for k in parameters if k.startswith(prefix) and "lora_B" in k))
    results = []
    reference = None
    for label, training, checkpoint in (("train_without_checkpoint", True, False),
                                        ("train_with_checkpoint", True, True),
                                        ("eval_without_checkpoint", False, False)):
        model.visual.grad_checkpoint = checkpoint
        model.visual.trunk.set_grad_checkpointing(checkpoint)
        if checkpoint:
            model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.llm.gradient_checkpointing_disable()
        snapshots, losses = [], []
        for repeat in range(3):
            model.train(training)
            model.zero_grad(set_to_none=True)
            seed_training_epoch(42, 0)
            out = model(batch, stage=1)
            out["loss"].backward()
            losses.append(float(out["loss"].detach().cpu()))
            snapshots.append({key: parameters[key].grad.detach().cpu().clone()
                              for key in selected})
        results.append({
            "case": label, "losses": losses,
            "gradient_max_differences_first_vs_second": {
                key: float((snapshots[0][key] - snapshots[1][key]).abs().max()) for key in selected},
            "gradient_max_differences_second_vs_third": {
                key: float((snapshots[1][key] - snapshots[2][key]).abs().max()) for key in selected},
        })
        if label == "train_without_checkpoint":
            reference = snapshots[0]
        if label == "train_with_checkpoint":
            results[-1]["gradient_max_differences_vs_no_checkpoint"] = {
                key: float((reference[key] - snapshots[0][key]).abs().max()) for key in selected}
        print(json.dumps(results[-1]), flush=True)
    report = {"torch": torch.__version__, "device": "mps", "optimizer_steps": 0,
              "scope": "same-process reseeded forward/backward; not full resume certification", "results": results}
    Path("refactor_v2/reports/resume_stochastic_isolation_v3.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
