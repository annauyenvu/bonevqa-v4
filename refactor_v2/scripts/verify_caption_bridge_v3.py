"""Bounded real-data checkpoint/routing audit; not a performance benchmark."""
import argparse
import json
from pathlib import Path

import torch

from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.train import build_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="refactor_v2/checkpoints/pilot_roco_bridge_trainmode_v3")
    parser.add_argument("--compare_resume_dir")
    parser.add_argument("--device", choices=["cpu", "mps"], default=None)
    args = parser.parse_args()
    folder = Path(args.run_dir)
    cfg = json.loads((folder / "config.json").read_text())
    data = build_split(build_dataset, "roco", "model_val", 4, True, cfg["data"]["loader_kwargs"])
    batch = collate_fn([data[0], data[1]])
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = BoneVQAModel.load(str(folder / "last.pt"), device=device).eval()

    def inspect(inputs):
        with torch.no_grad():
            core = model._encode_core(inputs["images"], inputs["questions"], inputs["regions"],
                                      [None] * len(inputs["images"]), "contour", False)
            prefix = model._llm_prefix(core).detach().cpu()
            embeds, attention, labels = model._build_llm_inputs(
                model._llm_prefix(core), core["q_texts"], inputs["answers"])
            logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
            # Compare ONLY answer-prediction positions; prefix differences
            # alone do not establish that the language answer path uses images.
            selected = labels[:, 1:] != -100
            answer_logits = logits[:, :-1][selected].float().detach().cpu()
            loss = float(model(inputs, stage=1)["loss"])
        return prefix, answer_logits, loss

    before = inspect(batch)
    repeat = inspect(batch)
    swapped = dict(batch)
    swapped["images"] = list(reversed(batch["images"]))
    intervention = inspect(swapped)
    model.load_trainable(str(folder / "last.pt"))
    model.eval()
    restored = inspect(batch)
    def delta(left, right):
        return float((left - right).abs().max())
    payload = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    optimizer = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=False)
    extra = payload["extra"]
    report = {
        "scope": "two held-out ROCO captions; technical audit, not semantic superiority",
        "device": device,
        "repeat_answer_logit_max_difference": delta(before[1], repeat[1]),
        "reload_answer_logit_max_difference": delta(before[1], restored[1]),
        "image_swap_prefix_max_difference": delta(before[0], intervention[0]),
        "image_swap_answer_logit_max_difference": delta(before[1], intervention[1]),
        "loss_before_reload": before[2], "loss_after_reload": restored[2],
        "optimizer_model_epoch_match": optimizer["epoch"] == extra["epoch"],
        "optimizer_model_step_match": optimizer["global_step"] == extra["global_step"],
        "resume_equivalence_verified": False,
    }
    report["passed"] = (
        report["repeat_answer_logit_max_difference"] == 0
        and report["reload_answer_logit_max_difference"] == 0
        and report["image_swap_prefix_max_difference"] > 0
        and report["image_swap_answer_logit_max_difference"] > 0
        and report["optimizer_model_epoch_match"] and report["optimizer_model_step_match"]
        and abs(before[2] - restored[2]) < 1e-6
    )
    if args.compare_resume_dir:
        resumed = torch.load(Path(args.compare_resume_dir) / "last.pt", map_location="cpu", weights_only=False)
        resumed_optimizer = torch.load(Path(args.compare_resume_dir) / "optimizer.pt", map_location="cpu", weights_only=False)
        def equal_tree(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and torch.equal(left, right)
            if isinstance(left, dict):
                return isinstance(right, dict) and left.keys() == right.keys() and all(
                    equal_tree(value, right[key]) for key, value in left.items())
            if isinstance(left, (tuple, list)):
                return type(left) is type(right) and len(left) == len(right) and all(
                    equal_tree(a, b) for a, b in zip(left, right))
            return left == right
        differences = {
            key: float((value.float() - resumed["state"][key].float()).abs().max())
            for key, value in payload["state"].items()
        }
        report["resume_comparison_run"] = args.compare_resume_dir
        report["resume_parameter_max_difference"] = max(differences.values())
        report["resume_differing_tensors"] = sum(value > 0 for value in differences.values())
        report["resume_optimizer_state_equal"] = equal_tree(optimizer["optimizer"], resumed_optimizer["optimizer"])
        report["resume_scheduler_state_equal"] = equal_tree(optimizer["scheduler"], resumed_optimizer["scheduler"])
        report["resume_model_config_equal"] = payload["config"] == resumed["config"]
        report["resume_equivalence_verified"] = (
            report["resume_parameter_max_difference"] == 0
            and payload["extra"]["global_step"] == resumed["extra"]["global_step"]
            and report["resume_optimizer_state_equal"] and report["resume_scheduler_state_equal"]
            and report["resume_model_config_equal"]
        )
        report["passed"] = report["passed"] and report["resume_equivalence_verified"]
    path = Path("refactor_v2/reports") / f"caption_bridge_real_audit_{folder.name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
