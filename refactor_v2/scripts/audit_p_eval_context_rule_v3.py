"""Frozen TRAIN-only P eval-context rule and batch-composition diagnostics."""
import hashlib
import json
import time
from pathlib import Path
import torch
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split


def main():
    start = time.perf_counter()
    checkpoint = Path("refactor_v2/checkpoints/component_sampler_resume_train_only_v3_p_shuffled/continuous/last.pt")
    output = Path("refactor_v2/reports/p_eval_context_rule_train_only_v3.json")
    if output.exists():
        raise RuntimeError("Existing eval-context audit must not be overwritten")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    dataset = build_split(build_dataset, config["data"]["dataset"], "train", None, True, config["data"]["loader_kwargs"])
    model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
    # This override is the NEW candidate rule, not a silent relabelling of the
    # historical smoke checkpoint's original evaluation outputs.
    model.cfg["latent_context_control_eval"] = "conditioned"
    captured = []
    def capture(module, args, kwargs):
        captured.append(kwargs["latent_context_control"])
    handle = model.fusion.register_forward_pre_hook(capture, with_kwargs=True)
    @torch.no_grad()
    def evaluate(indices):
        batch = collate_fn([dataset[i] for i in indices])
        nb = normalize_batch(batch)
        core = model._encode_core(nb["images"], nb["questions"], nb["regions"],
                                  [None] * len(indices), model.cfg["prompt_kind"], False)
        # Also exercise the production prediction entry point with singleton
        # batches: a leftover shuffled LF route would fail its >=2 check.
        predictions = model.predict_batch(batch, tra_chi_tiet=True)
        return {"closed": core["logits_closed"].cpu(), "fracture": core["fracture_logit"].cpu(),
                "p_yes": torch.tensor([row["p_yes"] for row in predictions])}
    indices = [0, 1, 4, 5]
    full = evaluate(indices)
    single = [evaluate([index]) for index in indices]
    reversed_batch = evaluate(list(reversed(indices)))
    errors = {key: {"singleton_vs_batch": float((value - torch.cat([row[key] for row in single])).abs().max()),
                    "reordered_batch": float((value - reversed_batch[key].flip(0)).abs().max())}
              for key, value in full.items()}
    handle.remove()
    tolerance = 1e-5
    report = {"scope": "4 existing CLOSED outer-TRAIN rows, frozen smoke weights; new explicit eval override; no efficacy claim",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "train_context": model.cfg["latent_context_control"], "eval_context": model.cfg["latent_context_control_eval"],
        "all_fusion_calls_conditioned": bool(captured) and all(value == "conditioned" for value in captured),
        "tolerance": tolerance, "endpoint_errors": errors, "seconds": time.perf_counter() - start,
        "passed": bool(captured) and all(value == "conditioned" for value in captured)
            and all(error <= tolerance for values in errors.values() for error in values.values())}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
