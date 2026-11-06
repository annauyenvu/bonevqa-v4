"""Independent completed PMC artifact/provenance checks; no test evaluation."""
import hashlib
import json
from pathlib import Path
import torch
from src.data import build_dataset
from src.protocol import manifest
from src.train import build_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(item):
    if isinstance(item, torch.Tensor):
        return bool(torch.isfinite(item).all())
    if isinstance(item, dict):
        return all(finite_tree(value) for value in item.values())
    if isinstance(item, (tuple, list)):
        return all(finite_tree(value) for value in item)
    return True


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_alignment_execution_v3"
    output = folder / "independent_artifact_verification.json"
    if output.exists():
        raise RuntimeError("Existing verification must not be overwritten")
    status = json.loads((folder / "progress.json").read_text())
    if status["status"] != "training_complete_pending_independent_checkpoint_evaluation" or status["exit_code"] != 0:
        raise RuntimeError("PMC not successfully completed")
    checkpoint = Path(status["checkpoint_dir"])
    cfg = json.loads((checkpoint / "config.json").read_text())
    recorded = json.loads((checkpoint / "data_manifest.json").read_text())
    release = json.loads((folder / "execution_release.json").read_text())
    checks = {}
    for role, recorded_role, count in (("train", "train", 28490), ("model_val", "val", 1500)):
        ds = build_split(build_dataset, "pmc_vqa", role, None, True, cfg["data"]["loader_kwargs"])
        checks[role + "_exact_manifest"] = len(ds) == count and manifest(ds) == recorded[recorded_role]
    best, last, optim = [torch.load(checkpoint / name, map_location="cpu", weights_only=False)
                         for name in ("best.pt", "last.pt", "optimizer.pt")]
    history = json.loads((checkpoint / "history.json").read_text())
    checks.update(best_hash=sha(checkpoint / "best.pt") == status["best_checkpoint_sha256"],
        last_hash=sha(checkpoint / "last.pt") == status["last_checkpoint_sha256"],
        weights_optimizer_finite=finite_tree(best["state"]) and finite_tree(last["state"]) and finite_tree(optim["optimizer"]),
        best_last_tensors_equal=best["state"].keys() == last["state"].keys() and all(torch.equal(value, last["state"][key]) for key, value in best["state"].items()),
        epoch_step_consistency=last["extra"]["epoch"] == optim["epoch"] == 2 and last["extra"]["global_step"] == optim["global_step"] == 3562,
        best_metric_matches_history=best["extra"]["best_metric"] == min(row["val_loss"] for row in history),
        released_config_binding=sha(folder / "released_config.json") == status["config_sha256"],
        fresh_upstream_binding=sha(cfg["init_from"]) == release["fresh_roco_checkpoint_sha256"],
        two_full_validation_epochs=len(history) == 2 and all(row["val_n"] == 1500 for row in history))
    result = {"scope": "PMC upstream checkpoint/provenance only, not independent-test or clinical quality",
        "checks": checks, "passed": all(checks.values()), "state_tensors": len(best["state"]),
        "best_checkpoint_sha256": sha(checkpoint / "best.pt"),
        "validation_losses": [row["val_loss"] for row in history],
        "validation_open_exact": [row["val_open_acc"] for row in history],
        "quality_gate_passed": False, "quality_gate_note": "Low open exact requires prediction/target diagnosis; finite execution is not useful VQA evidence",
        "downstream_stage_released": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
