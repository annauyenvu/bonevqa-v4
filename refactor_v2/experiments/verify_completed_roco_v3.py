"""Verify generated best/last/optimizer/data lineage, without re-fitting or claiming efficacy."""
import hashlib
import json
from pathlib import Path
import torch
from src.data import build_dataset
from src.protocol import manifest
from src.train import build_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/roco_alignment_execution_v3"
    output = folder / "independent_verification.json"
    if output.exists():
        raise RuntimeError("Existing independent verification must not be overwritten")
    status = json.loads((folder / "progress.json").read_text())
    checkpoint = Path(status["checkpoint_dir"])
    if status["status"] != "training_complete_pending_independent_checkpoint_evaluation" or status["exit_code"] != 0:
        raise RuntimeError("Training has not completed successfully")
    cfg = json.loads((checkpoint / "config.json").read_text())
    recorded = json.loads((checkpoint / "data_manifest.json").read_text())
    checks = {}
    for role, recorded_role in (("train", "train"), ("model_val", "val")):
        dataset = build_split(build_dataset, "roco", role, None, True, cfg["data"]["loader_kwargs"])
        checks[role + "_manifest_exact"] = manifest(dataset) == recorded[recorded_role]
    best, last, optim = [torch.load(checkpoint / name, map_location="cpu", weights_only=False)
                         for name in ("best.pt", "last.pt", "optimizer.pt")]
    history = json.loads((checkpoint / "history.json").read_text())
    bridge = json.loads((checkpoint / "prompt_module_qc.json").read_text())
    reload_report = root / "refactor_v2/reports/caption_bridge_real_audit_revised_roco_alignment_v3.json"
    reload = json.loads(reload_report.read_text())
    checks.update(
        best_hash_matches_driver=sha(checkpoint / "best.pt") == status["best_checkpoint_sha256"],
        last_hash_matches_driver=sha(checkpoint / "last.pt") == status["last_checkpoint_sha256"],
        checkpoint_states_finite=all(bool(torch.isfinite(value).all()) for payload in (best, last) for value in payload["state"].values()),
        best_last_tensors_equal=best["state"].keys() == last["state"].keys() and all(torch.equal(value, last["state"][key]) for key, value in best["state"].items()),
        epoch_step_match=last["extra"]["epoch"] == optim["epoch"] == 2 and last["extra"]["global_step"] == optim["global_step"] == 2446,
        best_metric_matches_history=best["extra"]["best_metric"] == min(row["val_loss"] for row in history),
        all_bridge_qc_passed=all(row["passed"] for row in bridge.values()),
        strict_reload_and_answer_route=reload["passed"] is True and reload["reload_answer_logit_max_difference"] == 0,
        released_config_exact=sha(folder / "released_config.json") == status["config_sha256"])
    report = {"scope": "completed ROCO upstream alignment technical/artifact checks, not downstream VQA/clinical efficacy",
        "checks": checks, "passed": all(checks.values()), "state_tensors": len(best["state"]),
        "best_checkpoint": str(checkpoint / "best.pt"), "best_checkpoint_sha256": sha(checkpoint / "best.pt"),
        "last_checkpoint_sha256": sha(checkpoint / "last.pt"), "independent_reload_report_sha256": sha(reload_report),
        "full_validation_losses": [row["val_loss"] for row in history],
        "resume_note": "full run not resumed; earlier bounded CPU reference resume gate is separate evidence",
        "new_upstream_initialization_parity_for_downstream_ablations": "still required after PMC",
        "next_stage_released": False}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
