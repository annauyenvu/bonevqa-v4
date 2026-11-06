"""Bounded PMC generative pilot from independently verified fresh ROCO."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from src.data import build_dataset
from src.protocol import records_of
from src.train import build_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_cpu_timing_v3"
    checkpoint = root / "refactor_v2/checkpoints/pmc_cpu_timing_v3"
    if folder.exists() or checkpoint.exists():
        raise RuntimeError("Existing PMC pilot must not be overwritten")
    processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in processes.splitlines()):
        raise RuntimeError("Existing training process alive; do not contend")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    verified_path = root / "refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json"
    verified = json.loads(verified_path.read_text())
    if verified.get("passed") is not True or sha(verified["best_checkpoint"]) != verified["best_checkpoint_sha256"]:
        raise RuntimeError("Fresh upstream ROCO has not passed verification")
    source = root / "refactor_v2/configs/revised_sealed_candidates_v3/pmc_alignment.json"
    cfg = json.loads(source.read_text())
    if Path(cfg["init_from"]).resolve() != Path(verified["best_checkpoint"]).resolve():
        raise RuntimeError("Wrong upstream initialization")
    cfg.update(run_name=checkpoint.name, recipe_status="bounded_pmc_generative_timing_pilot")
    cfg["data"].update(max_samples=32, val_max_samples=8)
    cfg["train"].update(epochs=2, batch_size=4, grad_accum=4, log_every=1)
    loader = cfg["data"]["loader_kwargs"]
    if loader["routing_policy"] != "generative":
        raise RuntimeError("PMC must not derive routing from split answer frequencies")
    selected = {}
    for role, limit in (("train", 32), ("model_val", 8)):
        dataset = build_split(build_dataset, "pmc_vqa", role, limit, True, loader)
        rows = records_of(dataset)
        if not rows or any(row["answer_type"] != "OPEN" for row in rows):
            raise RuntimeError("Sealed PMC generative pilot must route every QA to token CE")
        selected[role] = len(rows)
    folder.mkdir()
    config = folder / "pilot.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    started = time.perf_counter()
    with (folder / "train.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "src.train", "--config", str(config), "--device", "cpu"],
                       env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch
    payload = torch.load(checkpoint / "last.pt", map_location="cpu", weights_only=False)
    history = json.loads((checkpoint / "history.json").read_text())
    manifest = json.loads((checkpoint / "data_manifest.json").read_text())
    micro_batches = selected["train"] // 4 if selected["train"] > 4 else math.ceil(selected["train"] / 4)
    expected_steps = 2 * math.ceil(micro_batches / 4)
    report = {"scope": "bounded PMC sealed grouped TRAIN/model-validation generative execution/timing only; no accuracy or independent-test evidence",
        "selected_rows": selected, "selected_data_manifest": manifest,
        "fresh_roco_checkpoint_sha256": verified["best_checkpoint_sha256"],
        "candidate_sha256": sha(source), "pilot_config_sha256": sha(config),
        "wall_seconds_including_validation_startup_saving": time.perf_counter() - started,
        "train_epoch_seconds": [row["time_s"] for row in history],
        "full_validation_losses_in_pilot_subset": [row["val_loss"] for row in history],
        "optimizer_steps": payload["extra"]["global_step"], "expected_steps": expected_steps,
        "checkpoint_sha256": sha(checkpoint / "last.pt"),
        "all_checkpoint_weights_finite": all(bool(torch.isfinite(value).all()) for value in payload["state"].values()),
        "passed": len(history) == 2 and payload["extra"]["global_step"] == expected_steps
            and all(math.isfinite(row["train_loss_lm"]) and row["train_loss_lm"] > 0 for row in history),
        "full_pmc_run_released": False,
        "limitations": ["small subset does not establish downstream answer quality", "pilot checkpoint must NOT initialize full PMC or final ablations",
                        "microbenchmark does not certify GPU/MPS or changed thread runtime"]}
    report["passed"] = report["passed"] and report["all_checkpoint_weights_finite"]
    (folder / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_data_manifest"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
