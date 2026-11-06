"""Bounded throughput pilot from fresh bases; not an alignment benchmark."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from src.data.sealed_manifest import load_sealed_records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/roco_cpu_timing_train_only_v3"
    checkpoint = root / "refactor_v2/checkpoints/roco_cpu_timing_train_only_v3"
    if folder.exists() or checkpoint.exists():
        raise RuntimeError("Existing timing artifacts must be inspected, never overwritten")
    # Verify the frozen technical source/config snapshot before deriving the
    # bounded pilot. This new experiment does not mutate its190 artifacts.
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    cfg_path = root / "refactor_v2/configs/revised_sealed_candidates_v3/roco_alignment.json"
    cfg = json.loads(cfg_path.read_text())
    loader = cfg["data"]["loader_kwargs"]
    source = Path(loader["sealed_records_by_split"]["train"])
    parent_seal = Path(loader["seal_path"])
    rows = load_sealed_records(source, parent_seal, root)
    shuffled = list(range(len(rows)))
    random.Random(42).shuffle(shuffled)
    selected, seen = [], set()
    for index in shuffled:
        row = rows[index]
        if row["image_id"] not in seen:
            seen.add(row["image_id"])
            selected.append(row)
        if len(selected) == 40:
            break
    if len(selected) != 40 or any(not str(row.get("caption", "")).strip() for row in selected):
        raise RuntimeError("Pilot requires40 distinct existing OPEN TRAIN caption rows")
    folder.mkdir()
    artifacts, sources = {}, {}
    for role, subset in (("train", selected[:32]), ("model_val", selected[32:])):
        path = folder / (role + ".json")
        path.write_text(json.dumps(subset, indent=2) + "\n")
        artifacts[path.relative_to(root).as_posix()] = sha(path)
        sources[role] = str(path)
    seal = folder / "data_seal.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "scope": "technical_smoke_train_only",
        "parent_manifest": str(source), "parent_manifest_sha256": sha(source),
        "parent_seal_sha256": sha(parent_seal), "artifacts": artifacts}, indent=2))
    cfg.update(run_name=checkpoint.name, recipe_status="bounded_train_only_timing_pilot")
    cfg["data"].update(max_samples=None, val_max_samples=None)
    loader.update(seal_path=str(seal), sealed_records_by_split=sources)
    cfg["train"].update(epochs=2, batch_size=4, grad_accum=4, log_every=1)
    config = folder / "pilot.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    start = time.perf_counter()
    with (folder / "train.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "src.train", "--config", str(config), "--device", "cpu"],
                       env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter() - start
    history = json.loads((checkpoint / "history.json").read_text())
    qc = json.loads((checkpoint / "prompt_module_qc.json").read_text())
    expected_steps = 2 * math.ceil(math.ceil(32 / 4) / 4)
    import torch
    payload = torch.load(checkpoint / "last.pt", map_location="cpu", weights_only=False)
    epochs = [row["time_s"] for row in history]
    if not all(math.isfinite(row["train_loss"]) for row in history):
        raise RuntimeError("Nonfinite pilot loss")
    report = {"scope": "32 TRAIN captions +8 disjoint-image TRAIN captions for internal checking;2epochs; no quality claim",
        "reference_candidate_sha256": sha(cfg_path), "derived_config_sha256": sha(config),
        "wall_seconds_including_startup_validation_checkpointing": seconds,
        "train_epoch_seconds": epochs, "train_rows_per_epoch": 32,
        "full_train_rows": len(rows), "full_train_only_2epoch_projection_seconds": 2 * sum(epochs) / len(epochs) * len(rows) / 32,
        "projection_limitations": "Tiny pilot; excludes full validation/checkpoint overhead and thermal/I/O/sequence-length variation; not total project ETA",
        "optimizer_steps": payload["extra"]["global_step"], "expected_steps": expected_steps,
        "bridge_qc": qc, "checkpoint_sha256": sha(checkpoint / "last.pt"),
        "passed": payload["extra"]["global_step"] == expected_steps and len(history) == 2,
        "long_training_authorized": False}
    (folder / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
