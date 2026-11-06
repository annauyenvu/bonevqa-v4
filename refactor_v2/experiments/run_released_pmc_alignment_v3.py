"""Release PMC stage ONLY from verified fresh ROCO; preserve artifacts and never chain VQA."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from src.data.sealed_manifest import load_sealed_records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_alignment_execution_v3"
    checkpoint = root / "refactor_v2/checkpoints/revised_pmcvqa_alignment_v3"
    if folder.exists() or checkpoint.exists():
        raise RuntimeError("Existing PMC run requires inspection/resume, not duplicate launch")
    processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in processes.splitlines()):
        raise RuntimeError("Matching training alive; do not duplicate/contend")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    pilot_path = root / "refactor_v2/reports/pmc_cpu_timing_v3/timing.json"
    bridge_path = root / "refactor_v2/reports/pmc_cpu_timing_v3/bridge_verification.json"
    verified_path = root / "refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json"
    pilot, bridge, verified = [json.loads(path.read_text()) for path in (pilot_path, bridge_path, verified_path)]
    if not all(report.get("passed") is True for report in (pilot, bridge, verified)):
        raise RuntimeError("Upstream/pilot/bridge gate not passed")
    if pilot["checkpoint_sha256"] != bridge["checkpoint_sha256"]:
        raise RuntimeError("Pilot verification checkpoint mismatch")
    candidate = root / "refactor_v2/configs/revised_sealed_candidates_v3/pmc_alignment.json"
    if sha(candidate) != pilot["candidate_sha256"]:
        raise RuntimeError("PMC candidate changed since pilot")
    cfg = json.loads(candidate.read_text())
    if Path(cfg["init_from"]).resolve() != Path(verified["best_checkpoint"]).resolve() or sha(cfg["init_from"]) != verified["best_checkpoint_sha256"]:
        raise RuntimeError("PMC must use the independently verified fresh ROCO, not pilot/old PMC")
    expected_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
        "HF_HOME": str(root / ".evaluation_runtime/hf_cache"), "BONEVQA_DATA": str(root / ".evaluation_runtime/data")}
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise RuntimeError("Pinned offline CPU runtime changed")
    loader = cfg["data"]["loader_kwargs"]
    counts = {role: len(load_sealed_records(loader["sealed_records_by_split"][role], loader["seal_path"], root))
              for role in ("train", "model_val")}
    if counts != {"train": 28490, "model_val": 1500} or loader["routing_policy"] != "generative":
        raise RuntimeError("Sealed PMC source/routing mismatch")
    if cfg["data"]["max_samples"] is not None or cfg["data"]["val_max_samples"] is not None or cfg["train"]["eval_max_samples"] is not None:
        raise RuntimeError("Full PMC stage cannot silently cap train/model validation")
    folder.mkdir()
    cfg["recipe_status"] = "released_pmc_only_cpu_execution_v3"
    config = folder / "released_config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    release = {"scope": "PMC generative alignment ONLY; no independent PMC test or downstream VQA release",
        "released_utc": now(), "config_sha256": sha(config), "candidate_sha256": sha(candidate),
        "pilot_report_sha256": sha(pilot_path), "pilot_bridge_sha256": sha(bridge_path),
        "fresh_roco_checkpoint_sha256": verified["best_checkpoint_sha256"], "counts": counts,
        "runtime": "CPU8 threads float32 no checkpointing epoch-seeded", "epochs": 2,
        "historical_pmc_checkpoint_used": False, "pilot_checkpoint_used_for_full_initialization": False,
        "test_used_for_training_or_selection": False, "other_stage_released": False}
    (folder / "execution_release.json").write_text(json.dumps(release, indent=2) + "\n")
    status = {"stage": "pmc_generative_alignment", "status": "starting", "started_utc": now(),
        "driver_pid": os.getpid(), "config_sha256": sha(config), "checkpoint_dir": str(checkpoint)}
    progress = folder / "progress.json"
    def save():
        progress.write_text(json.dumps(status, indent=2) + "\n")
    save()
    started = time.perf_counter()
    with (folder / "train.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(config), "--device", "cpu"],
                                   stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        guard = subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)])
        status.update(status="running", training_pid=process.pid, idle_guard_pid=guard.pid)
        save()
        result = process.wait()
    status.update(exit_code=result, finished_utc=now(), wall_seconds=time.perf_counter() - started)
    if result:
        status.update(status="failed_preserved", failure="Inspect train.log; never auto-restart/duplicate")
        save()
        raise SystemExit(result)
    try:
        import torch
        payload = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
        history = json.loads((checkpoint / "history.json").read_text())
        if len(history) != 2 or not all(bool(torch.isfinite(value).all()) for value in payload["state"].values()):
            raise RuntimeError("PMC final history/checkpoint finite QC failed")
        status.update(status="training_complete_pending_independent_checkpoint_evaluation", epochs_completed=2,
            best_checkpoint_sha256=sha(checkpoint / "best.pt"), last_checkpoint_sha256=sha(checkpoint / "last.pt"))
        save()
    except Exception as error:
        status.update(status="verification_failed_preserved", failure=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
