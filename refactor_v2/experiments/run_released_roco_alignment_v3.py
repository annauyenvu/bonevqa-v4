"""One-stage CPU ROCO execution release after bounded gates; never chains ablations."""
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
    folder = root / "refactor_v2/reports/roco_alignment_execution_v3"
    checkpoint = root / "refactor_v2/checkpoints/revised_roco_alignment_v3"
    if folder.exists() or checkpoint.exists():
        raise RuntimeError("Existing full alignment artifacts require inspection/resume, not a duplicate launch")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    pilot_path = root / "refactor_v2/reports/roco_cpu_timing_train_only_v3/timing.json"
    pilot = json.loads(pilot_path.read_text())
    if pilot.get("passed") is not True or not all(row.get("passed") for row in pilot["bridge_qc"].values()):
        raise RuntimeError("ROCO bounded timing/bridge gate not passed")
    common = json.loads((root / "refactor_v2/reports/common_initialization_seed42_fold0_v3.json").read_text())
    cache_path = root / common["base_cache_audit"]
    if sha(cache_path) != common["base_cache_audit_sha256"]:
        raise RuntimeError("Base cache report changed")
    cache = json.loads(cache_path.read_text())
    for path, expected in cache["file_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Offline base cache changed: " + path)
    candidate = root / "refactor_v2/configs/revised_sealed_candidates_v3/roco_alignment.json"
    cfg = json.loads(candidate.read_text())
    if sha(candidate) != pilot["reference_candidate_sha256"]:
        raise RuntimeError("Candidate differs from bounded timing parent")
    required_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / ".evaluation_runtime/hf_cache"), "BONEVQA_DATA": str(root / ".evaluation_runtime/data"),
        "OMP_NUM_THREADS": "8"}
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        raise RuntimeError("Pinned offline environment mismatch")
    loader = cfg["data"]["loader_kwargs"]
    counts = {role: len(load_sealed_records(path, loader["seal_path"], root))
              for role, path in loader["sealed_records_by_split"].items() if path}
    if counts != {"train": 19556, "model_val": 9903}:
        raise RuntimeError("Unexpected ROCO sealed role counts")
    if cfg["init_from"] is not None or cfg["model"]["stage1_caption_loss_weight"] != 1.0:
        raise RuntimeError("Fresh ROCO base/caption protocol mismatch")
    folder.mkdir()
    cfg["recipe_status"] = "released_roco_only_cpu_execution_v3"
    config = folder / "released_config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    release = {"scope": "ROCO caption alignment ONLY; not VQA/module/clinical release",
        "released_utc": now(), "config_sha256": sha(config), "candidate_sha256": sha(candidate),
        "pilot_report_sha256": sha(pilot_path), "base_cache_audit_sha256": sha(cache_path),
        "technical_reference_seal_sha256": sha(root / "refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json"),
        "counts": counts, "device": "cpu", "epochs": 2,
        "fresh_base_initialization": True, "historical_pmc_checkpoint_used": False,
        "other_stage_release": False, "candidate_configs_unchanged": True}
    (folder / "execution_release.json").write_text(json.dumps(release, indent=2) + "\n")
    status = {"stage": "roco_alignment", "status": "starting", "started_utc": now(),
              "config_sha256": sha(config), "driver_pid": os.getpid(), "checkpoint_dir": str(checkpoint)}
    progress = folder / "progress.json"
    def save():
        progress.write_text(json.dumps(status, indent=2) + "\n")
    save()
    started = time.perf_counter()
    with (folder / "train.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(config), "--device", "cpu"],
                                   stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        status.update(status="running", training_pid=process.pid)
        save()
        result = process.wait()
    status.update(exit_code=result, finished_utc=now(), wall_seconds=time.perf_counter() - started)
    if result:
        status.update(status="failed_preserved", failure="Inspect train.log; do not auto-restart or duplicate")
        save()
        raise SystemExit(result)
    try:
        import torch
        payload = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
        history = json.loads((checkpoint / "history.json").read_text())
        qc = json.loads((checkpoint / "prompt_module_qc.json").read_text())
        if (len(history) != 2 or not all(torch.isfinite(value).all() for value in payload["state"].values())
                or not all(item["passed"] for item in qc.values())):
            raise RuntimeError("ROCO final finite/history/bridge QC failed")
        status.update(status="training_complete_pending_independent_checkpoint_evaluation",
                      best_checkpoint_sha256=sha(checkpoint / "best.pt"),
                      last_checkpoint_sha256=sha(checkpoint / "last.pt"), epochs_completed=len(history))
        save()
    except Exception as error:
        status.update(status="verification_failed_preserved", failure=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
