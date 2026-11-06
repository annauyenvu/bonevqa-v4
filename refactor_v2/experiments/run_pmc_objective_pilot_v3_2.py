"""Sealed bounded two-arm PMC objective pilot; never releases long training."""
import copy
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.data import build_dataset
from src.data.sealed_manifest import load_sealed_records
from src.models.bonevqa import BoneVQAModel
from src.protocol import assert_disjoint_splits, manifest
from src.train import build_split, build_answer_vocab, collect_answers, load_init_weights
from src.utils import set_seed
from experiments.prepare_pmc_raw_question_policy_v3_1 import classify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def article(row):
    match = re.search(r"PMC\d+", str(row["image"]), re.I)
    if not match:
        raise RuntimeError("Missing PMC article identity")
    return match.group().upper()


def select(rows, count):
    chosen, articles = [], set()
    for row in rows:
        key = article(row)
        if key not in articles:
            articles.add(key)
            chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError("Insufficient eligible distinct articles")
    return chosen


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_objective_pilot_v3_2"
    names = ("pmc_objective_current_v3_2", "pmc_objective_answer_control_v3_2")
    if folder.exists() or any((root / "refactor_v2/checkpoints" / name).exists() for name in names):
        raise RuntimeError("Existing pilot evidence: inspect/resume, never duplicate or overwrite")
    commands = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in commands.splitlines()):
        raise RuntimeError("Training alive: do not contend")
    expected_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
                    "HF_HOME": str(root / ".evaluation_runtime/hf_cache"), "BONEVQA_DATA": str(root / ".evaluation_runtime/data")}
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise RuntimeError("Offline CPU8 reference environment changed")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    policy_folder = root / "refactor_v2/reports/pmc_raw_question_policy_v3_2"
    verification = json.loads((policy_folder / "independent_loader_verification.json").read_text())
    if not verification["passed"] or sha(root / "refactor_v2/experiments/prepare_pmc_raw_question_policy_v3_1.py") != verification["implementation_sha256"]:
        raise RuntimeError("TRAIN-derived eligibility implementation changed")
    upstream = json.loads((root / "refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json").read_text())
    if not upstream["passed"] or sha(upstream["best_checkpoint"]) != upstream["best_checkpoint_sha256"]:
        raise RuntimeError("Fresh ROCO provenance failed")
    base = json.loads((root / "refactor_v2/reports/pmc_alignment_execution_v3/released_config.json").read_text())
    old_loader = base["data"]["loader_kwargs"]
    train = load_sealed_records(old_loader["sealed_records_by_split"]["train"], old_loader["seal_path"], root)
    val = load_sealed_records(old_loader["sealed_records_by_split"]["model_val"], old_loader["seal_path"], root)
    if {article(row) for row in train} & {article(row) for row in val}:
        raise RuntimeError("Original train/development-validation article overlap")
    eligible_train = [row for row in train if not classify(row)]
    candidate = json.loads((policy_folder / "train_candidate.json").read_text())
    if eligible_train != candidate:
        raise RuntimeError("TRAIN policy replay differs from verified candidate")
    eligible_val = [row for row in val if not classify(row)]
    selected = {"train": select(eligible_train, 256), "model_val": select(eligible_val, 128)}
    folder.mkdir()
    progress = folder / "progress.json"
    status = {"status": "preparing", "started_utc": now(), "driver_pid": os.getpid(),
              "scope": "bounded paired objective pilot; inspected model-validation only; no test or long-stage release"}
    def save():
        progress.write_text(json.dumps(status, indent=2) + "\n")
    save()
    try:
        artifacts = {}
        for role, rows in selected.items():
            path = folder / (role + ".json")
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
            artifacts[path.relative_to(root).as_posix()] = sha(path)
        seal = {"status": "sealed_development_data", "scope": "bounded_paired_development_objective_pilot",
                "parent_seal_sha256": sha(old_loader["seal_path"]), "eligibility_policy_sha256": verification["policy_sha256"],
                "eligibility_implementation_sha256": verification["implementation_sha256"], "artifacts": artifacts,
                "source_sha256": {role: sha(old_loader["sealed_records_by_split"][role]) for role in selected},
                "original_counts": {"train": len(train), "model_val": len(val)},
                "eligible_counts": {"train": len(eligible_train), "model_val": len(eligible_val)},
                "selected_counts": {role: len(rows) for role, rows in selected.items()},
                "selection": "first source-order QA per distinct article, identical in both arms; not random representative performance sample",
                "test_used": False, "long_training_released": False}
        seal_path = folder / "data_seal.json"
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
        configs = []
        for name, lam in zip(names, (0.1, 0.0)):
            cfg = copy.deepcopy(base)
            cfg.update(run_name=name, recipe_status="released_bounded_paired_objective_pilot_v3_2", init_from=upstream["best_checkpoint"])
            cfg["train"].update(epochs=4, log_every=1)
            cfg["figure_dir"] = str(folder / name / "figures")
            loader = cfg["data"]["loader_kwargs"]
            loader.update(seal_path=str(seal_path), sealed_records_by_split={"train": str(folder / "train.json"),
                          "model_val": str(folder / "model_val.json"), "calibration": None, "test": None})
            cfg["model"]["lam"] = lam
            path = folder / (name + ".json")
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            configs.append((cfg, path))
        loader = configs[0][0]["data"]["loader_kwargs"]
        ds = {role: build_split(build_dataset, "pmc_vqa", role, None, True, loader) for role in selected}
        assert_disjoint_splits(**ds)
        if any(len(ds[role]) != len(selected[role]) for role in selected):
            raise RuntimeError("Actual loader selected coverage changed")
        vocab = build_answer_vocab(collect_answers(ds["train"]), base["data"]["answer_vocab_size"])
        signatures = []
        for cfg, _ in configs:
            set_seed(42)
            model_cfg = copy.deepcopy(cfg["model"])
            model_cfg["module_init_seed"] = 42 + int(model_cfg.get("module_init_seed_offset", 1729))
            model = BoneVQAModel(model_cfg, vocab, device="cpu")
            load_init_weights(model, cfg["init_from"], cfg["checkpoint_allow_mismatch_prefixes"], False)
            signatures.append({name: hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
                               for name, param in model.named_parameters() if param.requires_grad})
            del model
            gc.collect()
        if signatures[0] != signatures[1]:
            raise RuntimeError("Paired arm trainable initialization is not byte-identical")
        gate = {"passed": True, "trainable_initial_tensor_count": len(signatures[0]), "trainable_initial_hashes": signatures[0],
                "initialization_byte_identical": True, "fresh_roco_sha256": upstream["best_checkpoint_sha256"],
                "actual_data_manifests": {role: manifest(data) for role, data in ds.items()},
                "selected_article_disjoint": True, "all_parent_train_val_article_disjoint": True,
                "config_sha256": {cfg["run_name"]: sha(path) for cfg, path in configs},
                "policy_verification_sha256": sha(policy_folder / "independent_loader_verification.json"),
                "only_config_difference_beyond_output_paths": "model.lam0.1_vs0.0",
                "long_training_released": False}
        (folder / "execution_release.json").write_text(json.dumps(gate, indent=2) + "\n")
        status.update(status="paired_initialization_verified", trainable_initial_tensors=len(signatures[0]))
        save()
        for cfg, path in configs:
            checkpoint = root / cfg["checkpoint_dir"] / cfg["run_name"]
            started = time.perf_counter()
            with (folder / (cfg["run_name"] + ".log")).open("w") as log:
                child = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(path), "--device", "cpu"],
                                         env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)])
                status.update(status="running", arm=cfg["run_name"], training_pid=child.pid, arm_started_utc=now())
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{cfg['run_name']} exit{code}; preserve log, no automatic retry")
            payload = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
            optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
            history = json.loads((checkpoint / "history.json").read_text())
            if len(history) != 4 or not all(bool(torch.isfinite(value).all()) for value in payload["state"].values()):
                raise RuntimeError("Incomplete/nonfinite pilot artifacts")
            expected_steps = 4 * math.ceil((len(ds["train"]) // 4) / 4)
            last = torch.load(checkpoint / "last.pt", map_location="cpu", weights_only=False)
            if last["extra"]["global_step"] != expected_steps:
                raise RuntimeError("Paired optimizer budget mismatch")
            result = {"wall_seconds": time.perf_counter() - started, "history": history, "expected_optimizer_steps": expected_steps,
                      "best_sha256": sha(checkpoint / "best.pt"), "last_sha256": sha(checkpoint / "last.pt"),
                      "complete": True, "quality_gate_passed": False}
            (folder / (cfg["run_name"] + "_completion.json")).write_text(json.dumps(result, indent=2) + "\n")
            del payload, optimizer, last
            gc.collect()
            status.setdefault("completed_arms", []).append(cfg["run_name"])
            save()
        status.update(status="both_arms_complete_pending_independent_evaluation", finished_utc=now(), long_training_released=False)
        save()
    except Exception as error:
        status.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=now())
        save()
        raise


if __name__ == "__main__":
    main()
