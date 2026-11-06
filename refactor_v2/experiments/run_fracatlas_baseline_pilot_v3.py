"""Bounded target-domain B pilot: fresh adapters vs ROCO; no PMC/VLP/test."""
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.context_controls import ComponentBatchSampler, load_component_lookup
from src.data import build_dataset
from src.data.sealed_manifest import load_sealed_records
from src.models.bonevqa import BoneVQAModel
from src.protocol import assert_disjoint_splits, manifest, records_of
from src.train import build_answer_vocab, build_split, collect_answers, load_init_weights
from src.utils import compute_metrics, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/fracatlas_baseline_pilot_v3"
    names = ("fracatlas_B_fresh_pilot_v3", "fracatlas_B_roco_pilot_v3")
    if folder.exists() or any((root / "refactor_v2/checkpoints" / name).exists() for name in names):
        raise RuntimeError("Pilot exists: inspect/resume, do not overwrite or duplicate")
    commands = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in commands.splitlines()):
        raise RuntimeError("Existing training process")
    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
                "HF_HOME": str(root / ".evaluation_runtime/hf_cache"), "BONEVQA_DATA": str(root / ".evaluation_runtime/data")}
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Offline CPU8 environment changed")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    base = json.loads((root / "refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
    loader = base["data"]["loader_kwargs"]
    component = base["train"]["component_batching"]
    lookup, component_hash = load_component_lookup(component["manifest"], component["seal"], root)
    parent = {role: load_sealed_records(loader["sealed_records_by_split"][role], loader["seal_path"], root)
              for role in ("train", "model_val")}
    def grouped(rows):
        result = defaultdict(list)
        for row in rows:
            result[lookup[row["image_id"]]].append(row)
        return result
    groups = {role: grouped(rows) for role, rows in parent.items()}
    assert not (set(groups["train"]) & set(groups["model_val"]))
    def rank(key):
        return hashlib.sha256(f"target_B_pilot_seed42_component{key}".encode()).hexdigest()
    strata = {0: [], 1: []}
    for key, rows in groups["train"].items():
        labels = {int(row["fractured"]) for row in rows}
        if len(labels) != 1:
            continue  # Explicitly avoid ambiguous mixed-label TRAIN components in this diagnostic only.
        strata[next(iter(labels))].append(key)
    chosen_train = set(sorted(strata[0], key=rank)[:32] + sorted(strata[1], key=rank)[:32])
    chosen_val = set(sorted(groups["model_val"], key=rank)[:32])
    assert len(chosen_train) == 64 and len(chosen_val) == 32
    selected = {"train": [row for row in parent["train"] if lookup[row["image_id"]] in chosen_train],
                "model_val": [row for row in parent["model_val"] if lookup[row["image_id"]] in chosen_val]}
    canonical = "Is there a fracture in this X-ray?"
    val_presence = [row for row in selected["model_val"] if row["question"] == canonical]
    assert {row["answer"] for row in val_presence} == {"yes", "no"}, "Fixed validation sample lacks both classes; do not resample silently"
    folder.mkdir()
    progress = folder / "progress.json"
    state = {"status": "preparing", "started_utc": now(), "driver_pid": os.getpid(),
             "scope": "bounded fold0 target-domain B pilot; development-informed grouped validation, not independent clinical evidence"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        artifacts = {Path(component["manifest"]).relative_to(root).as_posix(): component_hash}
        for role, rows in selected.items():
            path = folder / (role + ".json")
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
            artifacts[path.relative_to(root).as_posix()] = sha(path)
        seal_path = folder / "data_seal.json"
        seal = {"status": "sealed_development_data", "scope": "bounded_fold0_grouped_target_B_pilot",
                "parent_seal_sha256": sha(loader["seal_path"]), "artifacts": artifacts,
                "parent_source_sha256": {role: sha(loader["sealed_records_by_split"][role]) for role in selected},
                "selected_components": {"train": sorted(chosen_train), "model_val": sorted(chosen_val)},
                "selection": "TRAIN32positive+32negative homogeneous components, deterministic SHA256 rank; validation first32 ranked components without label balancing; retain all selected component QA",
                "original_data_unchanged": True, "test_or_calibration_read": False,
                "QA_counts": {role: len(rows) for role, rows in selected.items()}, "long_training_released": False}
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
        cfg = copy.deepcopy(base)
        cfg["recipe_status"] = "released_bounded_target_B_pilot_only"
        cfg["data"]["sealed_answer_vocab"] = None
        cfg["data"]["loader_kwargs"].update(seal_path=str(seal_path), sealed_records_by_split={"train": str(folder / "train.json"),
            "model_val": str(folder / "model_val.json"), "calibration": None, "test": None})
        cfg["train"].update(epochs=4, log_every=1)
        cfg["train"]["component_batching"]["seal"] = str(seal_path)
        ds = {role: build_split(build_dataset, cfg["data"]["dataset"], role, None, True, cfg["data"]["loader_kwargs"]) for role in selected}
        assert_disjoint_splits(**ds)
        vocab = build_answer_vocab(collect_answers(ds["train"]), cfg["data"]["answer_vocab_size"])
        actual_train = records_of(ds["train"])
        sampler = ComponentBatchSampler([lookup[row["image_id"]] for row in actual_train], 4, 42)
        for epoch in range(4):
            sampler.set_epoch(epoch)
            assert sorted(i for batch in sampler for i in batch) == list(range(len(ds["train"])))
        presence = [row for row in actual_train if row["question"] == canonical]
        counts = Counter(row["answer"] for row in presence)
        positive_weight = counts["no"] / counts["yes"]
        by_question = defaultdict(Counter)
        for row in actual_train:
            by_question[row["question"]][row["answer"]] += 1
        majority = {question: sorted(values, key=lambda answer: (-values[answer], answer))[0] for question, values in by_question.items()}
        global_counts = Counter(row["answer"] for row in actual_train)
        fallback = sorted(global_counts, key=lambda answer: (-global_counts[answer], answer))[0]
        prior = counts["yes"] / sum(counts.values())
        controls = []
        for row in records_of(ds["model_val"]):
            item = {"question": row["question"], "gt": row["answer"], "pred": majority.get(row["question"], fallback),
                    "answer_type": row["answer_type"], "image_id": row["image_id"]}
            if row["question"] == canonical:
                item["p_yes"] = prior
            controls.append(item)
        (folder / "question_frequency_control.json").write_text(json.dumps({"scope": "TRAIN-only majority per exact question; no image input",
            "metrics": compute_metrics(controls), "records": controls, "constant_fracture_score_from_TRAIN": prior,
            "unseen_question_fallback": "global TRAIN-majority answer, never validation labels",
            "warning": "Template/modality/class-frequency scores must not be described as visual reasoning"}, indent=2) + "\n")
        upstream = json.loads((root / "refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json").read_text())
        assert upstream["passed"] and sha(upstream["best_checkpoint"]) == upstream["best_checkpoint_sha256"]
        configs, init_hashes, heads = [], [], None
        for name, use_roco in zip(names, (False, True)):
            set_seed(42)
            model_cfg = copy.deepcopy(cfg["model"])
            model_cfg.update(module_init_seed=1771, fracture_positive_weight=positive_weight)
            model = BoneVQAModel(model_cfg, vocab, device="cpu")
            if heads is None:
                heads = {key: value.clone() for key, value in model.trainable_state_dict().items() if key.startswith(("closed_head.", "fracture_head."))}
            if use_roco:
                load_init_weights(model, upstream["best_checkpoint"], cfg["checkpoint_allow_mismatch_prefixes"], False)
            # Both downstream task heads are new and exactly shared, not transferred from caption training.
            model.load_state_dict(heads, strict=False)
            hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in model.trainable_state_dict().items()}
            init_hashes.append(hashes)
            init_path = folder / (name + "_initialization.pt")
            model.save(str(init_path), {"scope": "bounded_target_B_initialization_only", "ROCO_used": use_roco})
            arm = copy.deepcopy(cfg)
            arm.update(run_name=name, init_from=str(init_path), allow_random_init=False, figure_dir=str(folder / name / "figures"))
            path = folder / (name + ".json")
            path.write_text(json.dumps(arm, indent=2) + "\n")
            configs.append((arm, path))
            del model
            gc.collect()
        assert all(init_hashes[0][key] == init_hashes[1][key] for key in heads)
        assert set(init_hashes[0]) == set(init_hashes[1]) and init_hashes[0] != init_hashes[1]
        gate = {"passed": True, "scope": state["scope"], "actual_manifests": {role: manifest(data) for role, data in ds.items()},
                "new_closed_and_fracture_heads_byte_identical": True, "head_tensors": len(heads),
                "initial_tensor_hashes": dict(zip(names, init_hashes)), "ROCO_checkpoint_sha256": upstream["best_checkpoint_sha256"],
                "TRAIN_fracture_counts": dict(counts), "TRAIN_positive_weight": positive_weight,
                "validation_fracture_counts": dict(Counter(row["answer"] for row in val_presence)),
                "common_sampler_all4epochs_no_row_loss": True, "sampler_binding": sampler.binding(),
                "epochs": 4, "only_different_treatment": "ROCO-updated adapters versus fresh adapters on same cached pretrained encoders/LLM; downstream task heads identical",
                "pretraining_compute_is_NOT_matched": True, "downstream_data_optimization_budget_matched": True,
                "config_sha256": {arm["run_name"]: sha(path) for arm, path in configs}, "production_input": "original image + raw question only",
                "PMC_used": False, "test_or_calibration_used": False, "long_training_released": False}
        (folder / "execution_release.json").write_text(json.dumps(gate, indent=2) + "\n")
        for arm, path in configs:
            started = time.perf_counter()
            with (folder / (arm["run_name"] + ".log")).open("w") as log:
                child = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(path), "--device", "cpu"],
                                         env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)])
                state.update(status="running", arm=arm["run_name"], training_pid=child.pid, arm_started_utc=now())
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{arm['run_name']} failed exit{code}; inspect preserved log, no retry")
            checkpoint = root / arm["checkpoint_dir"] / arm["run_name"]
            payload = torch.load(checkpoint / "last.pt", map_location="cpu", weights_only=False)
            history = json.loads((checkpoint / "history.json").read_text())
            assert len(history) == 4 and all(bool(torch.isfinite(value).all()) for value in payload["state"].values())
            result = {"complete": True, "wall_seconds": time.perf_counter() - started, "history": history,
                      "optimizer_steps": payload["extra"]["global_step"], "best_sha256": sha(checkpoint / "best.pt"),
                      "last_sha256": sha(checkpoint / "last.pt"), "quality_gate_passed": False}
            (folder / (arm["run_name"] + "_completion.json")).write_text(json.dumps(result, indent=2) + "\n")
            del payload
            gc.collect()
            state.setdefault("completed_arms", []).append(arm["run_name"])
            save()
        state.update(status="both_arms_complete_pending_independent_evaluation", finished_utc=now(), long_training_released=False)
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=now())
        save()
        raise


if __name__ == "__main__":
    main()
