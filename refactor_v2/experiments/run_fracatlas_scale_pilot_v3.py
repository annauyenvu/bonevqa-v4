"""Release ONLY the predeclared1173-QA/296-step paired B pilot; preserve evidence."""
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
from src.models.bonevqa import BoneVQAModel
from src.protocol import assert_disjoint_splits, manifest, records_of
from src.train import build_answer_vocab, build_split, collect_answers, load_init_weights
from src.utils import compute_metrics, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/fracatlas_scale_pilot_v3"
    names = ("fracatlas_B_fresh_scale_pilot_v3", "fracatlas_B_roco_scale_pilot_v3")
    progress = folder / "progress.json"
    if progress.exists() or any((root / "refactor_v2/checkpoints" / name).exists() for name in names):
        raise RuntimeError("Existing run: inspect/resume, never duplicate or overwrite")
    commands = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in commands.splitlines()):
        raise RuntimeError("Training process already active")
    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
                "HF_HOME": str(root / ".evaluation_runtime/hf_cache"), "BONEVQA_DATA": str(root / ".evaluation_runtime/data")}
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Pinned offline CPU8 runtime changed")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    protocol = json.loads((folder / "protocol.json").read_text())
    assert protocol["status"] == "data_and_sampler_verified_NOT_execution_release"
    state = {"status": "verifying_execution_gates", "driver_pid": os.getpid(), "started_utc": now(),
             "scope": "bounded target B data-expansion pilot; no PMC, VLP, calibration/test or long scientific release"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        base = json.loads((root / "refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
        base["recipe_status"] = "released_bounded_target_B_scale_pilot_only"
        base["data"]["sealed_answer_vocab"] = None
        base["data"]["loader_kwargs"].update(seal_path=str(folder / "data_seal.json"),
            sealed_records_by_split={"train": str(folder / "train.json"), "model_val": str(folder / "model_val.json"), "calibration": None, "test": None})
        base["train"].update(epochs=4, log_every=1)
        base["train"]["component_batching"]["seal"] = str(folder / "data_seal.json")
        loader = base["data"]["loader_kwargs"]
        ds = {role: build_split(build_dataset, base["data"]["dataset"], role, None, True, loader) for role in ("train", "model_val")}
        assert_disjoint_splits(**ds)
        assert {role: manifest(data) for role, data in ds.items()} == protocol["actual_manifests"]
        batch_cfg = base["train"]["component_batching"]
        lookup, _ = load_component_lookup(batch_cfg["manifest"], batch_cfg["seal"], root)
        sampler = ComponentBatchSampler([lookup[row["image_id"]] for row in records_of(ds["train"])], 4, 42)
        for epoch in range(4):
            sampler.set_epoch(epoch)
            assert sorted(i for batch in sampler for i in batch) == list(range(1173))
            assert len(sampler) == 294
        assert protocol["maximum_optimizer_steps_per_arm"] == 296
        actual = records_of(ds["train"])
        canonical = "Is there a fracture in this X-ray?"
        counts = Counter(row["answer"] for row in actual if row["question"] == canonical)
        assert counts["yes"] and counts["no"]
        positive_weight = counts["no"] / counts["yes"]
        vocab = build_answer_vocab(collect_answers(ds["train"]), base["data"]["answer_vocab_size"])
        (folder / "TRAIN_answer_vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
        by_question = defaultdict(Counter)
        all_answers = Counter(row["answer"] for row in actual)
        fallback = sorted(all_answers, key=lambda answer: (-all_answers[answer], answer))[0]
        for row in actual:
            by_question[row["question"]][row["answer"]] += 1
        majority = {question: sorted(values, key=lambda answer: (-values[answer], answer))[0] for question, values in by_question.items()}
        records = [{"image_id": row["image_id"], "question": row["question"], "gt": row["answer"], "answer_type": row["answer_type"],
                    "pred": majority.get(row["question"], fallback)} for row in records_of(ds["model_val"])]
        controls = {}
        for label in ("no", "yes"):
            copied = copy.deepcopy(records)
            for row in copied:
                if row["question"] == canonical:
                    row.update(pred=label, p_yes=float(label == "yes"))
            controls["always_" + label + "_fracture_other_TRAIN_majority"] = {"metrics": compute_metrics(copied, fracture_threshold=0.5), "records": copied}
        controls["TRAIN_question_majority"] = {"metrics": compute_metrics(records), "records": records}
        (folder / "frozen_controls.json").write_text(json.dumps({"scope": "No-image controls; answer rules from TRAIN only, no validation fitting", "conditions": controls}, indent=2) + "\n")
        upstream = json.loads((root / "refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json").read_text())
        assert upstream["passed"] and sha(upstream["best_checkpoint"]) == upstream["best_checkpoint_sha256"]
        configs, init_hashes, heads = [], {}, None
        for name, roco in zip(names, (False, True)):
            set_seed(42)
            model_cfg = copy.deepcopy(base["model"])
            model_cfg.update(module_init_seed=1771, fracture_positive_weight=positive_weight)
            model = BoneVQAModel(model_cfg, vocab, device="cpu")
            if heads is None:
                heads = {key: value.clone() for key, value in model.trainable_state_dict().items() if key.startswith(("closed_head.", "fracture_head."))}
            if roco:
                load_init_weights(model, upstream["best_checkpoint"], base["checkpoint_allow_mismatch_prefixes"], False)
            model.load_state_dict(heads, strict=False)
            initial = model.trainable_state_dict()
            assert finite_tree(initial)
            init_hashes[name] = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in initial.items()}
            init_path = folder / (name + "_initialization.pt")
            model.save(str(init_path), {"scope": "bounded_target_B_scale_pilot_initialization", "ROCO_used": roco})
            cfg = copy.deepcopy(base)
            cfg.update(run_name=name, init_from=str(init_path), allow_random_init=False, figure_dir=str(folder / name / "figures"))
            path = folder / (name + ".json")
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            configs.append((cfg, path))
            del model, initial
            gc.collect()
        assert all(init_hashes[names[0]][key] == init_hashes[names[1]][key] for key in heads)
        assert set(init_hashes[names[0]]) == set(init_hashes[names[1]]) and init_hashes[names[0]] != init_hashes[names[1]]
        release = {"passed": True, "scope": state["scope"], "protocol_sha256": sha(folder / "protocol.json"),
            "data_seal_sha256": sha(folder / "data_seal.json"), "TRAIN_vocab_sha256": sha(folder / "TRAIN_answer_vocab.json"),
            "actual_manifests": protocol["actual_manifests"], "TRAIN_fracture_counts": dict(counts), "TRAIN_positive_weight": positive_weight,
            "initial_tensor_hashes": init_hashes, "new_head_tensors_byte_identical": len(heads),
            "fresh_ROCO_upstream_sha256": upstream["best_checkpoint_sha256"], "config_sha256": {cfg["run_name"]: sha(path) for cfg, path in configs},
            "same_downstream_budget": "4epochs296steps1173TRAIN142modelvalQA", "pretraining_compute_is_NOT_matched": True,
            "raw_threshold": 0.5, "primary_endpoint": "LAST296step checkpoints, best-by-development-AUPRC secondary only",
            "calibration_reserved": True, "test_used": False, "long_training_released": False}
        (folder / "execution_release.json").write_text(json.dumps(release, indent=2) + "\n")
        for cfg, path in configs:
            started = time.perf_counter()
            with (folder / (cfg["run_name"] + ".log")).open("w") as log:
                child = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(path), "--device", "cpu"], env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                guard = subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)])
                state.update(status="running", arm=cfg["run_name"], training_pid=child.pid, guard_pid=guard.pid, arm_started_utc=now())
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{cfg['run_name']} exit{code}; preserve log, no automatic retry")
            run = root / cfg["checkpoint_dir"] / cfg["run_name"]
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            optimizer = torch.load(run / "optimizer.pt", map_location="cpu", weights_only=False)
            history = json.loads((run / "history.json").read_text())
            assert len(history) == 4 and last["extra"]["global_step"] == optimizer["global_step"] == 296
            assert finite_tree(last["state"]) and finite_tree(optimizer)
            assert all(row["val_n"] == 142 and row["skipped_no_supervision_batches"] == 0 for row in history)
            completion = {"complete": True, "wall_seconds": time.perf_counter() - started, "history": history, "optimizer_steps": 296,
                "best_sha256": sha(run / "best.pt"), "last_sha256": sha(run / "last.pt"), "quality_gate_passed": False}
            (folder / (cfg["run_name"] + "_completion.json")).write_text(json.dumps(completion, indent=2) + "\n")
            del last, optimizer
            gc.collect()
            state.setdefault("completed_arms", []).append(cfg["run_name"])
            save()
        state.update(status="both_arms_complete_pending_independent_evaluation", finished_utc=now(), long_training_released=False)
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=now())
        save()
        raise


if __name__ == "__main__":
    main()
