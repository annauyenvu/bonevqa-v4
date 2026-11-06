"""Artifact/routing verification and component-disjoint image interventions for B pilot."""
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.context_controls import load_component_lookup
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_split
from src.utils import compute_metrics, normalize_answer, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/fracatlas_baseline_pilot_v3"
    progress = folder / "independent_evaluation_progress.json"
    output = folder / "independent_paired_evaluation.json"
    if progress.exists() or output.exists():
        raise RuntimeError("Existing independent evaluation: never duplicate/overwrite")
    driver = json.loads((folder / "progress.json").read_text())
    assert driver["status"] == "both_arms_complete_pending_independent_evaluation"
    state = {"status": "verifying", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "142 inspected grouped development QA; NOT independent test/clinical evidence"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    started = time.perf_counter()
    try:
        gate = json.loads((folder / "execution_release.json").read_text())
        assert gate["passed"] and gate["new_closed_and_fracture_heads_byte_identical"]
        names = ("fracatlas_B_fresh_pilot_v3", "fracatlas_B_roco_pilot_v3")
        checks, configs = {}, {}
        for name in names:
            run = root / "refactor_v2/checkpoints" / name
            done = json.loads((folder / (name + "_completion.json")).read_text())
            assert sha(folder / (name + ".json")) == gate["config_sha256"][name]
            assert sha(run / "last.pt") == done["last_sha256"] and sha(run / "best.pt") == done["best_sha256"]
            cfg = json.loads((run / "config.json").read_text())
            configs[name] = cfg
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            optimizer = torch.load(run / "optimizer.pt", map_location="cpu", weights_only=False)
            assert finite_tree(last["state"]) and finite_tree(best["state"]) and finite_tree(optimizer)
            assert last["extra"]["epoch"] == 4 and last["extra"]["global_step"] == optimizer["global_step"] == 76
            history = json.loads((run / "history.json").read_text())
            assert len(history) == 4 and all(row["skipped_no_supervision_batches"] == 0 and row["val_n"] == 142 for row in history)
            assert best["extra"]["best_metric"] == max(row["val_fracture_auprc"] for row in history)
            actual = {role: build_split(build_dataset, cfg["data"]["dataset"], role, None, True, cfg["data"]["loader_kwargs"])
                      for role in ("train", "model_val")}
            assert {role: manifest(data) for role, data in actual.items()} == gate["actual_manifests"]
            initial = torch.load(folder / (name + "_initialization.pt"), map_location="cpu", weights_only=False)["state"]
            assert {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in initial.items()} == gate["initial_tensor_hashes"][name]
            updates = {key: float((last["state"][key] - value).abs().max()) for key, value in initial.items() if key.startswith(("closed_head.", "fracture_head."))}
            assert updates and all(value > 0 for value in updates.values())
            checks[name] = {"passed": True, "weights_and_optimizer_finite": True, "last_epoch": 4, "last_steps": 76,
                           "best_epoch": best["extra"]["epoch"], "best_steps": best["extra"]["global_step"],
                           "actual_manifests_match": True, "new_task_head_max_updates": updates, "wall_seconds": done["wall_seconds"]}
            del initial, best, last, optimizer
            gc.collect()
        cfg = configs[names[0]]
        data = {role: build_split(build_dataset, cfg["data"]["dataset"], role, None, True, cfg["data"]["loader_kwargs"])
                for role in ("train", "model_val")}
        dataset = data["model_val"]
        rows = records_of(dataset)
        batches = cfg["train"]["component_batching"]
        lookup, _ = load_component_lookup(batches["manifest"], batches["seal"], root)
        first = {}
        for i, row in enumerate(rows):
            first.setdefault(row["image_id"], i)
        ids = sorted(first)
        donors = dict(zip(ids, ids[1:] + ids[:1]))
        assert len(set(donors.values())) == len(ids) and all(lookup[i] != lookup[d] for i, d in donors.items())
        # Only donor pixels replace the image. Question, target, type and identity stay fixed.
        donor_images = {iid: dataset[index]["image"] for iid, index in first.items()}
        train_answers = defaultdict(set)
        for row in records_of(data["train"]):
            train_answers[row["question"]].add(normalize_answer(row["answer"]))
        nonconstant = {q for q, answers in train_answers.items() if len(answers) > 1}
        def metrics(records):
            result = compute_metrics(records, fracture_threshold=0.5)
            groups = defaultdict(list)
            for record in records:
                groups[record["question"]].append(record)
            result["per_question"] = {q: {"n": len(group), "exact": sum(normalize_answer(r["pred"]) == normalize_answer(r["gt"]) for r in group) / len(group)} for q, group in groups.items()}
            result["macro_question_exact"] = sum(group["exact"] for group in result["per_question"].values()) / len(groups)
            subset = [r for r in records if r["question"] in nonconstant]
            result["TRAIN_nonconstant_question_subset"] = compute_metrics(subset, fracture_threshold=0.5)
            return result
        results = {}
        for name in names:
            for kind in ("last", "best"):
                checkpoint = root / "refactor_v2/checkpoints" / name / (kind + ".pt")
                set_seed(42)
                model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
                assert not any(model.cfg.get(key) for key in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "use_region_in_prompt"))
                if kind == "last":
                    # No optimizer: isolated answer-loss gradients on four existing TRAIN QA.
                    with torch.enable_grad():
                        params = dict(model.named_parameters())
                        selected = ["qformer.query", "prefix_proj.0.weight"]
                        selected += [key for key in params if key.startswith("closed_head.") and key.endswith("weight")][-1:]
                        selected += [key for key in params if key.startswith("fracture_head.") and key.endswith("weight")][-1:]
                        out = model(collate_fn([data["train"][i] for i in range(4)]), stage=3)
                        gradients = torch.autograd.grad(out["loss_answer"], [params[key] for key in selected], allow_unused=True)
                        grad_qc = {key: 0.0 if grad is None else float(grad.abs().max()) for key, grad in zip(selected, gradients)}
                        assert all(value > 0 for value in grad_qc.values())
                        checks[name]["isolated_answer_gradient_max_on_TRAIN"] = grad_qc
                        del out, gradients
                for condition in ("correct_image", "component_disjoint_wrong_image"):
                    records = []
                    for start in range(0, len(dataset), 4):
                        batch = collate_fn([dataset[i] for i in range(start, min(start + 4, len(dataset)))])
                        batch["regions"] = [None] * len(batch["images"])
                        batch["masks"] = [None] * len(batch["images"])
                        batch["boxes"] = [None] * len(batch["images"])
                        if condition != "correct_image":
                            batch["images"] = [donor_images[donors[iid]] for iid in batch["image_ids"]]
                        pred = model.predict_batch(batch, tra_chi_tiet=True)
                        for i, item in enumerate(pred):
                            row = {"image_id": batch["image_ids"][i], "question": batch["questions"][i], "gt": batch["answers"][i],
                                   "answer_type": batch["answer_types"][i], "pred": item["answer"]}
                            for field in ("p_yes", "p_yes_raw"):
                                if field in item:
                                    row[field] = item[field]
                            records.append(row)
                    assert len(records) == 142
                    key = name + "/" + kind + "/" + condition
                    results[key] = {"checkpoint_sha256": sha(checkpoint), "metrics": metrics(records), "records": records}
                    state.update(status="running", completed_condition=key)
                    save()
                del model
                gc.collect()
        (folder / "independent_artifact_verification.json").write_text(json.dumps({"passed": True, "arms": checks,
            "primary_comparison": "last checkpoints, exactly76steps each; selected best checkpoints secondary only"}, indent=2) + "\n")
        report = {"scope": state["scope"], "conditions": results, "primary_comparison": "last76step checkpoints",
                  "secondary": "best checkpoints selected on inspected development fracture AUPRC; unequal selected exposure, not primary budget-matched comparison",
                  "donor_bijection": donors, "nonconstant_questions_defined_from_TRAIN": sorted(nonconstant),
                  "elapsed_seconds": time.perf_counter() - started, "test_or_calibration_used": False,
                  "raw_fracture_threshold": 0.5, "independent_clinical_evidence": False, "long_training_released": False}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        state.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
