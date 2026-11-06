"""Independent296-step B pilot evaluation; image+question-only inference, no fitting."""
import gc
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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
    folder = root / "refactor_v2/reports/fracatlas_scale_pilot_v3"
    progress = folder / "independent_evaluation_progress.json"
    output = folder / "independent_evaluation.json"
    assert json.loads((folder / "progress.json").read_text())["status"] == "both_arms_complete_pending_independent_evaluation"
    if progress.exists() or output.exists():
        raise RuntimeError("Existing evaluation must not be overwritten/duplicated")
    state = {"status": "verifying", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Inspected development cohorts only, not independent test/clinical evidence"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    started = time.perf_counter()
    try:
        names = ("fracatlas_B_fresh_scale_pilot_v3", "fracatlas_B_roco_scale_pilot_v3")
        release = json.loads((folder / "execution_release.json").read_text())
        assert release["passed"] and release["new_head_tensors_byte_identical"] == 8
        checks, configs = {}, {}
        for name in names:
            run = root / "refactor_v2/checkpoints" / name
            done = json.loads((folder / (name + "_completion.json")).read_text())
            assert sha(folder / (name + ".json")) == release["config_sha256"][name]
            assert sha(run / "last.pt") == done["last_sha256"] and sha(run / "best.pt") == done["best_sha256"]
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            optimizer = torch.load(run / "optimizer.pt", map_location="cpu", weights_only=False)
            assert finite_tree(last["state"]) and finite_tree(best["state"]) and finite_tree(optimizer)
            assert last["extra"]["epoch"] == 4 and last["extra"]["global_step"] == optimizer["global_step"] == 296
            cfg = json.loads((run / "config.json").read_text())
            configs[name] = cfg
            ds = {role: build_split(build_dataset, cfg["data"]["dataset"], role, None, True, cfg["data"]["loader_kwargs"]) for role in ("train", "model_val")}
            assert {role: manifest(data) for role, data in ds.items()} == release["actual_manifests"]
            initial = torch.load(folder / (name + "_initialization.pt"), map_location="cpu", weights_only=False)["state"]
            assert {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in initial.items()} == release["initial_tensor_hashes"][name]
            updates = {key: float((last["state"][key]-value).abs().max()) for key, value in initial.items() if key.startswith(("closed_head.", "fracture_head."))}
            assert all(value > 0 for value in updates.values())
            checks[name] = {"passed": True, "last_steps": 296, "weights_optimizer_finite": True, "exact_manifests": True,
                "task_head_max_updates": updates, "wall_seconds": done["wall_seconds"]}
            del initial, last, best, optimizer
            gc.collect()
        cfg = configs[names[0]]
        pilot = build_split(build_dataset, cfg["data"]["dataset"], "model_val", None, True, cfg["data"]["loader_kwargs"])
        train = build_split(build_dataset, cfg["data"]["dataset"], "train", None, True, cfg["data"]["loader_kwargs"])
        base = json.loads((root / "refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
        full = build_split(build_dataset, base["data"]["dataset"], "model_val", None, True, base["data"]["loader_kwargs"])
        canonical = "Is there a fracture in this X-ray?"
        full_indices = [i for i, row in enumerate(records_of(full)) if row["question"] == canonical]
        assert len(pilot) == 142 and len(full_indices) == 321
        batch_cfg = cfg["train"]["component_batching"]
        lookup, _ = load_component_lookup(batch_cfg["manifest"], batch_cfg["seal"], root)
        image_first = {}
        for i, row in enumerate(records_of(pilot)):
            image_first.setdefault(row["image_id"], i)
        ids = sorted(image_first)
        donors = dict(zip(ids, ids[1:]+ids[:1]))
        assert all(lookup[i] != lookup[d] for i, d in donors.items())
        pixels = {iid: pilot[i]["image"] for iid, i in image_first.items()}
        old = json.loads((root / "refactor_v2/reports/fracatlas_baseline_pilot_v3/independent_paired_evaluation.json").read_text())
        shared_nonconstant = set(old["nonconstant_questions_defined_from_TRAIN"])
        def metrics(rows):
            result = compute_metrics(rows, fracture_threshold=0.5)
            byq = defaultdict(list)
            for row in rows:
                byq[row["question"]].append(row)
            result["per_question"] = {q: {"n": len(group), "exact": sum(normalize_answer(row["gt"]) == normalize_answer(row["pred"]) for row in group)/len(group)} for q, group in byq.items()}
            result["macro_question_exact"] = sum(v["exact"] for v in result["per_question"].values())/len(byq)
            result["shared_original_TRAIN_nonconstant_questions"] = compute_metrics([row for row in rows if row["question"] in shared_nonconstant], fracture_threshold=0.5)
            return result
        predictions, endpoints = {}, {}
        for name in names:
            checkpoint = root / "refactor_v2/checkpoints" / name / "last.pt"
            set_seed(42)
            model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
            assert not any(model.cfg.get(key) for key in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "use_region_in_prompt"))
            assert model.cfg["fracture_yes_threshold"] == 0.5 and model.cfg.get("fracture_platt_slope") is None
            with torch.enable_grad():
                params = dict(model.named_parameters())
                selected = ["qformer.query", "prefix_proj.0.weight"]
                selected += [key for key in params if key.startswith("closed_head.") and key.endswith("weight")][-1:]
                selected += [key for key in params if key.startswith("fracture_head.") and key.endswith("weight")][-1:]
                out = model(collate_fn([train[i] for i in range(4)]), stage=3)
                gradients = torch.autograd.grad(out["loss_answer"], [params[key] for key in selected], allow_unused=True)
                grad = {key: 0.0 if value is None else float(value.abs().max()) for key, value in zip(selected, gradients)}
                assert all(value > 0 and np.isfinite(value) for value in grad.values())
                checks[name]["isolated_TRAIN_answer_gradient_max"] = grad
                del out, gradients
            for condition in ("correct_image", "component_disjoint_wrong_image", "full321_fracture"):
                dataset = full if condition == "full321_fracture" else pilot
                indices = full_indices if condition == "full321_fracture" else list(range(142))
                rows = []
                for start in range(0, len(indices), 4):
                    batch = collate_fn([dataset[i] for i in indices[start:start+4]])
                    images = batch["images"] if condition != "component_disjoint_wrong_image" else [pixels[donors[iid]] for iid in batch["image_ids"]]
                    # ONLY these two fields reach inference: no target/answer type/region/ROI.
                    generated = model.predict_batch({"images": images, "questions": batch["questions"]}, tra_chi_tiet=True)
                    for i, pred in enumerate(generated):
                        row = {"image_id": batch["image_ids"][i], "question": batch["questions"][i], "gt": batch["answers"][i],
                               "answer_type": batch["answer_types"][i], "pred": pred["answer"]}
                        if "p_yes" in pred:
                            row["p_yes"] = pred["p_yes"]
                        rows.append(row)
                result = {"checkpoint_sha256": sha(checkpoint), "metrics": metrics(rows), "records": rows}
                if condition == "full321_fracture":
                    endpoints[name] = result
                else:
                    predictions[name + "/" + condition] = result
                state.update(status="running", completed_arm=name, completed_condition=condition)
                save()
            a = {row["image_id"]: row for row in predictions[name+"/correct_image"]["records"] if row["question"] == canonical}
            b = {row["image_id"]: row for row in predictions[name+"/component_disjoint_wrong_image"]["records"] if row["question"] == canonical}
            assert max(abs(row["p_yes"]-a[donors[iid]]["p_yes"]) for iid, row in b.items()) <= 1e-6
            del model
            gc.collect()
        reference = endpoints[names[0]]["records"]
        assert all([(row["image_id"], row["gt"]) for row in endpoints[name]["records"]] == [(row["image_id"], row["gt"]) for row in reference] for name in names)
        labels = np.asarray([row["gt"] == "yes" for row in reference], dtype=int)
        scores = {name: np.asarray([row["p_yes"] for row in endpoints[name]["records"]]) for name in names}
        groups = [lookup[row["image_id"]] for row in reference]
        clusters = sorted(set(groups))
        members = {group: [i for i, value in enumerate(groups) if value == group] for group in clusters}
        rng = random.Random(20261019)
        boot = {name: {"AUROC": [], "AUPRC": []} for name in names}
        diffs = []
        for _ in range(500):
            chosen = [i for group in rng.choices(clusters, k=len(clusters)) for i in members[group]]
            if len(set(labels[chosen])) < 2:
                continue
            for name in names:
                boot[name]["AUROC"].append(float(roc_auc_score(labels[chosen], scores[name][chosen])))
                boot[name]["AUPRC"].append(float(average_precision_score(labels[chosen], scores[name][chosen])))
            diffs.append(boot[names[1]]["AUROC"][-1]-boot[names[0]]["AUROC"][-1])
        intervals = {name: {key: np.percentile(values, [2.5,97.5]).tolist() for key, values in result.items()} for name, result in boot.items()}
        report = {"scope": state["scope"], "primary": "LAST296steps matched downstream budget; one seed; no matched pretraining budget",
            "artifact_and_gradient_verification": checks, "pilot142_QA": predictions, "full321_fracture": endpoints,
            "component_bootstrap95_descriptive_intervals": intervals, "paired_ROCO_minus_fresh_AUROC_interval": np.percentile(diffs,[2.5,97.5]).tolist(),
            "donor_bijection": donors, "input_keys_only": ["images","questions"], "test_or_calibration_used": False,
            "source_full_model_val_sha256": sha(base["data"]["loader_kwargs"]["sealed_records_by_split"]["model_val"]),
            "elapsed_seconds": time.perf_counter()-started, "raw_threshold": 0.5, "long_training_released": False}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        state.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
