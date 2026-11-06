"""Frozen76-step B checkpoints on ALL fold0 model-val fracture endpoints; no fitting."""
import gc
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.context_controls import load_component_lookup
from src.data import build_dataset, collate_fn
from src.protocol import records_of
from src.models.bonevqa import BoneVQAModel
from src.train import build_split
from src.utils import compute_metrics, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    root = Path.cwd()
    parent = root / "refactor_v2/reports/fracatlas_baseline_pilot_v3"
    folder = root / "refactor_v2/reports/fracatlas_full_fracture_development_v3"
    if folder.exists():
        raise RuntimeError("Existing evaluation evidence; never duplicate/overwrite")
    prior_state = json.loads((parent / "independent_evaluation_progress.json").read_text())
    assert prior_state["status"] == "complete_pending_interpretation"
    prior_path = parent / "independent_paired_evaluation.json"
    assert sha(prior_path) == prior_state["report_sha256"]
    prior = json.loads(prior_path.read_text())
    names = ("fracatlas_B_fresh_pilot_v3", "fracatlas_B_roco_pilot_v3")
    canonical = "Is there a fracture in this X-ray?"
    donor_consistency = {}
    for name in names:
        true = {row["image_id"]: row for row in prior["conditions"][name + "/last/correct_image"]["records"] if row["question"] == canonical}
        wrong = {row["image_id"]: row for row in prior["conditions"][name + "/last/component_disjoint_wrong_image"]["records"] if row["question"] == canonical}
        error = max(abs(row["p_yes"] - true[prior["donor_bijection"][iid]]["p_yes"]) for iid, row in wrong.items())
        assert error <= 1e-6
        donor_consistency[name] = error
    base = json.loads((root / "refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
    dataset = build_split(build_dataset, base["data"]["dataset"], "model_val", None, True, base["data"]["loader_kwargs"])
    rows = records_of(dataset)
    indices = [i for i, row in enumerate(rows) if row["question"] == canonical]
    assert len(rows) == 1432 and len(indices) == len({rows[i]["image_id"] for i in indices}) == 321
    batching = base["train"]["component_batching"]
    lookup, component_hash = load_component_lookup(batching["manifest"], batching["seal"], root)
    folder.mkdir()
    progress = folder / "progress.json"
    state = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Expanded321-image model-validation, after pilot inspection; NOT untouched/independent evaluation"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    started = time.perf_counter()
    try:
        release = {"scope": state["scope"], "target": "canonical fracture presence only, existing dataset GT",
            "source_manifest_sha256": sha(base["data"]["loader_kwargs"]["sealed_records_by_split"]["model_val"]),
            "source_seal_sha256": sha(base["data"]["loader_kwargs"]["seal_path"]), "component_manifest_sha256": component_hash,
            "selected_QA_indices": indices, "frozen_checkpoints": "LAST76-step checkpoints, no retraining/selection/threshold fitting",
            "production_input": "original image + raw canonical question only; regions/boxes/masks removed",
            "cached_score_derangement_controls": "64 predetermined seeds20261019..20261082, bijective different-component donor assignments; not new wrong-image inference; prior actual pixel-swap score identity verified",
            "prior_actual_pixel_swap_max_score_error": donor_consistency,
            "bootstrap": "500 paired component resamples, seed20261019; descriptive development percentile intervals, not confirmatory p-values",
            "test_or_calibration_used": False, "optimizer_steps": 0, "long_training_released": False}
        (folder / "execution_release.json").write_text(json.dumps(release, indent=2) + "\n")
        predictions = {}
        for name in names:
            checkpoint = root / "refactor_v2/checkpoints" / name / "last.pt"
            done = json.loads((parent / (name + "_completion.json")).read_text())
            assert sha(checkpoint) == done["last_sha256"]
            set_seed(42)
            model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
            assert model.cfg["use_fracture_head"] and model.cfg["fracture_yes_threshold"] == 0.5
            assert model.cfg.get("fracture_platt_slope") is None and model.cfg.get("fracture_platt_intercept") is None
            assert not any(model.cfg.get(key) for key in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "use_region_in_prompt"))
            records = []
            for start in range(0, len(indices), 4):
                batch = collate_fn([dataset[i] for i in indices[start:start + 4]])
                n = len(batch["images"])
                batch.update(regions=[None] * n, boxes=[None] * n, masks=[None] * n)
                pred = model.predict_batch(batch, tra_chi_tiet=True)
                for i, item in enumerate(pred):
                    records.append({"image_id": batch["image_ids"][i], "question": canonical, "gt": batch["answers"][i],
                        "answer_type": "CLOSED", "pred": item["answer"], "p_yes": item["p_yes"], "component_id": lookup[batch["image_ids"][i]]})
                state.update(arm=name, evaluated_images=len(records), planned_images=321)
                save()
            assert len(records) == 321
            predictions[name] = {"checkpoint_sha256": sha(checkpoint), "metrics": compute_metrics(records, fracture_threshold=0.5), "records": records}
            del model
            gc.collect()
        reference = predictions[names[0]]["records"]
        assert all([(r["image_id"], r["gt"]) for r in predictions[name]["records"]] == [(r["image_id"], r["gt"]) for r in reference] for name in names)
        labels = np.asarray([row["gt"] == "yes" for row in reference], dtype=int)
        groups = [row["component_id"] for row in reference]
        scores = {name: np.asarray([row["p_yes"] for row in predictions[name]["records"]]) for name in names}
        controls = {}
        for seed in range(20261019, 20261083):
            order = list(range(321))
            rng = random.Random(seed)
            for attempt in range(256):
                rng.shuffle(order)
                if all(groups[i] != groups[j] for i, j in enumerate(order)):
                    break
            else:
                raise RuntimeError("No component-disjoint derangement; never silently weaken control")
            controls[str(seed)] = {name: {"AUROC": float(roc_auc_score(labels, values[order])),
                "AUPRC": float(average_precision_score(labels, values[order]))} for name, values in scores.items()}
        clusters = sorted(set(groups))
        members = {group: [i for i, value in enumerate(groups) if value == group] for group in clusters}
        rng = random.Random(20261019)
        boot = {name: {"AUROC": [], "AUPRC": []} for name in names}
        differences = []
        for _ in range(500):
            chosen = [i for group in rng.choices(clusters, k=len(clusters)) for i in members[group]]
            target = labels[chosen]
            if len(set(target)) < 2:
                continue
            for name, values in scores.items():
                boot[name]["AUROC"].append(float(roc_auc_score(target, values[chosen])))
                boot[name]["AUPRC"].append(float(average_precision_score(target, values[chosen])))
            differences.append(boot[names[1]]["AUROC"][-1] - boot[names[0]]["AUROC"][-1])
        intervals = {name: {metric: np.percentile(values, [2.5, 97.5]).tolist() for metric, values in stats.items()} for name, stats in boot.items()}
        report = {"scope": state["scope"], "arms": predictions, "n_images": len(reference), "n_components": len(clusters),
            "n_positive": int(labels.sum()), "n_negative": int((1-labels).sum()), "raw_threshold": 0.5,
            "cached_derangement_control_metrics": controls, "component_bootstrap_95percent_descriptive_intervals": intervals,
            "paired_ROCO_minus_fresh_AUROC_descriptive_interval": np.percentile(differences, [2.5, 97.5]).tolist(),
            "elapsed_seconds": time.perf_counter() - started, "test_or_calibration_used": False,
            "limitations": ["Expanded already-inspected development evaluation; no independent evidence", "one seed and small TRAIN pilot, no final module efficacy", "cached score controls assume imagewise B inference, supported by exact prior donor-score check", "no fitted threshold/calibration or expert clinical labels"],
            "long_training_released": False}
        output = folder / "evaluation.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        state.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
