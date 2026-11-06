#!/usr/bin/env python3
"""Leakage-safe fracture calibration and image-clustered uncertainty.

Fit Platt scaling and select the operating threshold on the dedicated
calibration cohort. Apply both unchanged to the sealed test cohort.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.endpoints import is_canonical_fracture_presence_question
from src.calibration import apply_platt_probability
from src.data.sealed_manifest import load_sealed_records
from src.utils import compute_metrics, normalize_answer, normalize_yes_no


def validate_sealed_cohorts(calibration_payload, test_payload, root, component_manifest, component_seal):
    """Require complete authentic QA/GT coverage and disjoint duplicate components."""
    root = Path(root).resolve()
    component_manifest = Path(component_manifest).resolve()
    component_seal = Path(component_seal).resolve()
    seal = json.loads(component_seal.read_text())
    component_hash = hashlib.sha256(component_manifest.read_bytes()).hexdigest()
    if seal.get("status") != "sealed_development_data" or seal.get("artifacts", {}).get(
            component_manifest.relative_to(root).as_posix()) != component_hash:
        raise ValueError("Component manifest is not hash-bound to the data seal")
    groups = json.loads(component_manifest.read_text())["components"]
    membership = {}
    for index, group in enumerate(groups):
        for image in group:
            if image in membership:
                raise ValueError("Image belongs to multiple components")
            membership[image] = index
    used_components = []
    counts = {}
    scopes = {payload.get("evaluation_scope") for payload in (calibration_payload, test_payload)}
    if len(scopes) != 1 or not scopes <= {"technical_smoke_train_only", "development_informed"}:
        raise ValueError("Calibration/application evaluation scopes must match and be explicit")
    manifests = []
    def key(row, target):
        return (row["image_id"], row["question"], normalize_answer(row[target]), row["answer_type"])
    for role, payload in (("calibration", calibration_payload), ("test", test_payload)):
        provenance = payload.get("sealed_data_provenance")
        if not provenance or payload.get("split") != role:
            raise ValueError("Calibration requires sealed full calibration/test outputs")
        manifest = Path(provenance["manifest"]).resolve()
        manifests.append(manifest)
        if scopes == {"development_informed"} and manifest.name != "vqa_" + role + ".json":
            raise ValueError("Scientific cohort manifest role does not match split")
        for field in ("manifest", "seal"):
            if hashlib.sha256(Path(provenance[field]).read_bytes()).hexdigest() != provenance.get(field + "_sha256"):
                raise ValueError("Calibration input provenance hash mismatch")
        source = load_sealed_records(provenance["manifest"], provenance["seal"], root)
        if Counter(key(row, "answer") for row in source) != Counter(key(row, "gt") for row in payload["records"]):
            raise ValueError("Evaluation records do not exactly cover sealed QA and ground truth")
        endpoint = fracture_rows(payload)
        ids = [row["image_id"] for row in endpoint]
        if len(ids) != len(set(ids)):
            raise ValueError("Canonical fracture endpoint must occur once per image")
        all_ids = {row["image_id"] for row in source}
        if not all_ids <= membership.keys():
            raise ValueError("Evaluation image is absent from sealed components")
        used_components.append({membership[image] for image in all_ids})
        counts[role] = {"qa": len(source), "images": len(all_ids), "fracture_images": len(ids)}
    if used_components[0] & used_components[1]:
        raise ValueError("Calibration and test share duplicate components")
    if scopes == {"development_informed"} and manifests[0].parent != manifests[1].parent:
        raise ValueError("Calibration and application manifests must belong to the same outer fold")
    return {"component_manifest": str(component_manifest), "component_manifest_sha256": component_hash,
            "component_seal": str(component_seal),
            "component_seal_sha256": hashlib.sha256(component_seal.read_bytes()).hexdigest(),
            "full_qa_and_ground_truth_coverage": True, "component_overlap": 0, "counts": counts}


def fracture_rows(payload: dict) -> list[dict]:
    return [
        row for row in payload["records"]
        if row.get("answer_type") == "CLOSED"
        and is_canonical_fracture_presence_question(row.get("question"))
        and normalize_yes_no(row.get("gt", "")) in ("yes", "no")
    ]


def labels_scores(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any(row.get("p_yes") is None for row in rows):
        raise ValueError("Every fracture record must contain p_yes")
    labels = np.asarray([normalize_yes_no(row["gt"]) == "yes" for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["p_yes"]) for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Raw calibration probabilities must be finite and in [0,1]")
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires both positive and negative examples")
    return labels, scores


def logit(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(scores / (1.0 - scores))


def fit_platt(rows: list[dict]) -> tuple[float, float]:
    labels, scores = labels_scores(rows)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(logit(scores).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(rows: list[dict], slope: float, intercept: float) -> list[dict]:
    calibrated = copy.deepcopy(rows)
    _, scores = labels_scores(rows)
    probs = [apply_platt_probability(float(score), slope, intercept) for score in scores]
    for row, probability in zip(calibrated, probs):
        row["p_yes"] = float(probability)
        row["pred"] = "yes" if probability >= 0.5 else "no"
    return calibrated


def select_sensitivity_threshold(rows: list[dict], target: float) -> float:
    labels, scores = labels_scores(rows)
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    feasible = []
    for threshold in candidates:
        pred = scores >= threshold
        sensitivity = float(pred[labels == 1].mean())
        specificity = float((~pred[labels == 0]).mean())
        if sensitivity + 1e-12 >= target:
            feasible.append((specificity, sensitivity, float(threshold)))
    if not feasible:
        raise RuntimeError(f"No threshold reaches target sensitivity {target}")
    # Highest specificity first; then sensitivity; then the more conservative
    # (higher) threshold. All selection occurs on calibration only.
    return max(feasible)[2]


def metric_view(rows: list[dict], threshold: float) -> dict:
    return compute_metrics(rows, fracture_threshold=threshold)


def clustered_bootstrap(rows: list[dict], threshold: float, n_boot: int, seed: int) -> dict:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[str(row.get("image_id") or f"missing-{index}")].append(row)
    image_ids = sorted(by_image)
    rng = np.random.default_rng(seed)
    keys = [
        "fracture_sensitivity", "fracture_specificity", "fracture_balanced_accuracy",
        "fracture_auroc", "fracture_auprc", "fracture_brier", "fracture_ece10",
    ]
    values = {key: [] for key in keys}
    for _ in range(n_boot):
        sample_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = []
        for draw_index, image_id in enumerate(sample_ids):
            for row in by_image[image_id]:
                copied = dict(row)
                copied["image_id"] = f"draw-{draw_index}-{image_id}"
                sample.append(copied)
        metrics = metric_view(sample, threshold)
        for key in keys:
            if key in metrics and np.isfinite(metrics[key]):
                values[key].append(float(metrics[key]))
    return {
        key: {
            "estimate": float(metric_view(rows, threshold)[key]),
            "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5)),
            "n_boot_valid": len(samples),
        }
        for key, samples in values.items() if samples
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target_sensitivity", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--component_manifest", type=Path, default=Path(".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"))
    parser.add_argument("--component_seal", type=Path, default=Path("refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json"))
    args = parser.parse_args()
    if not 0.0 < args.target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")

    calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    test_payload = json.loads(args.test.read_text(encoding="utf-8"))
    cohort_verification = validate_sealed_cohorts(calibration_payload, test_payload, args.project_root,
        args.project_root / args.component_manifest, args.project_root / args.component_seal)
    calibration_raw = fracture_rows(calibration_payload)
    test_raw = fracture_rows(test_payload)
    if calibration_payload.get("fracture_calibration_provenance") or test_payload.get("fracture_calibration_provenance"):
        raise ValueError("Calibration inputs must be RAW outputs, not already calibrated")
    calibration_ids = {row["image_id"] for row in calibration_raw}
    test_ids = {row["image_id"] for row in test_raw}
    if calibration_ids & test_ids:
        raise ValueError("Calibration and application cohorts share image IDs")
    if calibration_payload.get("sealed_data_provenance") or test_payload.get("sealed_data_provenance"):
        if calibration_payload.get("split") != "calibration" or test_payload.get("split") != "test":
            raise ValueError("Sealed calibration requires calibration and test split labels")
        if not calibration_payload.get("checkpoint") or calibration_payload.get("checkpoint") != test_payload.get("checkpoint"):
            raise ValueError("Calibration and test must identify the same checkpoint")
        checkpoint_hash = hashlib.sha256(Path(calibration_payload["checkpoint"]).read_bytes()).hexdigest()
        if any(payload.get("checkpoint_sha256") != checkpoint_hash for payload in (calibration_payload, test_payload)):
            raise ValueError("Calibration input checkpoint hash mismatch or missing binding")
        if calibration_payload.get("fracture_score_space") != test_payload.get("fracture_score_space"):
            raise ValueError("Calibration input score spaces differ")
    slope, intercept = fit_platt(calibration_raw)
    calibration = apply_platt(calibration_raw, slope, intercept)
    test = apply_platt(test_raw, slope, intercept)
    threshold = select_sensitivity_threshold(calibration, args.target_sensitivity)

    report = {
        "protocol_version": 3,
        "cohort_verification": cohort_verification,
        "checkpoint": calibration_payload.get("checkpoint"),
        "checkpoint_sha256": calibration_payload.get("checkpoint_sha256"),
        "scope": calibration_payload.get("evaluation_scope", "legacy_unbound"),
        "calibration_input_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "application_input_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
        "calibration_data_provenance": calibration_payload.get("sealed_data_provenance"),
        "application_data_provenance": test_payload.get("sealed_data_provenance"),
        "model_selection_split": "model_val (not consumed here)",
        "calibration_split": calibration_payload.get("split"),
        "application_split": test_payload.get("split"),
        "test_used_for_fitting_or_threshold_selection": False,
        "calibration_method": "Platt scaling on logit(p_yes)",
        "platt_slope": slope,
        "platt_intercept": intercept,
        "threshold_policy": f"maximum calibration specificity subject to sensitivity >= {args.target_sensitivity:.3f}",
        "target_sensitivity": args.target_sensitivity,
        "fracture_yes_threshold": threshold,
        "calibration_metrics": metric_view(calibration, threshold),
        "raw_calibration_metrics_at_0_5": metric_view(calibration_raw, 0.5),
        "raw_test_metrics_at_0_5": metric_view(test_raw, 0.5),
        "platt_test_metrics_at_0_5": metric_view(test, 0.5),
        "negative_platt_slope": slope < 0,
        "test_metrics": metric_view(test, threshold),
        "test_image_clustered_bootstrap_ci95": clustered_bootstrap(
            test, threshold, args.bootstrap, args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
