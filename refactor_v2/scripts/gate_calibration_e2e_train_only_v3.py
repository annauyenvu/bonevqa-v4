"""Real evaluator -> fit -> frozen runtime test, all from outer TRAIN only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.calibration import apply_platt_probability, load_calibration_artifact
from src.data.sealed_manifest import load_sealed_records
from src.endpoints import is_canonical_fracture_presence_question
from src.utils import load_config, normalize_answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="refactor_v2/reports/calibration_e2e_train_only_v3")
    args = parser.parse_args()
    start = time.perf_counter()
    root = Path.cwd()
    folder = root / args.output_dir
    if folder.exists():
        raise RuntimeError("Existing calibration audit must be inspected, not overwritten")
    folder.mkdir()
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    loader = config["data"]["loader_kwargs"]
    source = Path(loader["sealed_records_by_split"]["train"])
    rows = load_sealed_records(source, loader["seal_path"], root)
    components = json.loads((root / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json").read_text())["components"]
    singleton = {group[0] for group in components if len(group) == 1}
    by_label = {label: [row for row in rows if row["image_id"] in singleton
                       and is_canonical_fracture_presence_question(row["question"])
                       and normalize_answer(row["answer"]) == label] for label in ("yes", "no")}
    paths, artifacts = {}, {}
    for role, offset in (("calibration", 0), ("test", 2)):
        selected = by_label["yes"][offset:offset + 2] + by_label["no"][offset:offset + 2]
        if len({row["image_id"] for row in selected}) != 4:
            raise RuntimeError("Need four distinct TRAIN images in each technical cohort")
        path = folder / (role + "_train_derivative.json")
        path.write_text(json.dumps(selected, indent=2) + "\n")
        paths[role] = path
        artifacts[path.relative_to(root).as_posix()] = sha(path)
    seal = folder / "data_seal.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "scope": "technical_smoke_train_only",
                               "parent_manifest": str(source), "parent_manifest_sha256": sha(source),
                               "parent_seal_sha256": sha(loader["seal_path"]), "artifacts": artifacts}, indent=2) + "\n")
    checkpoint = root / "refactor_v2/checkpoints/common_initialization_seed42_fold0_v3/initial.pt"
    def run(args, name):
        with (folder / (name + ".log")).open("w") as stream:
            subprocess.run([sys.executable, *args], env=os.environ.copy(), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    def evaluate(role, tag, calibration=None):
        args = ["-m", "src.evaluate", "--checkpoint", str(checkpoint), "--device", "cpu",
                "--dataset", "fracatlas_component_clean", "--split", role,
                "--sealed_records_path", str(paths[role]), "--data_seal", str(seal),
                "--project_root", str(root), "--technical_smoke", "--batch_size", "4",
                "--report_dir", str(folder), "--figure_dir", str(folder / "figures"), "--tag", tag]
        if calibration:
            args += ["--fracture_calibration", str(calibration)]
        run(args, tag)
        return folder / ("results_fracatlas_component_clean_" + tag + ".json")
    calibration_raw = evaluate("calibration", "calibration_raw")
    test_raw = evaluate("test", "test_raw")
    artifact = folder / "frozen_calibration.json"
    run(["refactor_v2/scripts/clinical_fracture_calibration_v3.py", "--calibration", str(calibration_raw),
         "--test", str(test_raw), "--output", str(artifact), "--bootstrap", "20"], "fit_calibration")
    calibrated = evaluate("test", "test_frozen", artifact)
    frozen = load_calibration_artifact(artifact, expected_checkpoint=checkpoint,
                                       expected_manifest=paths["test"], allow_technical_smoke=True)
    raw = json.loads(test_raw.read_text())["records"]
    evaluated = json.loads(calibrated.read_text())
    errors, decisions = [], []
    for left, right in zip(raw, evaluated["records"]):
        expected = apply_platt_probability(left["p_yes"], frozen["platt_slope"], frozen["platt_intercept"])
        errors.append(abs(expected - right["p_yes"]))
        decisions.append(right["pred"] == ("yes" if expected >= frozen["fracture_yes_threshold"] else "no"))
    clinical = json.loads(artifact.read_text())
    metrics = evaluated["metrics"]
    agreement = {key: abs(metrics[key] - clinical["test_metrics"][key])
                 for key in ("fracture_sensitivity", "fracture_specificity", "fracture_balanced_accuracy")}
    blocked = False
    try:
        load_calibration_artifact(artifact, expected_checkpoint=checkpoint)
    except RuntimeError as error:
        blocked = "production artifact" in str(error)
    report = {"scope": "8 outer-TRAIN images; fresh initialization; NOT clinical/generalization performance",
              "checkpoint_sha256": sha(checkpoint), "artifact_sha256": sha(artifact),
              "max_probability_transform_error": max(errors), "runtime_decisions_match": all(decisions),
              "evaluator_vs_frozen_report_metric_errors": agreement,
              "production_smoke_artifact_blocked": blocked,
              "raw_and_calibrated_metrics_both_retained": "raw_test_metrics_at_0_5" in clinical,
              "seconds": time.perf_counter() - start,
              "passed": max(errors) < 1e-12 and all(decisions) and blocked and all(value < 1e-12 for value in agreement.values())}
    (folder / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
