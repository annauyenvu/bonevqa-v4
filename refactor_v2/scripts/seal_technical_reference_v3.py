"""Seal/verify the CPU technical reference, never authorize long scientific runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--regression_log", default="refactor_v2/reports/step3_seal_regression_v3.log")
    args = parser.parse_args()
    root = Path.cwd()
    seal = root / "refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json"
    if args.verify:
        payload = json.loads(seal.read_text())
        if payload["status"] != "sealed_cpu_technical_reference" or payload["long_scientific_training_authorized"] is not False:
            raise RuntimeError("Unsupported technical seal scope")
        failures = [name for name, expected in payload["artifacts"].items() if digest(root / name) != expected]
        if failures:
            raise RuntimeError("Technical reference changed: " + str(failures))
        print(json.dumps({"verified": True, "artifacts": len(payload["artifacts"]), "long_training_authorized": False}))
        return
    log = root / args.regression_log
    if seal.exists() or log.exists():
        raise RuntimeError("Existing seal/log must be preserved; use --verify or a new version")
    checks = {
        "revised_recipe_audit_v3.json": "passed",
        "common_initialization_seed42_fold0_v3.json": "all_passed",
        "gate_p_answer_route_real_v3.json": "passed",
        "p_controls_real_interventions_v3.json": "passed_technical_interventions",
        "vl_detector_routes_real_v3.json": "all_passed",
        "vl_controls_real_mechanics_v3_recheck.json": "passed_technical_mechanics",
        "component_batch_schedules_v3.json": "passed",
        "component_sampler_resume_train_only_v3/gate.json": "passed",
        "component_sampler_resume_train_only_v3_p_shuffled/gate.json": "passed",
        "p_component_answer_route_v3.json": "passed",
        "p_eval_context_rule_train_only_v3.json": "passed",
        "calibration_e2e_train_only_v3_provenance_recheck/gate.json": "passed",
        "caption_bridge_real_audit_pilot_roco_cpu_reference_v3.json": "passed",
    }
    paths = []
    for relative, key in checks.items():
        path = root / "refactor_v2/reports" / relative
        if json.loads(path.read_text()).get(key) is not True:
            raise RuntimeError("Technical gate has not passed: " + relative)
        paths.append(path)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Offline environment required")
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-m", "pytest", "refactor_v2/tests", "-q"],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    paths += [log, root / "refactor_v2/reports/revised_recipe_candidates_v3.json",
              root / "refactor_v2/reports/ARCHITECTURE_LOSS_RUNTIME_PROTOCOL_V3_REVIEW.md",
              root / "refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json",
              root / "refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json"]
    for folder in ("src", "tests", "scripts"):
        paths += sorted((root / "refactor_v2" / folder).rglob("*.py"))
    paths += sorted((root / "refactor_v2/configs/revised_sealed_candidates_v3").glob("*.json"))
    paths += [root / "refactor_v2/app/inference.py"] if (root / "refactor_v2/app/inference.py").exists() else []
    # Bind source data bytes too, not only a seal file whose referenced inputs
    # could later change independently of the source snapshot.
    for name in ("STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json", "STEP1_DATA_SEAL_V3_ALIGNMENT.json"):
        data_seal = json.loads((root / "refactor_v2/reports" / name).read_text())
        for relative, expected in data_seal["artifacts"].items():
            path = root / relative
            if digest(path) != expected:
                raise RuntimeError("Sealed data changed: " + relative)
            paths.append(path)
    payload = {"status": "sealed_cpu_technical_reference", "created_utc": datetime.now(timezone.utc).isoformat(),
        "step": 3, "scope": "CPU technical implementation and candidate-recipe snapshot; NOT module efficacy or long-run release",
        "reference_python": sys.executable, "torch_version": torch.__version__,
        "regression_log": str(log.relative_to(root)), "gate_checks": checks,
        "long_scientific_training_authorized": False, "next_action": "bounded fresh-upstream timing and per-run execution release",
        "outstanding_execution_inputs": ["fresh ROCO/PMC checkpoints", "new-fold detector complete verified proposals"],
        "limitations": ["MPS reproducibility not certified", "candidate recipes intentionally not launch-sealed",
                        "full fold detector quality and matched module efficacy/seeds unmeasured",
                        "test-inspected development-informed evaluation, no untouched external/clinical validation",
                        "fresh upstream-derived common initialization needs byte-parity check before ablations"],
        "artifacts": {path.relative_to(root).as_posix(): digest(path) for path in sorted(set(paths))}}
    seal.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"sealed_cpu_technical_reference": True, "artifacts": len(payload["artifacts"]),
                      "long_training_authorized": False, "regression_log": log.read_text().strip().splitlines()[-1]}, indent=2))


if __name__ == "__main__":
    main()
