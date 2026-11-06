"""Validated clinical calibration artifacts and runtime transforms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def apply_platt_probability(probability: float, slope: float, intercept: float) -> float:
    """Apply Platt scaling to a raw probability in a numerically stable way."""

    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    raw_logit = math.log(p / (1.0 - p))
    z = float(slope) * raw_logit + float(intercept)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 40.0)))
    ez = math.exp(max(z, -40.0))
    return ez / (1.0 + ez)


def load_calibration_artifact(path: str | Path, expected_checkpoint=None, expected_manifest=None,
                              allow_technical_smoke=False) -> dict:
    """Load a v3 artifact and fail closed on leakage/provenance violations."""

    calibration_path = Path(path)
    raw = calibration_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if int(payload.get("protocol_version", 0)) < 3:
        raise RuntimeError("Calibration artifact must use protocol_version >= 3")
    if payload.get("test_used_for_fitting_or_threshold_selection") is not False:
        raise RuntimeError("Calibration artifact does not prove test-independent fitting")
    required = ("platt_slope", "platt_intercept", "fracture_yes_threshold")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Calibration artifact is missing {missing}")
    threshold = float(payload["fracture_yes_threshold"])
    slope = float(payload["platt_slope"])
    intercept = float(payload["platt_intercept"])
    if not all(math.isfinite(value) for value in (threshold, slope, intercept)):
        raise RuntimeError("Calibration parameters must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("Calibration threshold must be in [0, 1]")
    if payload.get("scope") == "technical_smoke_train_only" and not allow_technical_smoke:
        raise RuntimeError("Technical-smoke calibration cannot be used as a production artifact")
    if expected_checkpoint is not None:
        digest = hashlib.sha256(Path(expected_checkpoint).read_bytes()).hexdigest()
        if payload.get("checkpoint_sha256") != digest:
            raise RuntimeError("Calibration checkpoint SHA-256 mismatch or missing binding")
    for key in ("calibration_data_provenance", "application_data_provenance"):
        provenance = payload.get(key)
        if provenance:
            for field in ("manifest", "seal"):
                source = Path(provenance[field])
                if hashlib.sha256(source.read_bytes()).hexdigest() != provenance.get(field + "_sha256"):
                    raise RuntimeError("Calibration data provenance hash mismatch: " + field)
    cohort = payload.get("cohort_verification")
    if cohort:
        for field in ("component_manifest", "component_seal"):
            if hashlib.sha256(Path(cohort[field]).read_bytes()).hexdigest() != cohort.get(field + "_sha256"):
                raise RuntimeError("Calibration component provenance hash mismatch")
        if cohort.get("full_qa_and_ground_truth_coverage") is not True or cohort.get("component_overlap") != 0:
            raise RuntimeError("Calibration cohort verification failed")
    if expected_manifest is not None:
        expected = hashlib.sha256(Path(expected_manifest).read_bytes()).hexdigest()
        recorded = (payload.get("application_data_provenance") or {}).get("manifest_sha256")
        if recorded != expected:
            raise RuntimeError("Calibration application cohort mismatch")
    return {
        "path": str(calibration_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "scope": payload.get("scope"),
        "protocol_version": int(payload["protocol_version"]),
        "calibration_split": payload.get("calibration_split"),
        "method": payload.get("calibration_method"),
        "target_sensitivity": payload.get("target_sensitivity"),
        "platt_slope": slope,
        "platt_intercept": intercept,
        "fracture_yes_threshold": threshold,
    }
