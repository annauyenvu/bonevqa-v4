from scripts.clinical_fracture_calibration_v3 import apply_platt, fit_platt, select_sensitivity_threshold
from scripts.clinical_fracture_calibration_v3 import validate_sealed_cohorts
import copy
import hashlib
import json
import pytest


def _sealed_pair(tmp_path, same_component=False):
    artifacts, payloads = {}, []
    for role, image in (("calibration", "a"), ("test", "b")):
        source = [{"image_id": image, "question": "Is there a fracture in this X-ray?",
                   "answer_type": "CLOSED", "answer": "yes"}]
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(source))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[path.name] = digest
        record = dict(source[0])
        record["gt"] = record.pop("answer")
        record["p_yes"] = 0.6
        payloads.append({"split": role, "evaluation_scope": "technical_smoke_train_only", "records": [record], "sealed_data_provenance": {
            "manifest": str(path), "manifest_sha256": digest}})
    components = tmp_path / "components.json"
    components.write_text(json.dumps({"components": [["a", "b"]] if same_component else [["a"], ["b"]]}))
    artifacts[components.name] = hashlib.sha256(components.read_bytes()).hexdigest()
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "artifacts": artifacts}))
    for payload in payloads:
        payload["sealed_data_provenance"].update(seal=str(seal), seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest())
    return payloads, components, seal


def test_calibration_sealed_cohort_full_coverage(tmp_path):
    payloads, components, seal = _sealed_pair(tmp_path)
    result = validate_sealed_cohorts(*payloads, tmp_path, components, seal)
    assert result["component_overlap"] == 0
    assert result["counts"]["test"]["fracture_images"] == 1


@pytest.mark.parametrize("mutation", ["gt", "missing", "duplicate", "manifest_bytes"])
def test_calibration_rejects_forged_or_partial_records(tmp_path, mutation):
    payloads, components, seal = _sealed_pair(tmp_path)
    if mutation == "gt":
        payloads[0]["records"][0]["gt"] = "no"
    elif mutation == "missing":
        payloads[0]["records"] = []
    elif mutation == "duplicate":
        payloads[0]["records"].append(copy.deepcopy(payloads[0]["records"][0]))
    else:
        (tmp_path / "calibration.json").write_text("[]")
    with pytest.raises(ValueError):
        validate_sealed_cohorts(*payloads, tmp_path, components, seal)


def test_calibration_rejects_cross_image_duplicate_component(tmp_path):
    payloads, components, seal = _sealed_pair(tmp_path, same_component=True)
    with pytest.raises(ValueError, match="share duplicate components"):
        validate_sealed_cohorts(*payloads, tmp_path, components, seal)


def _rows():
    return [
        {"image_id": "p1", "gt": "yes", "p_yes": 0.8},
        {"image_id": "p2", "gt": "yes", "p_yes": 0.6},
        {"image_id": "p3", "gt": "yes", "p_yes": 0.4},
        {"image_id": "n1", "gt": "no", "p_yes": 0.3},
        {"image_id": "n2", "gt": "no", "p_yes": 0.2},
        {"image_id": "n3", "gt": "no", "p_yes": 0.1},
    ]


def test_platt_and_target_sensitivity_are_calibration_only_operations():
    rows = _rows()
    slope, intercept = fit_platt(rows)
    calibrated = apply_platt(rows, slope, intercept)
    threshold = select_sensitivity_threshold(calibrated, 2 / 3)
    positives = [r["p_yes"] >= threshold for r in calibrated if r["gt"] == "yes"]
    assert sum(positives) / len(positives) >= 2 / 3
    assert slope > 0


def test_saturated_calibration_threshold_matches_runtime_probability_space():
    rows = _rows()
    calibrated = apply_platt(rows, 100.0, -100.0)
    threshold = select_sensitivity_threshold(calibrated, 1.0)
    assert all(row["p_yes"] >= threshold for row in calibrated if row["gt"] == "yes")
