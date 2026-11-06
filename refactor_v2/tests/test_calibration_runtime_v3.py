import json
import hashlib

import pytest

from src.calibration import apply_platt_probability, load_calibration_artifact


def test_platt_identity_and_monotonicity():
    assert apply_platt_probability(0.2, 1.0, 0.0) == pytest.approx(0.2)
    assert apply_platt_probability(0.8, 1.0, 0.0) == pytest.approx(0.8)
    assert apply_platt_probability(0.2, 2.0, -0.3) < apply_platt_probability(0.8, 2.0, -0.3)


def test_calibration_artifact_is_hashed_and_validated(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "protocol_version": 3,
        "test_used_for_fitting_or_threshold_selection": False,
        "calibration_split": "calibration",
        "calibration_method": "Platt scaling on logit(p_yes)",
        "target_sensitivity": 0.9,
        "platt_slope": 1.2,
        "platt_intercept": -0.1,
        "fracture_yes_threshold": 0.31,
    }), encoding="utf-8")
    artifact = load_calibration_artifact(path)
    assert artifact["fracture_yes_threshold"] == pytest.approx(0.31)
    assert len(artifact["sha256"]) == 64


def test_calibration_artifact_fails_closed_on_leakage(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "protocol_version": 3,
        "test_used_for_fitting_or_threshold_selection": True,
        "platt_slope": 1.0,
        "platt_intercept": 0.0,
        "fracture_yes_threshold": 0.5,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="test-independent"):
        load_calibration_artifact(path)


def test_runtime_calibration_rejects_wrong_checkpoint_and_smoke_production(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"specific checkpoint")
    artifact = {"protocol_version": 3, "test_used_for_fitting_or_threshold_selection": False,
                "platt_slope": 1.0, "platt_intercept": 0.0, "fracture_yes_threshold": 0.3,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "scope": "technical_smoke_train_only"}
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(RuntimeError, match="production artifact"):
        load_calibration_artifact(path, expected_checkpoint=checkpoint)
    load_calibration_artifact(path, expected_checkpoint=checkpoint, allow_technical_smoke=True)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        load_calibration_artifact(path, expected_checkpoint=checkpoint, allow_technical_smoke=True)


def test_evaluator_uses_runtime_frozen_threshold():
    import torch
    from src.evaluate import evaluate_model
    class Model:
        cfg = {"fracture_yes_threshold": 0.3}
        def predict_batch(self, batch, **kwargs):
            return [{"answer": "yes", "answer_type": "CLOSED", "confidence": 0.4,
                     "p_yes": 0.4, "p_yes_raw": 0.4}]
    batch = {"images": [None], "questions": ["Is there a fracture in this X-ray?"],
             "answers": ["yes"], "answer_types": ["CLOSED"], "image_ids": ["a"]}
    metrics, _ = evaluate_model(Model(), [batch], 1, torch.device("cpu"))
    assert metrics["fracture_sensitivity"] == 1.0
