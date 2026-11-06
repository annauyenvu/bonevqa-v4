from src.utils import compute_metrics


def test_fracture_endpoint_exposes_sensitivity_and_specificity():
    records = [
        {"question": "Is there a fracture in this image?", "answer_type": "CLOSED", "gt": "yes", "pred": "no"},
        {"question": "Is there a fracture in this image?", "answer_type": "CLOSED", "gt": "no", "pred": "no"},
    ]
    metrics = compute_metrics(records)
    assert metrics["fracture_sensitivity"] == 0.0
    assert metrics["fracture_specificity"] == 1.0
    assert metrics["fracture_balanced_accuracy"] == 0.5


def test_fracture_threshold_uses_recorded_probability():
    records = [
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "yes", "pred": "no", "p_yes": 0.45},
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "no", "pred": "no", "p_yes": 0.20},
    ]
    raw = compute_metrics(records)
    calibrated = compute_metrics(records, fracture_threshold=0.4)
    assert raw["fracture_sensitivity"] == 0.0
    assert calibrated["fracture_sensitivity"] == 1.0
    assert calibrated["fracture_specificity"] == 1.0
    assert calibrated["fracture_yes_threshold"] == 0.4


def test_fracture_probability_metrics_are_reported():
    records = [
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "yes", "pred": "yes", "p_yes": 0.9},
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "yes", "pred": "no", "p_yes": 0.6},
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "no", "pred": "no", "p_yes": 0.2},
        {"question": "Is there a fracture?", "answer_type": "CLOSED", "gt": "no", "pred": "no", "p_yes": 0.1},
    ]
    metrics = compute_metrics(records)
    assert metrics["fracture_auroc"] == 1.0
    assert metrics["fracture_auprc"] == 1.0
    assert metrics["fracture_prevalence"] == 0.5
    assert 0.0 <= metrics["fracture_brier"] <= 1.0
    assert 0.0 <= metrics["fracture_ece10"] <= 1.0
