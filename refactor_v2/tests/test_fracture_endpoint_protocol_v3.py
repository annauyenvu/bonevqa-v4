from scripts.calibrate_fracture_threshold_v2 import fracture_rows as threshold_fracture_rows
from scripts.clinical_fracture_calibration_v3 import fracture_rows as clinical_fracture_rows
from src.endpoints import is_canonical_fracture_presence_question
from src.utils import compute_metrics


def test_canonical_fracture_presence_accepts_only_global_templates():
    accepted = [
        "Is there a fracture?",
        "Is there a fracture in this image?",
        "Is there a fracture in this X-ray?",
        " is   there a fracture in this x ray?! ",
        "Is there a fracture in this Xray?",
    ]
    rejected = [
        "Is the hand fractured?",
        "Is the femur fractured?",
        "Is there a fracture in the hand?",
        "Are there multiple fractures?",
        "How many fracture sites are visible?",
        "Where is the fracture located?",
        "Does this X-ray show a fracture?",
        "Is there a fracture in this X-ray and hand?",
    ]

    assert all(is_canonical_fracture_presence_question(q) for q in accepted)
    assert not any(is_canonical_fracture_presence_question(q) for q in rejected)


def test_metrics_do_not_mix_region_or_multiplicity_questions_into_endpoint():
    records = [
        {"question": "Is there a fracture in this X-ray?", "answer_type": "CLOSED",
         "gt": "yes", "pred": "yes", "p_yes": 0.9},
        {"question": "Is there a fracture in this X-ray?", "answer_type": "CLOSED",
         "gt": "no", "pred": "no", "p_yes": 0.1},
        {"question": "Is the hand fractured?", "answer_type": "CLOSED",
         "gt": "no", "pred": "yes", "p_yes": 0.99},
        {"question": "Are there multiple fractures?", "answer_type": "CLOSED",
         "gt": "yes", "pred": "no", "p_yes": 0.01},
    ]

    metrics = compute_metrics(records)

    assert metrics["n_fracture_presence"] == 2
    assert metrics["fracture_sensitivity"] == 1.0
    assert metrics["fracture_specificity"] == 1.0
    assert metrics["fracture_auroc"] == 1.0


def test_calibration_scripts_share_the_same_strict_endpoint_filter():
    records = [
        {"question": "Is there a fracture?", "answer_type": "CLOSED",
         "gt": "yes", "pred": "yes", "p_yes": 0.8},
        {"question": "Is there a fracture in this image?", "answer_type": "CLOSED",
         "gt": "no", "pred": "no", "p_yes": 0.2},
        {"question": "Is the hand fractured?", "answer_type": "CLOSED",
         "gt": "no", "pred": "no", "p_yes": 0.1},
        {"question": "Where is the fracture located?", "answer_type": "OPEN",
         "gt": "hand", "pred": "hand", "p_yes": 0.5},
    ]
    payload = {"records": records}

    assert threshold_fracture_rows(payload) == records[:2]
    assert clinical_fracture_rows(payload) == records[:2]
