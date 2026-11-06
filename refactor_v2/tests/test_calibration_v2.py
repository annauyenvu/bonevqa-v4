from scripts.calibrate_fracture_threshold_v2 import select_youden_threshold


def test_threshold_is_selected_from_validation_scores():
    rows = [
        {"gt": "yes", "p_yes": 0.8},
        {"gt": "yes", "p_yes": 0.7},
        {"gt": "no", "p_yes": 0.4},
        {"gt": "no", "p_yes": 0.2},
    ]
    threshold = select_youden_threshold(rows)
    assert 0.4 < threshold <= 0.7
