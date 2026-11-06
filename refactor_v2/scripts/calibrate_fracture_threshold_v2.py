#!/usr/bin/env python3
"""Select on validation only, then apply the frozen threshold to test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_curve

from src.endpoints import is_canonical_fracture_presence_question
from src.utils import compute_metrics, normalize_yes_no


def fracture_rows(payload: dict) -> list[dict]:
    return [
        row for row in payload["records"]
        if row.get("answer_type") == "CLOSED"
        and is_canonical_fracture_presence_question(row.get("question"))
    ]


def select_youden_threshold(rows: list[dict]) -> float:
    if not rows or any(row.get("p_yes") is None for row in rows):
        raise ValueError("Every validation fracture-presence record must contain p_yes")
    labels = [1 if normalize_yes_no(row["gt"]) == "yes" else 0 for row in rows]
    scores = [float(row["p_yes"]) for row in rows]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = max(
        range(len(thresholds)),
        key=lambda idx: (tpr[idx] - fpr[idx], -abs(float(thresholds[idx]) - 0.5)),
    )
    return float(thresholds[best])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))
    threshold = select_youden_threshold(fracture_rows(validation))
    report = {
        "selection_split": validation.get("split"),
        "application_split": test.get("split"),
        "threshold_policy": "Youden J selected on validation only",
        "fracture_yes_threshold": threshold,
        "validation_default": compute_metrics(validation["records"]),
        "validation_calibrated": compute_metrics(validation["records"], threshold),
        "test_default": compute_metrics(test["records"]),
        "test_calibrated": compute_metrics(test["records"], threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
