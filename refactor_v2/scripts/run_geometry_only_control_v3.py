#!/usr/bin/env python3
"""Question + detector-geometry placebo baseline with no image pixels.

This control quantifies how much VQA/fracture performance can be explained by
templated question text and proposal existence/count/geometry alone.  It must
never be described as visual grounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.endpoints import is_canonical_fracture_presence_question
from src.provenance import validate_detector_proposals
from src.utils import compute_metrics, normalize_answer


def geometry(boxes, size):
    boxes = boxes or []
    if not boxes:
        return [0.0] * 10
    array = np.asarray([box[:4] for box in boxes], dtype=np.float64)
    widths = np.maximum(array[:, 2] - array[:, 0], 0.0)
    heights = np.maximum(array[:, 3] - array[:, 1], 0.0)
    max_x = max(float(size.get("width", 0)), 1.0)
    max_y = max(float(size.get("height", 0)), 1.0)
    areas = widths * heights / (max_x * max_y)
    return [
        1.0, float(len(boxes)), float(areas.sum()), float(areas.max()), float(areas.mean()),
        float((widths / np.maximum(heights, 1e-6)).mean()),
        float(((array[:, 0] + array[:, 2]) / (2 * max_x)).mean()),
        float(((array[:, 1] + array[:, 3]) / (2 * max_y)).mean()),
        float((widths / max_x).mean()), float((heights / max_y).mean()),
    ]


def load_rows(data_root: Path, split: str):
    return json.loads((data_root / f"vqa_{split}.json").read_text(encoding="utf-8"))


def design(train, rows, proposals, image_metadata):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    text_train = vectorizer.fit_transform([row["question"] for row in train])
    text_rows = vectorizer.transform([row["question"] for row in rows])
    scaler = StandardScaler()
    geo_train = scaler.fit_transform([
        geometry(proposals.get(row["image_id"]), image_metadata.get(row["image_id"], {})) for row in train
    ])
    geo_rows = scaler.transform([
        geometry(proposals.get(row["image_id"]), image_metadata.get(row["image_id"], {})) for row in rows
    ])
    return hstack([text_train, csr_matrix(geo_train)]), hstack([text_rows, csr_matrix(geo_rows)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance = validate_detector_proposals(args.boxes)
    proposals = json.loads(args.boxes.read_text(encoding="utf-8"))
    provenance_payload = json.loads(args.boxes.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    image_metadata = provenance_payload["images"]
    train = load_rows(args.data_root, "train")
    cohorts = {split: load_rows(args.data_root, split) for split in ("model_val", "calibration", "test")}

    x_train, x_all = design(
        train, [row for split in cohorts.values() for row in split], proposals, image_metadata
    )
    answers = np.asarray([normalize_answer(row["answer"]) for row in train])
    vqa = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    vqa.fit(x_train, answers)
    all_predictions = vqa.predict(x_all)

    fracture_train = [i for i, row in enumerate(train)
                      if is_canonical_fracture_presence_question(row["question"])
                      and normalize_answer(row["answer"]) in ("yes", "no")]
    fracture = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    fracture.fit(x_train[fracture_train], [answers[i] == "yes" for i in fracture_train])
    fracture_prob = fracture.predict_proba(x_all)[:, list(fracture.classes_).index(True)]

    report = {
        "protocol_version": 3,
        "control": "question + detector geometry only; zero image pixels",
        "detector_provenance": provenance,
        "features": ["question TF-IDF", "proposal presence/count", "area", "aspect", "center", "size"],
        "splits": {},
    }
    offset = 0
    for split, rows in cohorts.items():
        predictions = all_predictions[offset:offset + len(rows)]
        probabilities = fracture_prob[offset:offset + len(rows)]
        records = []
        for row, prediction, probability in zip(rows, predictions, probabilities):
            pred = str(prediction)
            record = {"image_id": row["image_id"], "question": row["question"],
                      "answer_type": row["answer_type"], "gt": row["answer"], "pred": pred}
            if is_canonical_fracture_presence_question(row["question"]):
                record["p_yes"] = float(probability)
                record["pred"] = "yes" if probability >= 0.5 else "no"
            records.append(record)
        report["splits"][split] = {"metrics": compute_metrics(records), "records": records}
        offset += len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: value["metrics"] for split, value in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
