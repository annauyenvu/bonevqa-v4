#!/usr/bin/env python3
"""Deterministically split the old clean validation set by duplicate component.

The sealed test set is never read or modified.  Positive and negative image
groups are balanced separately so both model-selection and calibration cohorts
contain useful fracture endpoints.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.endpoints import is_canonical_fracture_presence_question


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".evaluation_runtime/data/fracatlas_component_clean"
SEED = 20261015


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = DATA / "vqa_val.json"
    component_path = DATA / "component_split_manifest.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    component_manifest = json.loads(component_path.read_text(encoding="utf-8"))
    image_rows: dict[str, list[dict]] = {}
    for row in rows:
        image_rows.setdefault(row["image_id"], []).append(row)
    val_images = set(image_rows)

    groups: list[list[str]] = []
    assigned: set[str] = set()
    for component in component_manifest["components"]:
        members = sorted(val_images.intersection(component))
        if members:
            groups.append(members)
            assigned.update(members)
    groups.extend([[image_id] for image_id in sorted(val_images - assigned)])

    positive, negative = [], []
    for group in groups:
        is_positive = any(int(image_rows[i][0].get("fractured", 0)) == 1 for i in group)
        (positive if is_positive else negative).append(group)

    rng = random.Random(SEED)
    calibration_groups: list[list[str]] = []
    model_groups: list[list[str]] = []
    for stratum in (positive, negative):
        rng.shuffle(stratum)
        target = sum(len(group) for group in stratum) / 2.0
        calibration_count = 0
        for group in stratum:
            if calibration_count < target:
                calibration_groups.append(group)
                calibration_count += len(group)
            else:
                model_groups.append(group)

    calibration_images = {image for group in calibration_groups for image in group}
    model_images = {image for group in model_groups for image in group}
    if calibration_images & model_images or calibration_images | model_images != val_images:
        raise RuntimeError("Validation/calibration image partition is invalid")

    model_rows = [row for row in rows if row["image_id"] in model_images]
    calibration_rows = [row for row in rows if row["image_id"] in calibration_images]
    model_path = DATA / "vqa_model_val.json"
    calibration_path = DATA / "vqa_calibration.json"
    model_path.write_text(json.dumps(model_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(selected_rows: list[dict]) -> dict:
        images = {row["image_id"] for row in selected_rows}
        fractured = {row["image_id"] for row in selected_rows if int(row.get("fractured", 0)) == 1}
        fracture_presence = [
            row for row in selected_rows
            if row.get("answer_type") == "CLOSED"
            and is_canonical_fracture_presence_question(row.get("question"))
        ]
        return {
            "images": len(images),
            "fractured_images": len(fractured),
            "qa": len(selected_rows),
            "fracture_presence_qa": len(fracture_presence),
            "fracture_presence_positive": sum(str(row.get("answer", "")).lower() == "yes" for row in fracture_presence),
        }

    manifest = {
        "version": 3,
        "seed": SEED,
        "policy": "component-preserving, fracture-stratified split of the former validation cohort only",
        "source": str(source),
        "source_sha256": digest(source),
        "component_manifest_sha256": digest(component_path),
        "test_accessed": False,
        "model_val": summary(model_rows),
        "calibration": summary(calibration_rows),
        "image_overlap": len(model_images & calibration_images),
        "model_val_images": sorted(model_images),
        "calibration_images": sorted(calibration_images),
    }
    out = DATA / "validation_calibration_manifest_v3.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("model_val", "calibration", "image_overlap")}, indent=2))


if __name__ == "__main__":
    main()
