#!/usr/bin/env python3
"""Lesion, matched non-lesion, and background interventions for Detector-V."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from runtime import PROJECT, ROOT, configure


CONDITIONS = ["original", "lesion_mask", "matched_nonlesion_mask", "background_replacement"]


def shifted_boxes(lesions, width, height):
    output = []
    for box in lesions:
        x1, y1, x2, y2 = box; bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        candidates = []
        for fx, fy in ((.02, .02), (.98, .02), (.02, .98), (.98, .98), (.5, .02), (.5, .98), (.02, .5), (.98, .5)):
            xa = max(0, min(width - bw, fx * width - bw / 2)); ya = max(0, min(height - bh, fy * height - bh / 2))
            candidate = [xa, ya, xa + bw, ya + bh]
            overlap = 0.0
            for lesion in lesions:
                ix1, iy1 = max(candidate[0], lesion[0]), max(candidate[1], lesion[1])
                ix2, iy2 = min(candidate[2], lesion[2]), min(candidate[3], lesion[3])
                overlap += max(0, ix2 - ix1) * max(0, iy2 - iy1)
            candidates.append((overlap, candidate))
        output.append(min(candidates, key=lambda value: value[0])[1])
    return output


def box_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(round(x1))), max(0, int(round(y1)))
        xb, yb = min(width, int(round(x2))), min(height, int(round(y2)))
        mask[ya:yb, xa:xb] = True
    return mask


def transform(image: Image.Image, lesions, condition):
    if condition == "original": return image
    array = np.asarray(image).copy()
    height, width = array.shape[:2]
    median = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)
    lesion_mask = box_mask(lesions, width, height)
    if condition == "lesion_mask":
        array[lesion_mask] = median
    elif condition == "matched_nonlesion_mask":
        nonlesion = box_mask(shifted_boxes(lesions, width, height), width, height)
        array[nonlesion] = median
    elif condition == "background_replacement":
        replaced = np.broadcast_to(median, array.shape).copy()
        replaced[lesion_mask] = array[lesion_mask]
        array = replaced
    else:
        raise ValueError(condition)
    return Image.fromarray(array)


class InterventionDataset(Dataset):
    def __init__(self, records, images, roi_index, condition):
        self.records, self.images, self.roi_index, self.condition = records, images, roi_index, condition
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.images[row["image_id"]]["path"]) as source:
            meta = self.images[row["image_id"]]
            expected_size = (meta["width"], meta["height"])
            image = (source if source.size == expected_size else ImageOps.exif_transpose(source)).convert("RGB")
        image = transform(image, row["boxes"], self.condition)
        roi = self.roi_index[row["image_id"]]
        with np.load(roi["path"]) as data:
            masks = data["masks"].copy()
        return {"image": image, "question": row["question"], "answer": row["answer"],
                "answer_type": row["answer_type"], "region": row.get("region"), "image_id": row["image_id"],
                "masks": masks, "boxes": None}


@torch.no_grad()
def evaluate_subset(engine, model, loader):
    model.eval(); records = []
    for batch in loader:
        predictions = model.predict_batch(batch); normalized = engine.normalize_batch(batch)
        for prediction, answer, answer_type, question, image_id in zip(predictions, normalized["answers"], normalized["answer_types"], normalized["questions"], normalized["image_ids"]):
            confidence = float(prediction["confidence"])
            records.append({"image_id": image_id, "question": question, "gt": answer, "pred": prediction["answer"],
                            "answer_type": answer_type, "pred_type": prediction["answer_type"],
                            "confidence": confidence if np.isfinite(confidence) else 0.0,
                            "confidence_nonfinite": not np.isfinite(confidence)})
    return engine.richer_metrics(records), records


def correctness(engine, row):
    return (engine.normalize_yes_no(row["pred"]) == engine.normalize_yes_no(row["gt"])) if row["answer_type"] == "CLOSED" else (engine.normalize_answer(row["pred"]) == engine.normalize_answer(row["gt"]))


def compare(engine, original, changed):
    assert len(original) == len(changed)
    groups = defaultdict(list); flips = family_rows = family_flips = 0
    for left, right in zip(original, changed):
        assert (left["image_id"], left["question"], left["gt"]) == (right["image_id"], right["question"], right["gt"])
        delta = float(correctness(engine, right)) - float(correctness(engine, left))
        groups[left["image_id"]].append(delta)
        flips += left["pred"] != right["pred"]
        if engine.family(left["question"]) == "fracture presence":
            family_rows += 1; family_flips += left["pred"] != right["pred"]
    ids = sorted(groups); nums = np.asarray([sum(groups[i]) for i in ids]); dens = np.asarray([len(groups[i]) for i in ids])
    rng = np.random.default_rng(20261011); draws = rng.integers(len(ids), size=(10000, len(ids)))
    values = 100 * nums[draws].sum(1) / dens[draws].sum(1)
    return {"accuracy_change_pp": 100 * nums.sum() / dens.sum(), "ci95_low_pp": float(np.quantile(values, .025)),
            "ci95_high_pp": float(np.quantile(values, .975)), "prediction_flip_rate": flips / len(original),
            "fracture_presence_flip_rate": family_flips / family_rows}


def run_cohort(common, engine, model, output, cohort_name, records, images, roi_index):
    cohort_output = output / cohort_name
    cohort_output.mkdir(parents=True, exist_ok=True)
    annotated_ids = {row["image_id"] for row in records}
    summaries = {}; predictions = {}
    for condition in CONDITIONS:
        dataset = InterventionDataset(records, images, roi_index, condition)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=engine.collate_fn)
        metrics, rows = evaluate_subset(engine, model, loader)
        summaries[condition] = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        predictions[condition] = rows
        common.write(cohort_output / f"{condition}_predictions.json", rows)
        common.write(cohort_output / f"{condition}_metrics.json", metrics)
        print("GROUNDING", cohort_name, condition, summaries[condition], flush=True)
    comparisons = {condition: compare(engine, predictions["original"], predictions[condition]) for condition in CONDITIONS[1:]}
    result = {"n_annotated_images": len(annotated_ids), "n_qa": len(records),
              "conditions": summaries, "versus_original": comparisons}
    common.write(cohort_output / "results.json", result)
    return result


def main() -> None:
    common, engine, specs = configure(); common.setup(42); common.verify_lock()
    output = ROOT / "grounding_interventions"; output.mkdir(exist_ok=True)
    checkpoint = ROOT / "detector/v/seed_42/checkpoint_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = engine.make_model(payload["config"], payload["answer_vocab"], checkpoint)
    clean_test = common.read(ROOT / "protocol/test_manifest.json")
    clean_records = [row for row in clean_test if row.get("boxes")]
    clean_images = {row["image_id"]: row for row in common.read(ROOT / "protocol/image_content_manifest.json")}
    clean_roi = {row["image_id"]: row for row in common.read(ROOT / "roi_bank/detector/index.json")}
    historical_root = PROJECT / "exp4_multiseed"
    historical_test = common.read(historical_root / "protocol/test_manifest.json")
    historical_records = [row for row in historical_test if row.get("boxes")]
    historical_images = {row["image_id"]: row for row in common.read(historical_root / "protocol/image_content_manifest.json")}
    historical_roi = {row["image_id"]: row for row in common.read(historical_root / "roi_bank/detector/index.json")}
    cohorts = {
        "component_clean_heldout": run_cohort(common, engine, model, output, "component_clean_heldout", clean_records, clean_images, clean_roi),
        "original_108": run_cohort(common, engine, model, output, "original_108", historical_records, historical_images, historical_roi),
    }
    common.write(output / "results.json", {"checkpoint": str(checkpoint), "cohorts": cohorts,
                                            "warning": "original_108 is a historical cohort; some images may belong to training after component reassignment. Use component_clean_heldout for unbiased accuracy."})
    lines = ["# Detector-V lesion grounding interventions", "",
             "Seed-42 component-clean checkpoint. The component-clean held-out cohort is the primary unbiased analysis. The historical 108-image cohort is reported for direct comparability but is not used as a clean accuracy estimate because component reassignment can place some of those images in training.", ""]
    for cohort_name, cohort in cohorts.items():
        lines.extend([f"## {cohort_name}", "", f"{cohort['n_annotated_images']} annotated images / {cohort['n_qa']} QA.", "",
                      "| Condition | Overall | Closed | Open exact | Change vs original | Flip rate |",
                      "|---|---:|---:|---:|---:|---:|"])
        for condition in CONDITIONS:
            summary = cohort["conditions"][condition]
            if condition == "original": change = "-"; flip = "-"
            else:
                result = cohort["versus_original"][condition]
                change = f"{result['accuracy_change_pp']:+.2f} pp [{result['ci95_low_pp']:+.2f}, {result['ci95_high_pp']:+.2f}]"
                flip = f"{100*result['prediction_flip_rate']:.2f}%"
            lines.append(f"| {condition} | {100*summary['overall_acc']:.2f}% | {100*summary['closed_acc']:.2f}% | {100*summary['open_acc']:.2f}% | {change} | {flip} |")
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
