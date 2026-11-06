#!/usr/bin/env python3
"""Production-parity pixel interventions with annotation ROI used only as audit labels.

The VQA model always receives detector proposals.  Ground-truth lesion boxes
are used solely to alter pixels for the four causal audit conditions and never
as an inference prompt.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.calibration import load_calibration_artifact
from src.data import collate_fn
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.preprocessing import canonicalize_raster_and_boxes
from src.provenance import validate_detector_proposals
from src.utils import compute_metrics, get_device, normalize_answer


CONDITIONS = ("original", "lesion_mask", "matched_nonlesion_mask", "background_replacement")


def box_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2, *_ in boxes or []:
        xa, xb = sorted((max(0, int(round(x1))), min(width, int(round(x2)))))
        ya, yb = sorted((max(0, int(round(y1))), min(height, int(round(y2)))))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def matched_boxes(boxes, width, height):
    shifted = []
    original_mask = box_mask(boxes, width, height)
    for x1, y1, x2, y2, *_ in boxes:
        # Match the actual rasterized mask, not fractional annotation widths:
        # shifting a fractional box can otherwise gain/lose an entire pixel row.
        x1, x2 = sorted((max(0, min(width, int(round(x1)))), max(0, min(width, int(round(x2))))))
        y1, y2 = sorted((max(0, min(height, int(round(y1)))), max(0, min(height, int(round(y2))))))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
        candidates = []
        for x in (0, max(0, width - box_w), max(0, (width - box_w) // 2)):
            for y in (0, max(0, height - box_h), max(0, (height - box_h) // 2)):
                candidate = [x, y, x + box_w, y + box_h]
                overlap = np.logical_and(box_mask([candidate], width, height), original_mask).sum()
                displacement = ((x + box_w / 2) - (x1 + x2) / 2) ** 2 + ((y + box_h / 2) - (y1 + y2) / 2) ** 2
                candidates.append((int(overlap), -float(displacement), candidate))
        shifted.append(min(candidates)[2])
    return shifted


def intervene(image, lesion_boxes, condition):
    if condition == "original":
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    height, width = array.shape[:2]
    median = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)
    lesion = box_mask(lesion_boxes, width, height)
    if condition == "lesion_mask":
        array[lesion] = median
    elif condition == "matched_nonlesion_mask":
        array[box_mask(matched_boxes(lesion_boxes, width, height), width, height)] = median
    elif condition == "background_replacement":
        replacement = np.broadcast_to(median, array.shape).copy()
        replacement[lesion] = array[lesion]
        array = replacement
    else:
        raise ValueError(condition)
    return Image.fromarray(array)


class InterventionDataset(Dataset):
    def __init__(self, rows, image_root, proposals, condition):
        self.rows = rows
        self.image_root = image_root
        self.proposals = proposals
        self.condition = condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_root / row["image"]) as raw:
            image, lesion_boxes = canonicalize_raster_and_boxes(raw, row.get("boxes") or [])
        image = intervene(image, lesion_boxes, self.condition)
        return {"image": image, "question": row["question"], "answer": row["answer"],
                "answer_type": row["answer_type"], "region": row.get("region"),
                "masks": None, "boxes": self.proposals.get(row["image_id"]) or None,
                "image_id": row["image_id"]}


@torch.no_grad()
def evaluate(model, loader, device):
    rows = []
    model.eval()
    for batch in loader:
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            predictions = model.predict_batch(batch)
        normalized = normalize_batch(batch)
        for pred, answer, answer_type, question, image_id in zip(
                predictions, normalized["answers"], normalized["answer_types"],
                normalized["questions"], normalized["image_ids"]):
            row = {"image_id": image_id, "question": question, "gt": answer,
                   "answer_type": answer_type, "pred": pred["answer"],
                   "confidence": float(pred["confidence"])}
            for key in ("p_yes", "p_yes_raw"):
                if key in pred:
                    row[key] = float(pred[key])
            rows.append(row)
    return rows


def paired_summary(original, changed):
    if len(original) != len(changed):
        raise RuntimeError("Intervention cohorts do not align")
    by_image = defaultdict(list)
    flips = 0
    fracture_probability_deltas = []
    for left, right in zip(original, changed):
        key = (left["image_id"], left["question"], left["gt"])
        if key != (right["image_id"], right["question"], right["gt"]):
            raise RuntimeError("Intervention row order mismatch")
        left_correct = normalize_answer(left["pred"]) == normalize_answer(left["gt"])
        right_correct = normalize_answer(right["pred"]) == normalize_answer(right["gt"])
        by_image[left["image_id"]].append(float(right_correct) - float(left_correct))
        flips += left["pred"] != right["pred"]
        if is_canonical_fracture_presence_question(left["question"]) and "p_yes" in left and "p_yes" in right:
            fracture_probability_deltas.append(abs(right["p_yes"] - left["p_yes"]))
    return {
        "accuracy_change_pp": 100.0 * np.mean([np.mean(values) for values in by_image.values()]),
        "prediction_flip_rate": flips / max(len(original), 1),
        "fracture_mean_abs_probability_change": float(np.mean(fracture_probability_deltas))
        if fracture_probability_deltas else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    detector_provenance = validate_detector_proposals(args.boxes)
    proposals = json.loads(args.boxes.read_text(encoding="utf-8"))
    rows = json.loads(args.records.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("boxes")]
    override = {}
    calibration_provenance = None
    if args.calibration:
        calibration_provenance = load_calibration_artifact(args.calibration, expected_checkpoint=args.checkpoint)
        override = {"fracture_platt_slope": calibration_provenance["platt_slope"],
                    "fracture_platt_intercept": calibration_provenance["platt_intercept"],
                    "fracture_yes_threshold": calibration_provenance["fracture_yes_threshold"]}
    device = get_device()
    model = BoneVQAModel.load(str(args.checkpoint), device=str(device), cfg_override=override or None)
    results = {}
    predictions = {}
    for condition in CONDITIONS:
        dataset = InterventionDataset(rows, args.image_root, proposals, condition)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        condition_rows = evaluate(model, loader, device)
        predictions[condition] = condition_rows
        results[condition] = compute_metrics(condition_rows,
                                             fracture_threshold=float(model.cfg.get("fracture_yes_threshold", 0.5)))
    comparisons = {
        condition: paired_summary(predictions["original"], predictions[condition])
        for condition in CONDITIONS[1:]
    }
    report = {"protocol_version": 3,
              "annotation_roi_role": "pixel-intervention audit only; model prompts are detector-derived",
              "n_annotated_images": len({row["image_id"] for row in rows}), "n_qa": len(rows),
              "detector_provenance": detector_provenance,
              "calibration_provenance": calibration_provenance,
              "metrics": results, "versus_original": comparisons, "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": results, "versus_original": comparisons}, indent=2))


if __name__ == "__main__":
    main()
