#!/usr/bin/env python3
"""Balanced annotation-lesion versus geometry-matched placebo ROI inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[2]
EXP4 = PROJECT / "exp4_multiseed"
sys.path[:0] = [str(EXP4 / "code"), str(PROJECT / ".evaluation_runtime/deps")]
import common
import engine
from src.data import collate_fn


OUT = PROJECT / "mandatory_experiments_20261011/matched_placebo"
CHECKPOINT = PROJECT / ".evaluation_runtime/checkpoints/chiL_hotro_no_visual_prompt_latent_prompt/best.pt"


def seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def candidate_boxes(box, width, height):
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    candidates = []
    for fx, fy in ((.02, .02), (.98, .02), (.02, .98), (.98, .98), (.5, .02), (.5, .98), (.02, .5), (.98, .5), (.33, .33), (.67, .33), (.33, .67), (.67, .67)):
        cx, cy = fx * width, fy * height
        candidate = [max(0, min(width - bw, cx - bw / 2)), max(0, min(height - bh, cy - bh / 2)), 0, 0]
        candidate[2], candidate[3] = candidate[0] + bw, candidate[1] + bh
        candidates.append(candidate)
    return candidates


def shift_off_lesion(box, lesions, width, height):
    candidates = []
    for candidate in candidate_boxes(box, width, height):
        overlap = 0.0
        for lesion in lesions:
            ix1, iy1 = max(candidate[0], lesion[0]), max(candidate[1], lesion[1])
            ix2, iy2 = min(candidate[2], lesion[2]), min(candidate[3], lesion[3])
            overlap += max(0, ix2 - ix1) * max(0, iy2 - iy1)
        candidates.append((overlap, candidate))
    return min(candidates, key=lambda value: value[0])[1]


def rectangle_masks(boxes, width, height):
    masks = []
    for x1, y1, x2, y2 in boxes[:3]:
        mask = np.zeros((height, width), dtype=np.uint8)
        xa, ya = max(0, int(round(x1))), max(0, int(round(y1)))
        xb, yb = min(width, int(round(x2))), min(height, int(round(y2)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 1
            masks.append(mask)
    return np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)


def build_plans(records):
    by_image = {}
    for row in records:
        by_image.setdefault(row["image_id"], row)
    donors = []
    for row in by_image.values():
        for x1, y1, x2, y2 in row.get("boxes") or []:
            donors.append([(x2 - x1) / row["width"], (y2 - y1) / row["height"]])
    plans = {}
    for image_id, row in sorted(by_image.items()):
        width, height = row["width"], row["height"]
        lesions = row.get("boxes") or []
        if lesions:
            lesion_boxes = lesions
            placebo_boxes = [shift_off_lesion(box, lesions, width, height) for box in lesions]
        else:
            donor = donors[seed_for(image_id) % len(donors)]
            bw, bh = donor[0] * width, donor[1] * height
            candidates = candidate_boxes([0, 0, bw, bh], width, height)
            rng = np.random.default_rng(seed_for(image_id))
            order = rng.permutation(len(candidates))
            lesion_boxes = [candidates[int(order[0])]]
            placebo_boxes = [candidates[int(order[1])]]
        plans[image_id] = {"lesion_or_matched_negative": lesion_boxes, "placebo_matched": placebo_boxes}
    return plans


class RoiDataset(Dataset):
    def __init__(self, records, image_paths, plans, condition):
        self.records, self.image_paths, self.plans, self.condition = records, image_paths, plans, condition

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.image_paths[row["image_id"]]) as source:
            expected_size = (row["width"], row["height"])
            image = (source if source.size == expected_size else ImageOps.exif_transpose(source)).convert("RGB")
        boxes = self.plans[row["image_id"]][self.condition]
        masks = rectangle_masks(boxes, image.width, image.height)
        return {"image": image, "question": row["question"], "answer": row["answer"],
                "answer_type": row["answer_type"], "region": row.get("region"),
                "image_id": row["image_id"], "masks": masks, "boxes": None}


def image_bootstrap(engine, left, right, allowed_ids=None, reps=10000, seed=20261011):
    groups = defaultdict(list)
    for a, b in zip(left, right):
        assert (a["image_id"], a["question"], a["gt"]) == (b["image_id"], b["question"], b["gt"])
        if a["answer_type"] == "CLOSED" and (allowed_ids is None or a["image_id"] in allowed_ids):
            ca = engine.normalize_yes_no(a["pred"]) == engine.normalize_yes_no(a["gt"])
            cb = engine.normalize_yes_no(b["pred"]) == engine.normalize_yes_no(b["gt"])
            groups[a["image_id"]].append(float(ca) - float(cb))
    ids = sorted(groups)
    nums = np.asarray([sum(groups[image_id]) for image_id in ids]); dens = np.asarray([len(groups[image_id]) for image_id in ids])
    rng = np.random.default_rng(seed); draws = rng.integers(len(ids), size=(reps, len(ids)))
    values = nums[draws].sum(1) / dens[draws].sum(1)
    return {"difference_pp": 100 * nums.sum() / dens.sum(), "ci95_low_pp": 100 * float(np.quantile(values, .025)),
            "ci95_high_pp": 100 * float(np.quantile(values, .975)), "n_images": len(ids),
            "n_closed_qa": int(dens.sum())}


def plan_audit(plans, records):
    by_image = {row["image_id"]: row for row in records}
    positive_ids = {image_id for image_id, row in by_image.items() if row.get("fractured") == 1}
    negative_ids = set(by_image) - positive_ids
    for image_id, plan in plans.items():
        left, right = plan["lesion_or_matched_negative"], plan["placebo_matched"]
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert abs((a[2] - a[0]) - (b[2] - b[0])) < 1e-6
            assert abs((a[3] - a[1]) - (b[3] - b[1])) < 1e-6
    return {"positive_images": len(positive_ids), "negative_images": len(negative_ids),
            "roi_present_both_conditions": len(plans), "geometry_exactly_matched_images": len(plans)}


def main() -> None:
    common.setup(42)
    common.verify_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    records = common.read(EXP4 / "protocol/test_manifest.json")
    image_paths = {row["image_id"]: row["path"] for row in common.read(EXP4 / "protocol/image_content_manifest.json")}
    plans = build_plans(records)
    common.write(OUT / "roi_plans.json", plans)
    audit = plan_audit(plans, records)
    common.write(OUT / "roi_plan_audit.json", audit)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    cfg["llm_dtype"] = "float32"; cfg["sam"]["sam_med2d_ckpt"] = None; cfg["sam"]["cache_dir"] = str(OUT / "unused")
    model = engine.make_model(cfg, payload["answer_vocab"], CHECKPOINT)
    results = {}
    for condition in ("lesion_or_matched_negative", "placebo_matched"):
        dataset = RoiDataset(records, image_paths, plans, condition)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
        metrics, predictions = engine.evaluate(model, loader, "test")
        engine.csvwrite(OUT / f"{condition}_predictions.csv", predictions)
        common.write(OUT / f"{condition}_predictions.json", predictions)
        common.write(OUT / f"{condition}_metrics.json", metrics)
        results[condition] = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        print("PLACEBO CONDITION COMPLETE", condition, results[condition], flush=True)
    left = common.read(OUT / "lesion_or_matched_negative_predictions.json")
    right = common.read(OUT / "placebo_matched_predictions.json")
    positive_ids = {row["image_id"] for row in records if row.get("fractured") == 1}
    negative_ids = {row["image_id"] for row in records if row.get("fractured") == 0}
    contrast = image_bootstrap(engine, left, right)
    positive_contrast = image_bootstrap(engine, left, right, positive_ids, seed=20261012)
    negative_contrast = image_bootstrap(engine, left, right, negative_ids, seed=20261013)
    common.write(OUT / "results.json", {"checkpoint": str(CHECKPOINT), "conditions": results,
                                         "closed_lesion_minus_placebo": contrast,
                                         "positive_closed_difference": positive_contrast,
                                         "negative_closed_difference": negative_contrast,
                                         "roi_plan_audit": audit,
                                         "interpretation": "fixed historical checkpoint; identical ROI presence/count/size; positive images use lesion versus shifted placebo, negative images use two position-distinct matched placebos"})
    report = ["# Matched placebo ROI control", "",
              "Both conditions provide ROI to every image with exactly matched ROI count and size. Positive images receive either lesion boxes or shifted non-lesion boxes; negative images receive two position-distinct boxes generated from the same lesion-size donor.", "",
              "| Condition | Closed | Overall | Open exact |", "|---|---:|---:|---:|",
              f"| Lesion or matched-negative ROI | {100*results['lesion_or_matched_negative']['closed_acc']:.2f}% | {100*results['lesion_or_matched_negative']['overall_acc']:.2f}% | {100*results['lesion_or_matched_negative']['open_acc']:.2f}% |",
              f"| Matched placebo ROI | {100*results['placebo_matched']['closed_acc']:.2f}% | {100*results['placebo_matched']['overall_acc']:.2f}% | {100*results['placebo_matched']['open_acc']:.2f}% |", "",
              f"Paired CLOSED all-image difference: **{contrast['difference_pp']:+.2f} pp [{contrast['ci95_low_pp']:+.2f}, {contrast['ci95_high_pp']:+.2f}]**.",
              f"Positive images: **{positive_contrast['difference_pp']:+.2f} pp [{positive_contrast['ci95_low_pp']:+.2f}, {positive_contrast['ci95_high_pp']:+.2f}]**.",
              f"Negative images: **{negative_contrast['difference_pp']:+.2f} pp [{negative_contrast['ci95_low_pp']:+.2f}, {negative_contrast['ci95_high_pp']:+.2f}]**."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
