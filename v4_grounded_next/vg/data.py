"""Batch construction.  Public inference keys and TRAIN-only supervision keys
are built by separate functions so an eval batch cannot carry GT by accident."""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import geometry, hard_negatives
from .paths import FRACATLAS, read_json

PUBLIC_KEYS = ("images", "questions", "proposals", "proposal_source", "image_ids", "component_ids")


def load_image(rel) -> Image.Image:
    from src.preprocessing import canonicalize_raster
    with Image.open(FRACATLAS / rel) as im:
        return canonicalize_raster(im)


def gt_geometry(record):
    """Canonical-image GT boxes, polygons and mask for one image record."""
    o, raw = geometry.orientation_of(FRACATLAS / record["image"])
    boxes = geometry.canonical_boxes(record.get("boxes"), o, raw)
    polys = geometry.canonical_polygons(record.get("segmentation"), o, raw)
    size = (raw[1], raw[0]) if o in (5, 6, 7, 8) else raw
    return boxes, polys, size


def public_batch(records, images, proposals, source, component_lookup=None):
    return {"images": images, "questions": [r["question"] for r in records],
            "proposals": [proposals.get(r["image_id"], []) for r in records],
            "proposal_source": [source] * len(records), "image_ids": [r["image_id"] for r in records],
            "component_ids": [component_lookup.get(r["image_id"], -1) for r in records] if component_lookup else None}


def training_supervision(records, images, cfg, n_neg=4, seed=0):
    """TRAIN-only: gt_masks at vg_loss_res and same-radiograph contrast crops."""
    res, target = cfg["vg_loss_res"], cfg["image_size"]
    gt_masks, contrast = [], []
    for r, im in zip(records, images):
        if not r.get("fractured"):
            gt_masks.append(np.zeros((res, res), dtype=np.float32)); contrast.append(None); continue
        boxes, polys, size = gt_geometry(r)
        assert size == im.size, (r["image_id"], size, im.size)
        mask = geometry.rasterize(polys, size, boxes)
        gt_masks.append(geometry.mask_to_grid(mask, size, target, res))
        gray = np.asarray(im.convert("L"))
        groups = hard_negatives.sample(gray, boxes, r["image_id"], n_neg=n_neg, seed=seed)
        g = groups[0] if groups else None
        contrast.append(None if not g or not g["negatives"] else {
            "positive": im.crop(tuple(int(round(v)) for v in g["positive"])),
            "negatives": [im.crop(tuple(int(round(v)) for v in b)) for b in g["negatives"]],
            "relax": g["relax"]})
    return {"gt_masks": gt_masks, "contrast_crops": contrast}


def training_batch(records, images, proposals, source, cfg, component_lookup=None, seed=0):
    b = public_batch(records, images, proposals, source, component_lookup)
    b.update(answers=[r["answer"] for r in records], answer_types=[r["answer_type"] for r in records])
    b.update(training_supervision(records, images, cfg, seed=seed))
    return b
