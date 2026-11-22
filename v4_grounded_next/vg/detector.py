"""Detector utilities: the exact V4 recipe, plus a score-returning runtime.

The sealed ``FractureDetectorRuntime.predict`` drops scores; recovery needs
s_max, so :func:`predict_scored` re-implements the same preprocessing and
coordinate inversion and keeps scores.  Parity with sealed boxes.json is
checked in ``cross_fit_proposals.py heldout``.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from .paths import FRACATLAS, dump_json, guard_write


def predict_raw(model, image: Image.Image, floor=0.001, image_size=640):
    """Every box the detector returns with score >= ``floor`` (no top-3 cap), in the
    model's own score-descending order, clipped to the canonical image, degenerate
    boxes dropped.  torchvision drops boxes below ``roi_heads.score_thresh`` (0.05)
    *before* NMS, so it is lowered for this call.  Boxes that would score >= 0.05 are
    unchanged by this: a lower-score box can never suppress a higher-score one in
    NMS and the 100-detection cap keeps the highest scores.  ``vg.eqm.v4_contract``
    on this output with the identity map reproduces :func:`predict_scored`."""
    old = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = floor
    try:
        return predict_scored(model, image, image_size=image_size, score_threshold=floor, max_boxes=None)
    finally:
        model.roi_heads.score_thresh = old


def predict_scored(model, image: Image.Image, image_size=640, score_threshold=0.05, max_boxes=3):
    from src.preprocessing import canonicalize_raster
    canonical = canonicalize_raster(image)
    w, h = canonical.size
    scale = image_size / max(w, h)
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = canonical.resize((rw, rh), Image.BILINEAR)
    px, py = (image_size - rw) // 2, (image_size - rh) // 2
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(resized, (px, py))
    t = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = model([t])[0]
    out = []
    for box, score in zip(pred["boxes"].cpu().tolist(), pred["scores"].cpu().tolist()):
        if score < score_threshold or (max_boxes is not None and len(out) >= max_boxes):
            continue
        x1 = max(0.0, min(float(w), (box[0] - px) / scale)); y1 = max(0.0, min(float(h), (box[1] - py) / scale))
        x2 = max(0.0, min(float(w), (box[2] - px) / scale)); y2 = max(0.0, min(float(h), (box[3] - py) / scale))
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2, float(score)])
    return out, canonical.size


def train_detector_v4_recipe(ids, dest, epochs=2, seed=42, max_updates=None, progress_path=None):
    """Byte-for-byte the V4 recipe (refactor_v2/experiments/local_noaux_refinement_v1.detector)."""
    from scripts.train_detector_v2 import BoPhatHienDataset, PositiveAnchoredBatchSampler, collate, tao_mo_hinh
    from src.preprocessing import RASTER_PREPROCESSING_VERSION
    from src.utils import set_seed
    from torch.utils.data import DataLoader

    dest = guard_write(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds =BoPhatHienDataset("train", canh_dai=640, tang_cuong=True, allowed_ids=set(ids))
    set_seed(seed)
    model = tao_mo_hinh("mobilenet_320").cpu()
    model.transform.min_size = (640,); model.transform.max_size = 640
    norm = {k: {"type": type(v).__name__, "eps": v.eps} for k, v in model.named_modules()
            if hasattr(v, "running_mean") and hasattr(v, "eps")}
    assert all(v["type"] == "FrozenBatchNorm2d" for v in norm.values())
    sampler = PositiveAnchoredBatchSampler(ds, 4, seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005, momentum=.9, weight_decay=.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    updates, start, history = 0, time.time(), []
    for epoch in range(epochs):
        sampler.epoch = epoch; model.train(); losses = []
        for images, targets, _ in loader:
            loss = sum(model(list(images), list(targets)).values())
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
            opt.step(); updates += 1; losses.append(float(loss.detach()))
            if progress_path and updates % 25 == 0:
                dump_json(progress_path, {"epoch": epoch + 1, "updates": updates, "loss": losses[-1], "wall": time.time() - start})
            if max_updates and updates >= max_updates:
                break
        sched.step(); history.append({"epoch": epoch + 1, "updates": updates, "mean_loss": float(np.mean(losses))})
        if max_updates and updates >= max_updates:
            break
    torch.save({"state": model.state_dict(), "architecture": "mobilenet_320", "canh_dai": 640,
                "detector_transform": {"min_size": [640], "max_size": 640, "size_divisible": 32},
                "normalization_spec": norm, "training_image_ids": sorted(ids),
                "preprocessing_version": RASTER_PREPROCESSING_VERSION,
                "initial_weight_provenance": model.initial_weight_provenance, "epoch": epochs,
                "optimizer_steps": updates}, dest)
    return {"history": history, "updates": updates, "wall_seconds": time.time() - start}


def load_detector(path):
    from experiments.detector640_checkpoint_v3 import load_bound_640
    return load_bound_640(str(path))


def open_canonical(rel):
    with Image.open(FRACATLAS / rel) as im:
        im.load()
        return im.copy()
