"""Mask/proposal generation for v5 (run once, cached, resumable).

Sub-commands (run from /Users/uyen/Downloads/BONEVQA with PYTHONPATH set):

  anatomy --source medsam|sam_huge|otsu [--device mps]
      Localization Lens views (paper Sec. 2.1): anatomical segmentation for
      EVERY image (positives and negatives alike) so view presence carries no
      label.  The paper combines SAM (coarse) and MedSAM (precise); run both
      and use ``--merge`` in the study config.

  avpc_tune --fold 0 [--device mps]
      FAVP Adaptive Visual Prompt Creator with SAM-Med2D.  Uses the AUTHOR
      default filters (pred_iou 0.68, stability 0.70, offset 0.70; favp.py),
      not the stricter 0.80/0.90 used by the earlier port (which kept zero
      masks on X-rays).  Sweeps keypoints (points_per_side) and NMS tau as in
      FAVP Fig. 4, scored ONLY on TRAIN images of the fold.

  avpc_generate --points N --nms T [--device mps]
      Proposals for all images with the selected setting -> boxes json in the
      same format as the detector boxes used by v4.

Masks are stored downsampled (long side <= 512) as packed npz; the prompt
builders resize them back with nearest-neighbour interpolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
DATA = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3"
OUT = ROOT / ".evaluation_runtime/v5_masks"
SAM_HUGE = ROOT / "refactor_v2/references/native_assets_lens_v1/sam-vit-huge"
MAX_SIDE = 512


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(v, indent=2))
    tmp.replace(p)


def all_images(fold: str = "0"):
    """Every image of the cohort (the five outer folds partition the same set)."""
    seen = {}
    roles = {}
    for role in ("train", "model_val", "calibration", "test"):
        for r in read(DATA / f"fold_{fold}/vqa_{role}.json"):
            if r["image_id"] not in seen:
                seen[r["image_id"]] = r
                roles[r["image_id"]] = role
    return seen, roles


def load_image(rel):
    from src.data.fracatlas_vqa import FRACATLAS_ROOT
    from src.preprocessing import canonicalize_raster

    with Image.open(FRACATLAS_ROOT / rel) as im:
        return canonicalize_raster(im).convert("RGB")


def _shrink(masks: np.ndarray, size):
    w, h = size
    s = min(1.0, MAX_SIDE / max(w, h))
    if s >= 1.0 or len(masks) == 0:
        return masks.astype(np.uint8)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return np.stack([np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 0
                     for m in masks]).astype(np.uint8)


def save_masks(path: Path, masks: np.ndarray, size, scores=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    small = _shrink(np.asarray(masks), size)
    np.savez_compressed(path, masks=np.packbits(small, axis=-1), shape=np.array(small.shape),
                        size=np.array(size), scores=np.asarray(scores if scores is not None else [], dtype=np.float32))


def load_masks(path: Path) -> np.ndarray:
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    if shape[0] == 0:
        w, h = (int(v) for v in z["size"])
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.unpackbits(z["masks"], axis=-1)[..., : shape[-1]].reshape(shape).astype(np.uint8)


def _filter(cands, image, max_masks=6, min_area=0.005, max_area=0.6, iou_dedup=0.85):
    """Keep bone-like masks: bright (above image median), plausible area."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    w, h = image.size
    median = float(np.median(gray))
    kept, scores = [], []
    for s, m in sorted(cands, key=lambda t: -t[0]):
        m = m.astype(bool)
        a = m.sum() / float(w * h)
        if a < min_area or a > max_area or gray[m].mean() <= median:
            continue
        if any(np.logical_and(m, k).sum() / max(np.logical_or(m, k).sum(), 1) > iou_dedup for k in kept):
            continue
        kept.append(m)
        scores.append(float(s))
        if len(kept) >= max_masks:
            break
    arr = np.stack(kept).astype(np.uint8) if kept else np.zeros((0, h, w), dtype=np.uint8)
    return arr, scores


# ------------------------------------------------------------------ sources
def _float32_on_device(pipe):
    """HF pipelines move float64 preprocessor tensors as-is; MPS has no float64."""
    import torch

    move = pipe._ensure_tensor_on_device

    def ensure(inputs, device):
        if isinstance(inputs, torch.Tensor) and inputs.dtype == torch.float64:
            inputs = inputs.float()
        return move(inputs, device)

    pipe._ensure_tensor_on_device = ensure


class MedSAMGrid:
    def __init__(self, device):
        from src.models.segment_prompt import SegmentPromptCreator

        self.seg = SegmentPromptCreator(device=device, max_masks=9, grid=3, default_prompt="grid_boxes",
                                        min_area_ratio=0.0, max_area_ratio=1.0, min_mean_intensity=0.0,
                                        iou_dedup=1.01)

    def __call__(self, image):
        os.environ["BONEVQA_STRICT_SAM"] = "1"
        m = self.seg(image)
        return [(1.0 - 0.01 * i, x) for i, x in enumerate(m)]


class SAMHugeAuto:
    """Localization Lens author pipeline (sam_augment.py, pinned 22a94007), unchanged settings:
    facebook/sam-vit-huge, points_per_crop 32, points_per_batch 64, crops_n_layers 0,
    pred_iou 0.80, stability 0.95, crop-edge rejection, score-ordered de-duplication at IoU 0.7.
    No bone-brightness filter is applied (the author pipeline has none); at most 24 masks are
    stored per image (storage bound, highest scores first)."""

    def __init__(self, device):
        import importlib
        from transformers import pipeline

        sys.path.insert(0, str(ROOT / "refactor_v2/references/lens_author_22a94007/src"))
        self.author = importlib.import_module("localization_lens.sam_augment")
        dev = 0 if str(device).startswith("cuda") else (device if device == "mps" else -1)
        self.gen = pipeline("mask-generation", model=str(SAM_HUGE), device=dev)
        self.args = argparse.Namespace(points_per_batch=64, points_per_crop=32, crops_n_layers=0,
                                       pred_iou_thresh=0.80, stability_score_thresh=0.95)

    def __call__(self, image):
        masks = self.author.generate_masks(self.gen, image, self.args)
        return [(1.0 - 1e-3 * k, m.astype(np.uint8)) for k, m in enumerate(masks[:24])]


class OtsuBone:
    """Deterministic fallback: Otsu foreground, largest connected components."""

    def __init__(self, device=None):
        pass

    def __call__(self, image):
        from scipy import ndimage

        g = np.asarray(image.convert("L"), dtype=np.uint8)
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
        sigma = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
        t = int(np.argmax(sigma))
        lab, n = ndimage.label(g > t)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) if n else []
        order = np.argsort(sizes)[::-1][:6]
        return [(float(sizes[i]), (lab == i + 1).astype(np.uint8)) for i in order]


SOURCES = {"medsam": MedSAMGrid, "sam_huge": SAMHugeAuto, "otsu": OtsuBone}


def run_anatomy(source: str, device: str, limit=None, train_first: str = None, only_train: bool = False):
    images, _ = all_images()
    if train_first is not None:
        train_ids = {r["image_id"] for r in read(DATA / f"fold_{train_first}/vqa_train.json")}
        order = sorted(images, key=lambda i: (i not in train_ids, i))
        if only_train:
            order = [i for i in order if i in train_ids]
        images = {i: images[i] for i in order}
    gen = SOURCES[source](device)
    folder = OUT / f"anatomy_{source}"
    stats, start = {"images": 0, "nonempty": 0, "masks": 0, "failures": []}, time.time()
    for k, (iid, rec) in enumerate(images.items() if train_first is not None else sorted(images.items())):
        if limit and k >= limit:
            break
        path = folder / f"{iid}.npz"
        if path.exists():
            continue
        im = load_image(rec["image"])
        try:
            cands = gen(im)
            if source == "sam_huge":
                masks = np.stack([m for _, m in cands]).astype(np.uint8) if cands else np.zeros((0, im.height, im.width), np.uint8)
                scores = [s for s, _ in cands]
            else:
                masks, scores = _filter(cands, im)
        except Exception as exc:  # recorded, never silently replaced
            stats["failures"].append([iid, repr(exc)[:200]])
            continue
        save_masks(path, masks, im.size, scores)
        stats["images"] += 1
        stats["nonempty"] += int(len(masks) > 0)
        stats["masks"] += int(len(masks))
        if stats["images"] % 25 == 0:
            dump(folder / "progress.json", {**stats, "total": len(images), "seconds": time.time() - start})
    files = sorted(folder.glob("*.npz"))
    cover = sum(int(load_masks(f).shape[0] > 0) for f in files)
    dump(folder / "receipt.json", {**stats, "total": len(images), "files": len(files), "coverage": cover / max(1, len(files)),
                                   "source": source, "device": device, "seconds": time.time() - start})


# --------------------------------------------------------------------- AVPC
def sammed2d_generator(points_per_side: int, nms: float, device: str, pred_iou=0.68, stability=0.70):
    import importlib
    from experiments.favp_visual_bridge_v2 import namespace  # same loader path as the sealed FAVP pilot
    from experiments.favp_sam_restricted_loader_v1 import load_archive

    namespace()
    SamAutomaticMaskGenerator = importlib.import_module("favp.segmed.automatic_mask_generator").SamAutomaticMaskGenerator
    build_sam_vit_b = importlib.import_module("favp.segmed.build_sam").build_sam_vit_b

    model = build_sam_vit_b(256, checkpoint=None, encoder_adapter=True).eval()
    archive = load_archive(ROOT / "refactor_v2/references/native_assets_favp_v1/sam-med2d_b.pth", map_location="cpu")
    model.load_state_dict(archive["model"], strict=True)
    model.requires_grad_(False).to(device)
    gen = SamAutomaticMaskGenerator(model, points_per_side=points_per_side, points_per_batch=64,
                                    pred_iou_thresh=pred_iou, stability_score_thresh=stability,
                                    stability_score_offset=0.7, box_nms_thresh=nms, crop_n_layers=0,
                                    crop_nms_thresh=0.7, min_mask_region_area=100, output_mode="binary_mask")
    # the author AMG moves float64 numpy point grids to the device; MPS has no float64
    tf = gen.predictor.transform
    apply_coords = tf.apply_coords
    tf.apply_coords = lambda coords, size: apply_coords(coords, size).astype(np.float32)
    if str(device).startswith("mps"):
        # same MPS allocator trap as SAMHugeAuto when indexing full-res logits; filter on CPU
        predict = gen.predictor.predict_torch
        gen.predictor.predict_torch = lambda *a, **k: tuple(t.cpu() for t in predict(*a, **k))
    return gen


def _avpc_boxes(gen, im, k=3):
    out = gen.generate(np.asarray(im))
    out = sorted(out, key=lambda r: -r["predicted_iou"])[:k]
    return [[float(v) for v in (r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3])]
            for r in out], [r["segmentation"].astype(np.uint8) for r in out]


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_avpc_tune(fold: str, device: str, n_images: int = 120):
    """Selection on TRAIN only: lesion-box hit rate (IoU >= 0.1) of top-3 proposals."""
    train = read(DATA / f"fold_{fold}/vqa_train.json")
    pos = {}
    for r in train:
        if r.get("boxes") and r["image_id"] not in pos:
            pos[r["image_id"]] = r
    ids = sorted(pos)[:n_images]
    results = []
    for points in (30, 40, 50, 60):
        for nms in (0.75, 0.80, 0.85, 0.90):
            gen = sammed2d_generator(points, nms, device)
            hit, count, start = 0, 0, time.time()
            for iid in ids:
                im = load_image(pos[iid]["image"])
                boxes, _ = _avpc_boxes(gen, im)
                gt = [b[:4] for b in pos[iid]["boxes"]]  # canonical coordinates (EXIF-rotated images have no GT box)
                count += len(boxes)
                hit += int(any(_iou(b, g) >= 0.1 for b in boxes for g in gt))
            results.append({"points_per_side": points, "nms": nms, "train_images": len(ids),
                            "hit_rate": hit / len(ids), "mean_proposals": count / len(ids),
                            "seconds": time.time() - start})
            dump(OUT / f"avpc_tune_fold{fold}.json", {"scope": "TRAIN only", "results": results})
    best = max(results, key=lambda r: (r["hit_rate"], -abs(r["mean_proposals"] - 3)))
    dump(OUT / f"avpc_tune_fold{fold}.json", {"scope": "TRAIN only", "results": results, "selected": best})


def run_avpc_generate(points: int, nms: float, device: str):
    images, _ = all_images()
    gen = sammed2d_generator(points, nms, device)
    folder = OUT / f"avpc_p{points}_nms{nms}"
    boxes_path = folder / "boxes.json"
    boxes = read(boxes_path) if boxes_path.exists() else {}
    start = time.time()
    for iid, rec in sorted(images.items()):
        if iid in boxes:
            continue
        im = load_image(rec["image"])
        b, m = _avpc_boxes(gen, im)
        boxes[iid] = b
        save_masks(folder / "masks" / f"{iid}.npz", np.stack(m) if m else np.zeros((0, im.height, im.width), np.uint8), im.size)
        if len(boxes) % 25 == 0:
            dump(boxes_path, boxes)
            dump(folder / "progress.json", {"images": len(boxes), "total": len(images), "seconds": time.time() - start})
    dump(boxes_path, boxes)
    sha = hashlib.sha256(boxes_path.read_bytes()).hexdigest()
    dump(folder / "receipt.json", {"points_per_side": points, "nms": nms, "pred_iou": 0.68, "stability": 0.70,
                                   "images": len(boxes), "nonempty": sum(bool(v) for v in boxes.values()),
                                   "boxes_sha256": sha, "seconds": time.time() - start})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anatomy", "avpc_tune", "avpc_generate"])
    ap.add_argument("--source", default="medsam")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--points", type=int, default=50)
    ap.add_argument("--nms", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train_first", default=None, help="fold whose TRAIN images are generated first")
    ap.add_argument("--only_train", action="store_true")
    a = ap.parse_args()
    if a.cmd == "anatomy":
        run_anatomy(a.source, a.device, a.limit, a.train_first, a.only_train)
    elif a.cmd == "avpc_tune":
        run_avpc_tune(a.fold, a.device)
    else:
        run_avpc_generate(a.points, a.nms, a.device)
