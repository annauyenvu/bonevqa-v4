"""Same-radiograph hard negatives for L_contrast (TRAIN only).

For a fractured image with canonical GT boxes {g}, R+ is the (context-padded)
union-free crop around each GT box, and each R-_j is a box of the same w,h
(+-10% jitter) on the same radiograph such that
  * IoU(R-_j, g) == 0 for every GT box (strict: no pixel intersection),
  * mean grey intensity > 40 and bone-foreground fraction >= 0.5 (no black
    background crops),
  * |mean_intensity(R-_j) - mean_intensity(R+)| <= 0.25 * mean_intensity(R+).
The sampler is deterministic in (image_id, seed).  Relaxations are recorded
per negative so the cache states exactly which constraint was dropped.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

from .geometry import box_iou

FOREGROUND = 40


def _rng(image_id: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{image_id}:{seed}".encode()).hexdigest()[:16], 16))


def pad_box(b, size, ratio=0.15, min_side=32):
    w, h = size
    bw, bh = b[2] - b[0], b[3] - b[1]
    px, py = max(bw * ratio, (min_side - bw) / 2, 0), max(bh * ratio, (min_side - bh) / 2, 0)
    return [max(0.0, b[0] - px), max(0.0, b[1] - py), min(float(w), b[2] + px), min(float(h), b[3] + py)]


def crop_stats(gray: np.ndarray, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    patch = gray[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if patch.size == 0:
        return 0.0, 0.0
    return float(patch.mean()), float((patch > FOREGROUND).mean())


def _disjoint(c, gts):
    return all(min(c[2], g[2]) <= max(c[0], g[0]) or min(c[3], g[3]) <= max(c[1], g[1]) for g in gts)


def sample(gray: np.ndarray, gt_boxes, image_id: str, n_neg: int = 4, seed: int = 0, tries: int = 400):
    """Returns [{'positive': box, 'negatives': [box...], 'relax': [...], ...}] per GT box."""
    h, w = gray.shape
    rng = _rng(image_id, seed)
    out = []
    for g in gt_boxes:
        pos = pad_box(g, (w, h))
        pos_mean, _ = crop_stats(gray, pos)
        bw, bh = pos[2] - pos[0], pos[3] - pos[1]
        negs, relax = [], []
        levels = [("strict", 0.25, 0.5), ("intensity_relaxed", 0.6, 0.5), ("foreground_only", None, 0.3)]
        for level, tol, fg_min in levels:
            for _ in range(tries):
                if len(negs) >= n_neg:
                    break
                sw, sh = bw * rng.uniform(0.9, 1.1), bh * rng.uniform(0.9, 1.1)
                if sw >= w or sh >= h:
                    continue
                x, y = rng.uniform(0, w - sw), rng.uniform(0, h - sh)
                c = [x, y, x + sw, y + sh]
                if not _disjoint(c, gt_boxes) or any(box_iou(c, n) > 0.5 for n in negs):
                    continue
                m, fg = crop_stats(gray, c)
                if m <= FOREGROUND or fg < fg_min:
                    continue
                if tol is not None and abs(m - pos_mean) > tol * max(pos_mean, 1.0):
                    continue
                negs.append(c)
                relax.append(level)
        out.append({"gt": list(map(float, g)), "positive": pos, "positive_mean": pos_mean,
                    "negatives": negs, "relax": relax, "complete": len(negs) == n_neg})
    return out
