"""Coordinate contract: raw-file annotations -> EXIF-canonical image -> letterbox grid.

FracAtlas COCO boxes/polygons are in raw-file coordinates.  21/3371 audited
images carry EXIF orientation 3/6/8, so GT must be transformed exactly like
``ImageOps.exif_transpose`` before it is compared with canonical-image
proposals or the 224 letterboxed ViT grid.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def exif_point_fn(orientation: int, raw_w: float, raw_h: float):
    w, h = raw_w, raw_h
    table = {
        1: lambda x, y: (x, y), 2: lambda x, y: (w - x, y), 3: lambda x, y: (w - x, h - y),
        4: lambda x, y: (x, h - y), 5: lambda x, y: (y, x), 6: lambda x, y: (h - y, x),
        7: lambda x, y: (h - y, w - x), 8: lambda x, y: (y, w - x),
    }
    return table.get(int(orientation), table[1])


def orientation_of(path) -> tuple[int, tuple[int, int]]:
    with Image.open(path) as im:
        return int(im.getexif().get(274, 1) or 1), im.size


def canonical_polygons(polys_per_ann, orientation, raw_size):
    """[[ [x,y,x,y,...], ...] per annotation] -> list of (N,2) float arrays."""
    fn = exif_point_fn(orientation, *raw_size)
    out = []
    for ann in polys_per_ann or []:
        for flat in ann:
            pts = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
            out.append(np.array([fn(x, y) for x, y in pts], dtype=np.float64))
    return out


def canonical_boxes(boxes, orientation, raw_size):
    fn = exif_point_fn(orientation, *raw_size)
    out = []
    for b in boxes or []:
        x1, y1, x2, y2 = map(float, b[:4])
        xs, ys = zip(*[fn(x1, y1), fn(x1, y2), fn(x2, y1), fn(x2, y2)])
        out.append([min(xs), min(ys), max(xs), max(ys)])
    return out


def rasterize(polygons, size, boxes=None) -> np.ndarray:
    """Canonical-image uint8 mask (H,W).  Falls back to boxes if no polygon."""
    w, h = size
    m = np.zeros((h, w), dtype=np.uint8)
    if polygons:
        assert cv2 is not None, "opencv required for polygon rasterization"
        cv2.fillPoly(m, [np.round(p).astype(np.int32) for p in polygons], 1)
    elif boxes:
        for x1, y1, x2, y2 in boxes:
            m[int(max(0, y1)):int(np.ceil(min(h, y2))), int(max(0, x1)):int(np.ceil(min(w, x2)))] = 1
    return m


def letterbox_params(size, target: int):
    """Exactly mirrors refactor_v2/src/preprocessing.letterbox."""
    w, h = size
    s = min(target / max(w, 1), target / max(h, 1))
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return s, (target - nw) // 2, (target - nh) // 2, nw, nh


def mask_to_grid(mask: np.ndarray, size, target: int, grid: int) -> np.ndarray:
    """Canonical mask -> letterboxed target x target -> area-averaged grid x grid soft map."""
    s, px, py, nw, nh = letterbox_params(size, target)
    canvas = np.zeros((target, target), dtype=np.float32)
    if mask.any():
        small = np.asarray(Image.fromarray(mask * 255).resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        canvas[py:py + nh, px:px + nw] = small
    k = target // grid
    assert k * grid == target
    return canvas.reshape(grid, k, grid, k).mean(axis=(1, 3))


def boxes_to_grid(boxes, size, target: int, grid: int, weights=None) -> np.ndarray:
    """Score-weighted proposal footprint on the letterbox grid (max over boxes)."""
    s, px, py, nw, nh = letterbox_params(size, target)
    out = np.zeros((grid, grid), dtype=np.float32)
    cell = target / grid
    for i, b in enumerate(boxes or []):
        x1, y1, x2, y2 = (b[0] * s + px) / cell, (b[1] * s + py) / cell, (b[2] * s + px) / cell, (b[3] * s + py) / cell
        gx = np.clip(np.minimum(np.arange(grid) + 1, x2) - np.maximum(np.arange(grid), x1), 0, 1)
        gy = np.clip(np.minimum(np.arange(grid) + 1, y2) - np.maximum(np.arange(grid), y1), 0, 1)
        w = 1.0 if weights is None else float(weights[i])
        out = np.maximum(out, w * np.outer(gy, gx).astype(np.float32))
    return out


def grid_to_image_point(iy: int, ix: int, size, target: int, grid: int):
    """Centre of a grid cell mapped back to canonical-image pixels."""
    s, px, py, _, _ = letterbox_params(size, target)
    cell = target / grid
    return ((ix + 0.5) * cell - px) / s, ((iy + 0.5) * cell - py) / s


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(proposals, gts) -> float:
    if not proposals or not gts:
        return 0.0
    return max(box_iou(p[:4], g) for p in proposals for g in gts)


def bone_foreground_bbox(gray: np.ndarray, thr: int = 40):
    ys, xs = np.where(gray > thr)
    if not len(xs):
        h, w = gray.shape
        return [0, 0, w, h]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def multiscale_grid_boxes(gray: np.ndarray, overlap: float = 0.15):
    """Detector-miss fallback: 2x2 overlapping quadrants of the bone foreground bbox."""
    x1, y1, x2, y2 = bone_foreground_bbox(gray)
    w, h = x2 - x1, y2 - y1
    cw, ch = w * (0.5 + overlap / 2), h * (0.5 + overlap / 2)
    out = []
    for oy in (y1, y2 - ch):
        for ox in (x1, x2 - cw):
            out.append([float(ox), float(oy), float(ox + cw), float(oy + ch)])
    return out
