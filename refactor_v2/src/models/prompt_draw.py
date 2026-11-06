import random
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 200, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255),
]


def _to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _fit_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def draw_prompt(image: Image.Image, mask: np.ndarray, kind: str = "contour",
                color=(255, 0, 0), thickness: int = 2) -> Image.Image:
    arr = _to_np(image)
    h, w = arr.shape[:2]
    m = _fit_mask(mask, h, w)
    if m.sum() == 0:
        return Image.fromarray(arr)
    ys, xs = np.where(m > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if kind == "contour":
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, contours, -1, color, thickness)
    elif kind == "box":
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, thickness)
    elif kind == "circle":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = int(max(x2 - x1, y2 - y1) / 2 * 1.1) + 2
        cv2.circle(arr, (cx, cy), r, color, thickness)
    elif kind == "mask":
        overlay = arr.copy()
        overlay[m > 0] = color
        arr = cv2.addWeighted(arr, 0.6, overlay, 0.4, 0)
    else:
        raise ValueError(f"Loại prompt không hỗ trợ: {kind}")
    return Image.fromarray(arr)


def build_visual_prompts(image: Image.Image, masks: Optional[np.ndarray], kind: str = "contour") -> List[Image.Image]:
    if masks is None or len(masks) == 0:
        return []
    return [draw_prompt(image, masks[i], kind=kind, color=PALETTE[i % len(PALETTE)]) for i in range(len(masks))]


def build_lens(image: Image.Image, masks: Optional[np.ndarray], mode: str = "unicolor",
               alpha: Optional[float] = None, rng: Optional[random.Random] = None) -> Image.Image:
    arr = _to_np(image)
    if masks is None or len(masks) == 0:
        return Image.fromarray(arr)
    h, w = arr.shape[:2]
    rng = rng or random
    if alpha is None:
        alpha = float(np.clip(rng.gauss(0.5, 0.1), 0.2, 0.8))
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        union |= _fit_mask(m, h, w)
    if mode == "unicolor":
        overlay = arr.copy()
        overlay[union > 0] = PALETTE[0]
        out = arr.copy()
        out[union > 0] = (arr[union > 0] * (1 - alpha) + overlay[union > 0] * alpha).astype(np.uint8)
    elif mode == "multicolor":
        overlay = arr.copy()
        for i, m in enumerate(masks):
            mm = _fit_mask(m, h, w)
            overlay[mm > 0] = PALETTE[i % len(PALETTE)]
        out = arr.copy()
        out[union > 0] = (arr[union > 0] * (1 - alpha) + overlay[union > 0] * alpha).astype(np.uint8)
    elif mode == "masked":
        out = arr.copy()
        out[union == 0] = 0
    else:
        raise ValueError(f"Chế độ lens không hỗ trợ: {mode}")
    return Image.fromarray(out)


def build_all_lenses(image: Image.Image, masks: Optional[np.ndarray], rng: Optional[random.Random] = None):
    return {mode: build_lens(image, masks, mode=mode, rng=rng) for mode in ("unicolor", "multicolor", "masked")}
