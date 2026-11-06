import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None

RED = (255, 0, 0)
LINE_WIDTH = 3
MASK_ALPHA = 0.4
PALETTE = [
    (255, 0, 0), (0, 200, 0), (0, 90, 255), (255, 200, 0), (255, 0, 255),
    (0, 220, 220), (255, 128, 0), (128, 0, 255), (0, 128, 90), (200, 200, 200),
]
LENS_ALPHA_MEAN = 0.5
LENS_ALPHA_STD = 0.1
LENS_ALPHA_RANGE = (0.2, 0.8)


def _to_rgb(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def _as_mask_array(masks):
    if masks is None:
        return np.zeros((0, 1, 1), dtype=np.uint8)
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None]
    return (arr > 0).astype(np.uint8)


def _fit_masks(masks, size):
    w, h = size
    if masks.shape[0] == 0 or (masks.shape[1] == h and masks.shape[2] == w):
        return masks
    resized = [np.array(Image.fromarray(m * 255).resize((w, h), Image.NEAREST)) > 0 for m in masks]
    return np.stack(resized).astype(np.uint8)


def masks_to_boxes(masks):
    arr = _as_mask_array(masks)
    boxes = []
    for m in arr:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            boxes.append([0, 0, 0, 0])
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes


def draw_boxes(image, boxes, color=RED, width=LINE_WIDTH):
    img = _to_rgb(image).copy()
    if not boxes:
        return img
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def overlay_masks(image, masks, alpha=MASK_ALPHA, colors=None):
    img = _to_rgb(image)
    arr = _fit_masks(_as_mask_array(masks), img.size)
    base = np.asarray(img).astype(np.float32)
    for i, m in enumerate(arr):
        color = np.array((colors[i % len(colors)] if colors else PALETTE[i % len(PALETTE)]), dtype=np.float32)
        sel = m > 0
        base[sel] = base[sel] * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _contours_of(mask):
    if cv2 is not None:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2) for c in cs if len(c) >= 3]
    return None


def draw_contours(image, masks, color=RED, width=LINE_WIDTH):
    img = _to_rgb(image).copy()
    arr = _fit_masks(_as_mask_array(masks), img.size)
    if arr.shape[0] == 0:
        return img
    if cv2 is not None:
        canvas = np.array(img, dtype=np.uint8, copy=True)
        for m in arr:
            cs = _contours_of(m)
            if cs:
                cv2.drawContours(canvas, [c.reshape(-1, 1, 2).astype(np.int32) for c in cs], -1, color, width)
        return Image.fromarray(canvas)
    edge = np.zeros(arr.shape[1:], dtype=bool)
    for m in arr:
        mb = m > 0
        inner = mb.copy()
        inner[1:-1, 1:-1] = mb[1:-1, 1:-1] & mb[:-2, 1:-1] & mb[2:, 1:-1] & mb[1:-1, :-2] & mb[1:-1, 2:]
        edge |= mb & ~inner
    canvas = np.asarray(img).copy()
    ys, xs = np.where(edge)
    draw = ImageDraw.Draw(img)
    for x, y in zip(xs, ys):
        draw.ellipse([x - width // 2, y - width // 2, x + width // 2, y + width // 2], fill=color)
    return img


def draw_circles(image, masks, color=RED, width=LINE_WIDTH):
    img = _to_rgb(image).copy()
    arr = _fit_masks(_as_mask_array(masks), img.size)
    draw = ImageDraw.Draw(img)
    for b in masks_to_boxes(arr):
        x1, y1, x2, y2 = b
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r = max((x2 - x1), (y2 - y1)) / 2.0 * 1.1 + width
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)
    return img


def build_visual_prompts(image, masks, kind="contour"):
    img = _to_rgb(image)
    arr = _fit_masks(_as_mask_array(masks), img.size)
    outputs = []
    if arr.shape[0] == 0:
        return [img.copy()]
    for m in arr:
        single = m[None]
        if kind == "contour":
            outputs.append(draw_contours(img, single))
        elif kind == "box":
            outputs.append(draw_boxes(img, masks_to_boxes(single)))
        elif kind == "circle":
            outputs.append(draw_circles(img, single))
        elif kind == "mask":
            outputs.append(overlay_masks(img, single))
        else:
            raise ValueError(f"kind không hợp lệ: {kind}")
    return outputs


def build_lens(image, masks, mode="unicolor", rng=None, alpha=None):
    img = _to_rgb(image)
    arr = _fit_masks(_as_mask_array(masks), img.size)
    base = np.asarray(img).astype(np.float32)
    if arr.shape[0] == 0:
        if mode == "masked":
            return Image.fromarray(np.zeros_like(base, dtype=np.uint8))
        return img.copy()
    if mode == "masked":
        union = arr.max(axis=0) > 0
        out = np.zeros_like(base)
        out[union] = base[union]
        return Image.fromarray(out.astype(np.uint8))
    rng = rng if rng is not None else np.random  # EXP4: use the process seed
    if alpha is None:
        alpha = float(np.clip(rng.normal(LENS_ALPHA_MEAN, LENS_ALPHA_STD), *LENS_ALPHA_RANGE))
    if mode == "unicolor":
        union = arr.max(axis=0) > 0
        color = np.array(PALETTE[0], dtype=np.float32)
        base[union] = base[union] * (1 - alpha) + color * alpha
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if mode == "multicolor":
        for i, m in enumerate(arr):
            color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
            sel = m > 0
            base[sel] = base[sel] * (1 - alpha) + color * alpha
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    raise ValueError(f"mode không hợp lệ: {mode}")


def textual_prompt(question, region=None, modality="X-ray"):
    modality_text = f"bone {modality} image" if modality.lower() in ("x-ray", "xray", "radiograph") else f"{modality} image"
    region_text = f" of the {region}" if region else ""
    return f"This is a {modality_text}{region_text}. Answer the question briefly. Question: {question}"
