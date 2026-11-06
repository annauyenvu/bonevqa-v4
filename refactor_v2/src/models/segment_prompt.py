import hashlib
import logging
import os
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("bonevqa")

MEDSAM_ID = "wanglab/medsam-vit-base"


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


class SegmentPromptCreator:
    def __init__(self, model_id: str = MEDSAM_ID, sam_med2d_ckpt: Optional[str] = None,
                 device: Optional[str] = None, max_masks: int = 3, grid: int = 2,
                 default_prompt: str = "grid_boxes", min_area_ratio: float = 0.002,
                 iou_dedup: float = 0.85, cache_dir: Optional[str] = None, enabled: bool = True,
                 max_area_ratio: float = 0.6, min_mean_intensity: float = 40.0):
        self.model_id = model_id
        self.sam_med2d_ckpt = sam_med2d_ckpt
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_masks = max_masks
        self.grid = grid
        self.default_prompt = default_prompt
        self.min_area_ratio = min_area_ratio
        self.iou_dedup = iou_dedup
        self.max_area_ratio = max_area_ratio
        self.min_mean_intensity = min_mean_intensity
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.model = None
        self.processor = None
        self.load_failed = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _lazy_load(self) -> bool:
        if self.model is not None:
            return True
        if self.load_failed or not self.enabled:
            return False
        try:
            from transformers import SamModel, SamProcessor

            self.processor = SamProcessor.from_pretrained(self.model_id)
            self.model = SamModel.from_pretrained(self.model_id)
            if self.sam_med2d_ckpt and os.path.exists(self.sam_med2d_ckpt):
                self._try_load_sam_med2d(self.sam_med2d_ckpt)
            self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            logger.info("SegmentPromptCreator: đã tải %s trên %s", self.model_id, self.device)
            return True
        except Exception as exc:
            logger.warning("Không tải được SAM (%s) → bỏ qua sinh mask, dùng ảnh gốc", exc)
            self.load_failed = True
            return False

    TY_LE_KHOP_TOI_THIEU = 0.9

    def _try_load_sam_med2d(self, path: str):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            state = {k.replace("image_encoder.", "vision_encoder."): v for k, v in state.items()}
            hien_co = self.model.state_dict()
            khop = {k: v for k, v in state.items() if k in hien_co and hien_co[k].shape == v.shape}
            ty_le = len(khop) / max(len(hien_co), 1)
            if ty_le < self.TY_LE_KHOP_TOI_THIEU:
                logger.warning(
                    "Bỏ qua SAM-Med2D %s: chỉ khớp %d/%d tensor (%.1f%%). Checkpoint SAM-Med2D dùng kiến trúc "
                    "riêng của nhóm tác giả (có %d tensor adapter) nên không nạp được vào transformers.SamModel. "
                    "Tiếp tục dùng %s.",
                    path, len(khop), len(hien_co), ty_le * 100,
                    sum(1 for k in state if "adapter" in k.lower()), self.model_id)
                return False
            self.model.load_state_dict(khop, strict=False)
            logger.info("Đã nạp SAM-Med2D từ %s (%d/%d tensor)", path, len(khop), len(hien_co))
            return True
        except Exception as exc:
            logger.warning("Không nạp được SAM-Med2D checkpoint %s: %s", path, exc)
            return False

    def _default_boxes(self, w: int, h: int) -> List[List[float]]:
        if self.default_prompt == "full":
            return [[0, 0, w - 1, h - 1]]
        g = max(1, self.grid)
        boxes = []
        for i in range(g):
            for j in range(g):
                boxes.append([j * w / g, i * h / g, (j + 1) * w / g - 1, (i + 1) * h / g - 1])
        return boxes

    def _default_points(self, w: int, h: int) -> List[List[float]]:
        g = max(1, self.grid)
        return [[(j + 0.5) * w / g, (i + 0.5) * h / g] for i in range(g) for j in range(g)]

    def _cache_path(self, image: Image.Image, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.md5((key + str(image.size)).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npz")

    @torch.no_grad()
    def __call__(self, pil_image: Image.Image, boxes=None, points=None, cache_key: Optional[str] = None) -> np.ndarray:
        image = pil_image.convert("RGB")
        w, h = image.size
        cache_path = self._cache_path(image, cache_key + str(boxes) + str(points)) if cache_key else None
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)["masks"]
        if not self._lazy_load():
            if os.environ.get("BONEVQA_STRICT_SAM") == "1" and self.enabled:
                raise RuntimeError("SAM loading failed; refusing silent ROI omission")
            return np.zeros((0, h, w), dtype=np.uint8)
        use_points = False
        if boxes is None or len(boxes) == 0:
            if points is not None and len(points) > 0:
                use_points = True
            elif self.default_prompt == "points":
                points = self._default_points(w, h)
                use_points = True
            else:
                boxes = self._default_boxes(w, h)
        candidates = []
        try:
            if use_points:
                inputs = self.processor(image, input_points=[[[list(map(float, p))] for p in points]], return_tensors="pt")
            else:
                inputs = self.processor(image, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt")
            # SamProcessor creates box coordinates as float64.  MPS supports
            # float32/float16 but not float64, so normalize before transfer.
            inputs = {k: (v.float() if v.is_floating_point() else v).to(self.device) for k, v in inputs.items()}
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
            with amp_context:
                outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0]
            scores = outputs.iou_scores.float().cpu()[0]
            masks = masks[:, 0].numpy().astype(np.uint8)
            scores = scores[:, 0].numpy()
            for m, s in zip(masks, scores):
                candidates.append((float(s), m))
        except Exception as exc:
            logger.warning("SAM lỗi khi sinh mask: %s", exc)
            if os.environ.get("BONEVQA_STRICT_SAM") == "1":
                raise RuntimeError("SAM inference failed; refusing silent ROI omission") from exc
            return np.zeros((0, h, w), dtype=np.uint8)
        candidates.sort(key=lambda t: -t[0])
        kept = []
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        min_area = self.min_area_ratio * w * h
        max_area = self.max_area_ratio * w * h
        for s, m in candidates:
            area = m.sum()
            if area < min_area or area > max_area:
                continue
            if gray[m > 0].mean() < self.min_mean_intensity:
                continue
            if any(_mask_iou(m, k) > self.iou_dedup for k in kept):
                continue
            kept.append(m)
            if len(kept) >= self.max_masks:
                break
        result = np.stack(kept).astype(np.uint8) if kept else np.zeros((0, h, w), dtype=np.uint8)
        if cache_path:
            np.savez_compressed(cache_path, masks=result)
        return result
