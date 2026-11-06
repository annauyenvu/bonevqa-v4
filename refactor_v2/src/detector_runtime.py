"""Production detector runtime using the sealed EXIF-canonical contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .preprocessing import RASTER_PREPROCESSING_VERSION, canonicalize_raster


def build_detector(architecture: str):
    if architecture == "mobilenet_320":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None,
        )
    elif architecture == "resnet50_640":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None,
        )
    else:
        raise ValueError(f"Unsupported detector architecture: {architecture}")
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


class FractureDetectorRuntime:
    def __init__(self, checkpoint: str | Path, device: str, score_threshold=0.05, max_boxes=3):
        self.checkpoint = Path(checkpoint)
        raw = self.checkpoint.read_bytes()
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("preprocessing_version") != RASTER_PREPROCESSING_VERSION:
            raise RuntimeError("Detector runtime requires EXIF-canonical v3 checkpoint")
        self.device = torch.device(device)
        self.image_size = int(payload["canh_dai"])
        self.score_threshold = float(score_threshold)
        self.max_boxes = int(max_boxes)
        self.model = build_detector(payload["architecture"]).to(self.device)
        self.model.load_state_dict(payload["state"], strict=True)
        self.model.eval()
        self.provenance = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            "preprocessing_version": RASTER_PREPROCESSING_VERSION,
            "architecture": payload["architecture"],
            "score_threshold": self.score_threshold,
            "max_boxes": self.max_boxes,
        }

    def _preprocess(self, image: Image.Image):
        image = canonicalize_raster(image)
        width, height = image.size
        scale = self.image_size / max(width, height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = image.resize((resized_w, resized_h), Image.BILINEAR)
        pad_x = (self.image_size - resized_w) // 2
        pad_y = (self.image_size - resized_h) // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        canvas.paste(resized, (pad_x, pad_y))
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return image, tensor, scale, pad_x, pad_y

    @torch.no_grad()
    def predict(self, image: Image.Image):
        canonical, tensor, scale, pad_x, pad_y = self._preprocess(image)
        prediction = self.model([tensor.to(self.device)])[0]
        boxes = []
        for box, score in zip(prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist()):
            if score < self.score_threshold or len(boxes) >= self.max_boxes:
                continue
            x1 = max(0.0, min(float(canonical.width), (box[0] - pad_x) / scale))
            y1 = max(0.0, min(float(canonical.height), (box[1] - pad_y) / scale))
            x2 = max(0.0, min(float(canonical.width), (box[2] - pad_x) / scale))
            y2 = max(0.0, min(float(canonical.height), (box[3] - pad_y) / scale))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return boxes
