"""Final-solution inference: radiograph + question -> answer (+ fracture probability, proposals).

Public input contract (the only inputs used at inference):
  * image: a bone radiograph (PIL / path); EXIF orientation is applied;
  * question: English question within the trained question families.
Detector proposals are computed internally from the image.  Lens views,
annotations and answers are never inputs.  Loading fails closed: a local-route
checkpoint without its detector raises instead of silently degrading to B.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.endpoints import is_canonical_fracture_presence_question
from src.preprocessing import canonicalize_raster


def checkpoint_class(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    name = (payload.get("extra") or {}).get("model_class", "BoneVQAModel")
    cfg = payload["config"]
    if name == "BoneVQAv5" or any(k.startswith("v5_") for k in cfg):
        from experiments.v5.model_v5 import BoneVQAv5 as cls
    elif name == "VNextModel" or cfg.get("vnext_enabled"):
        from experiments.vnext_model_v1 import VNextModel as cls
    else:
        from src.models.bonevqa import BoneVQAModel as cls
    return cls, cfg


def uses_local_route(cfg: dict) -> bool:
    return bool(cfg.get("v5_local") or cfg.get("vnext_enabled") or cfg.get("use_visual_prompt"))


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BoneVQAEngine:
    def __init__(self, checkpoint: str, detector: Optional[str] = None, device: Optional[str] = None,
                 score_threshold: float = 0.05, max_boxes: int = 3):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        self.device = pick_device(device)
        cls, cfg = checkpoint_class(checkpoint)
        self.model = cls.load(checkpoint, device=self.device).eval()
        self.model.refresh_answer_bank()
        self.detector = None
        if uses_local_route(cfg):
            if not detector or not Path(detector).exists():
                raise RuntimeError("Checkpoint uses detector proposals; provide its fold detector (fail closed)")
            from src.detector_runtime import FractureDetectorRuntime

            self.detector = FractureDetectorRuntime(detector, self.device, score_threshold=score_threshold,
                                                    max_boxes=max_boxes)
        self.checkpoint, self.detector_path = checkpoint, detector

    @torch.no_grad()
    def answer(self, image, question: str) -> dict:
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                image = canonicalize_raster(im)
        else:
            image = canonicalize_raster(image)
        boxes = self.detector.predict(image) if self.detector is not None else None
        batch = {"images": [image], "questions": [question]}
        if boxes is not None:
            # Same rectangle-mask construction as the training/evaluation runners
            # (FracAtlasVQADataset._boxes_to_masks); works for v4 and v5.
            from src.data.fracatlas_vqa import FracAtlasVQADataset

            batch.update(boxes=[boxes or None], masks=[FracAtlasVQADataset._boxes_to_masks(boxes or [], image.size)])
        out = self.model.predict_batch(batch)[0]
        result = {"question": question, "answer": out["answer"], "answer_type": out["answer_type"],
                  "confidence": float(out.get("confidence", 0.0)), "proposals": boxes or []}
        if is_canonical_fracture_presence_question(question) and "p_yes" in out:
            result["p_fracture"] = float(out["p_yes"])
        return result


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="BoneVQA-Prompt v5 inference")
    ap.add_argument("image")
    ap.add_argument("question")
    ap.add_argument("--checkpoint", default=os.environ.get("BONEVQA_CKPT"))
    ap.add_argument("--detector", default=os.environ.get("BONEVQA_DETECTOR_CKPT"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    engine = BoneVQAEngine(a.checkpoint, a.detector, a.device)
    print(json.dumps(engine.answer(a.image, a.question), indent=2))


if __name__ == "__main__":
    main()
