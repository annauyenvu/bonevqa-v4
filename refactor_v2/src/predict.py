import argparse
import json
import os
from contextlib import nullcontext

import torch
from PIL import Image

from .calibration import load_calibration_artifact
from .detector_runtime import FractureDetectorRuntime
from .models.bonevqa import BoneVQAModel
from .preprocessing import canonicalize_raster
from .utils import get_device, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()


def parse_boxes(text):
    if not text:
        return None
    boxes = []
    for part in text.split(";"):
        vals = [float(v) for v in part.replace(" ", "").split(",") if v]
        if len(vals) == 4:
            boxes.append(vals)
    return boxes or None


def main():
    parser = argparse.ArgumentParser(description="Dự đoán câu trả lời cho 1 ảnh X-quang + câu hỏi")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--boxes", default="", help="x1,y1,x2,y2;x1,y1,x2,y2")
    parser.add_argument("--detector_checkpoint", default=None,
                        help="detector v3 bắt buộc cho automatic V khi không truyền --boxes")
    parser.add_argument("--fracture_calibration", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--prompt_kind", default="contour", choices=["contour", "box", "circle", "mask"])
    parser.add_argument("--out_dir", default="reports/predictions")
    args = parser.parse_args()

    device = get_device()
    override = {}
    if args.fracture_calibration:
        artifact = load_calibration_artifact(args.fracture_calibration, expected_checkpoint=args.checkpoint)
        override.update({"fracture_platt_slope": artifact["platt_slope"],
                         "fracture_platt_intercept": artifact["platt_intercept"],
                         "fracture_yes_threshold": artifact["fracture_yes_threshold"]})
    model = BoneVQAModel.load(args.checkpoint, device=str(device), cfg_override=override or None)
    with Image.open(args.image) as raw:
        image = canonicalize_raster(raw)
    boxes = parse_boxes(args.boxes)
    if boxes is None and model.cfg.get("use_visual_prompt", False):
        if not args.detector_checkpoint:
            raise RuntimeError("Automatic V requires --detector_checkpoint when --boxes is absent")
        detector = FractureDetectorRuntime(args.detector_checkpoint, str(device))
        boxes = detector.predict(image)
    amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with amp_context:
        res = model.answer(image, args.question, boxes=boxes, prompt_kind=args.prompt_kind, region=args.region)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    prompt_path = os.path.join(args.out_dir, f"{stem}_prompt_{args.prompt_kind}.png")
    lens_path = os.path.join(args.out_dir, f"{stem}_lens.png")
    res["prompt_image"].save(prompt_path)
    res["lens_image"].save(lens_path)
    n_masks = 0 if res["masks"] is None else len(res["masks"])
    summary = {"question": args.question, "answer": res["answer"], "answer_type": res["answer_type"],
               "confidence": round(float(res["confidence"]), 4), "n_masks": n_masks,
               "detector_boxes": boxes,
               "prompt_image": prompt_path, "lens_image": lens_path}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Đã lưu ảnh prompt: %s và lens: %s", prompt_path, lens_path)


if __name__ == "__main__":
    main()
