import argparse
import json
import os

import torch
from PIL import Image

from .models.bonevqa import BoneVQAModel
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
    parser.add_argument("--region", default=None)
    parser.add_argument("--prompt_kind", default="contour", choices=["contour", "box", "circle", "mask"])
    parser.add_argument("--out_dir", default="reports/predictions")
    args = parser.parse_args()

    device = get_device()
    model = BoneVQAModel.load(args.checkpoint, device=str(device))
    image = Image.open(args.image).convert("RGB")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        res = model.answer(image, args.question, boxes=parse_boxes(args.boxes), prompt_kind=args.prompt_kind, region=args.region)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    prompt_path = os.path.join(args.out_dir, f"{stem}_prompt_{args.prompt_kind}.png")
    lens_path = os.path.join(args.out_dir, f"{stem}_lens.png")
    res["prompt_image"].save(prompt_path)
    res["lens_image"].save(lens_path)
    n_masks = 0 if res["masks"] is None else len(res["masks"])
    summary = {"question": args.question, "answer": res["answer"], "answer_type": res["answer_type"],
               "confidence": round(float(res["confidence"]), 4), "n_masks": n_masks,
               "prompt_image": prompt_path, "lens_image": lens_path}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Đã lưu ảnh prompt: %s và lens: %s", prompt_path, lens_path)


if __name__ == "__main__":
    main()
