"""Export detector proposals for every component-clean image."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from src.detector_protocol import load_detector_manifest, verify_detector_checkpoint_binding
from torch.utils.data import DataLoader

from src.preprocessing import RASTER_PREPROCESSING_VERSION, canonicalize_raster


def load_detector_module():
    path = Path(__file__).with_name("train_detector_v2.py")
    spec = importlib.util.spec_from_file_location("detector_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split_manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--max_boxes", type=int, default=3)
    # Dynamic module loading is intentionally single-process on macOS spawn;
    # detector inference itself still uses all configured torch CPU threads.
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--torch_threads", type=int, default=10)
    parser.add_argument("--max_images_per_split", type=int,
                        help="chỉ dùng cho smoke; artifact sẽ bị đánh dấu incomplete")
    args = parser.parse_args()
    module = load_detector_module()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("preprocessing_version") != RASTER_PREPROCESSING_VERSION:
        raise RuntimeError(
            "Detector checkpoint không dùng EXIF-canonical preprocessing v3; phải retrain trước khi export"
        )
    torch.set_num_threads(args.torch_threads)
    device = module.get_device() if args.device == "auto" else torch.device(args.device)
    # The checkpoint contains every detector tensor; avoid an unnecessary
    # ImageNet/COCO weight download during sealed proposal export.
    model = module.tao_mo_hinh(payload["architecture"], weights=None).to(device)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    split_payload, split_ids, split_sha = load_detector_manifest(args.split_manifest)
    verify_detector_checkpoint_binding(payload, split_payload, split_sha)
    output = {}
    provenance = {}
    with torch.no_grad():
        for split in split_ids:
            dataset = module.BoPhatHienDataset(split, canh_dai=payload["canh_dai"],
                                               allowed_ids=set(split_ids[split]))
            if args.max_images_per_split is not None:
                dataset.muc = dataset.muc[:args.max_images_per_split]
            records_by_id = {item[0]: item for item in dataset.muc}
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=module.collate,
                                num_workers=args.num_workers)
            count = 0
            for images, _, image_ids in loader:
                predictions = model([image.to(device) for image in images])
                for prediction, image_id, image in zip(predictions, image_ids, images):
                    record = records_by_id[image_id]
                    from PIL import Image
                    with Image.open(Path(dataset.root) / record[1][0]) as original:
                        original_w, original_h = canonicalize_raster(original).size
                    scale = payload["canh_dai"] / max(original_w, original_h)
                    resized_w = max(1, int(round(original_w * scale)))
                    resized_h = max(1, int(round(original_h * scale)))
                    pad_x = (payload["canh_dai"] - resized_w) // 2
                    pad_y = (payload["canh_dai"] - resized_h) // 2
                    boxes = []
                    for box, score in zip(prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist()):
                        if score < args.score or len(boxes) >= args.max_boxes:
                            continue
                        x1 = max(0.0, min(float(original_w), (box[0] - pad_x) / scale))
                        y1 = max(0.0, min(float(original_h), (box[1] - pad_y) / scale))
                        x2 = max(0.0, min(float(original_w), (box[2] - pad_x) / scale))
                        y2 = max(0.0, min(float(original_h), (box[3] - pad_y) / scale))
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                    output[image_id] = boxes
                    provenance[image_id] = {"split": split, "scores_above_threshold": len(boxes),
                                            "width": original_w, "height": original_h}
                    count += 1
            print(f"{split}: {count} images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = json.dumps(output, indent=2).encode("utf-8")
    args.output.write_bytes(output_bytes)
    artifact = {
        "schema_version": 3,
        "coordinate_space": "exif_canonical_pixels",
        "preprocessing_version": RASTER_PREPROCESSING_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "proposal_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "score_threshold": args.score,
        "max_boxes": args.max_boxes,
        "coverage": len(output),
        "expected_coverage": len(set().union(*(set(ids) for ids in split_ids.values()))),
        "training_scope": payload.get("training_scope"),
        "training_split_sha256": payload.get("training_split_sha256"),
        "complete": args.max_images_per_split is None,
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps({"artifact": artifact, "images": provenance}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
