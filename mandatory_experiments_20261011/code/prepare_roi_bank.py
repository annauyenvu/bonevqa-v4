#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from runtime import EXP4_CODE, LEGACY, PROJECT, ROOT, configure


def main() -> None:
    common, engine, specs = configure()
    common.setup(42)
    detector = common.read(LEGACY / "data/fracatlas/box_bo_phat_hien.json")
    image_rows = common.read(ROOT / "protocol/image_content_manifest.json")
    old_shared = PROJECT / "exp4_multiseed/roi_bank/shared"
    new_shared = ROOT / "roi_bank/shared"
    new_shared.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "roi_bank/detector"
    folder.mkdir(parents=True, exist_ok=True)
    spec = common.read(PROJECT / "exp4_multiseed/roi_bank/generation_spec.json")
    common.write(ROOT / "roi_bank/generation_spec.json", spec)

    from src.models.segment_prompt import SegmentPromptCreator
    cfg = common.read(ROOT / "protocol/base_config.json")["model"]
    model = SegmentPromptCreator(model_id=cfg["sam"]["model_id"], device="mps", max_masks=cfg["n_masks_max"],
                                 grid=cfg["sam"]["grid"], default_prompt=cfg["sam"]["default_prompt"],
                                 cache_dir=None, enabled=True, max_area_ratio=.6, min_mean_intensity=40.0)
    original_interpolate = F.interpolate
    counter = {"calls": 0}

    def interpolate(input, *args, **kwargs):
        mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
        if input.device.type == "mps" and input.ndim == 3 and mode == "linear":
            counter["calls"] += 1
            return original_interpolate(input.cpu(), *args, **kwargs).to(input.device)
        return original_interpolate(input, *args, **kwargs)

    F.interpolate = interpolate
    rows = []
    reused = generated = 0
    for index, image_meta in enumerate(image_rows, 1):
        image_id = image_meta["image_id"]
        path = Path(image_meta["path"])
        assert common.sha(path) == image_meta["sha256"]
        boxes = detector[image_id] or None
        with Image.open(path) as source:
            needs_exif_transpose = source.size != (image_meta["width"], image_meta["height"])
        key_payload = {"image_sha256": image_meta["sha256"], "boxes": boxes, "spec": spec}
        if needs_exif_transpose:
            key_payload["orientation"] = "exif_transpose"
        key = common.jhash(key_payload)
        old_npz, old_json = old_shared / f"{key}.npz", old_shared / f"{key}.json"
        new_npz, new_json = new_shared / f"{key}.npz", new_shared / f"{key}.json"
        if old_npz.exists() and old_json.exists():
            mask_path, audit_path = old_npz, old_json
            reused += 1
        elif new_npz.exists() and new_json.exists():
            mask_path, audit_path = new_npz, new_json
            generated += 1
        else:
            with Image.open(path) as source:
                image = (ImageOps.exif_transpose(source) if needs_exif_transpose else source).convert("RGB")
            assert image.size == (image_meta["width"], image_meta["height"])
            calls_before = counter["calls"]
            started = time.time()
            with torch.inference_mode():
                masks = model(image, boxes=boxes, cache_key=None)
            assert masks.ndim == 3 and masks.shape[1:] == (image.height, image.width) and len(masks) <= 3
            assert np.isin(masks, [0, 1]).all()
            temp = new_npz.with_suffix(".tmp.npz")
            np.savez_compressed(temp, masks=masks)
            temp.replace(new_npz)
            audit = {
                "image_id_source": image_id,
                "mask_sha256": common.sha(new_npz),
                "mask_count": len(masks),
                "roi_success": True,
                "roi_failed": 0,
                "sam_fallback_count": int(counter["calls"] > calls_before),
                "sam_fallback_operator_calls": counter["calls"] - calls_before,
                "seconds": time.time() - started,
                "key": key,
            }
            common.write(new_json, audit)
            mask_path, audit_path = new_npz, new_json
            generated += 1
        audit = common.read(audit_path)
        assert common.sha(mask_path) == audit["mask_sha256"]
        with np.load(mask_path) as data:
            masks = data["masks"]
            assert masks.shape[1:] == (image_meta["height"], image_meta["width"])
        rows.append({
            "image_id": image_id,
            "path": str(mask_path),
            "boxes": boxes,
            "width": image_meta["width"],
            "height": image_meta["height"],
            **{name: audit[name] for name in ("mask_sha256", "mask_count", "roi_success", "roi_failed",
                                               "sam_fallback_count", "sam_fallback_operator_calls")},
        })
        if index % 10 == 0 or index == len(image_rows):
            common.write(folder / "progress.json", {"processed_images": index, "expected_images": len(image_rows),
                                                      "reused": reused, "generated": generated, "roi_failed": 0})
            common.status("roi_preflight", processed_images=index, expected_images=len(image_rows),
                          reused=reused, generated=generated)
            print("ROI", index, "/", len(image_rows), "reused", reused, "generated", generated, flush=True)

    common.write(folder / "index.json", rows)
    manifest_ids = {split: {row["image_id"] for row in common.read(ROOT / f"protocol/{split}_manifest.json")} for split in specs}
    split_audit = {}
    for split, ids in manifest_ids.items():
        selected = [row for row in rows if row["image_id"] in ids]
        split_audit[split] = {
            "expected_images": specs[split][1],
            "processed_images": len(selected),
            "roi_success": len(selected),
            "roi_failed": 0,
            "roi_nonempty": sum(row["mask_count"] > 0 for row in selected),
            "roi_empty_valid_after_filters": sum(row["mask_count"] == 0 for row in selected),
            "mask_count": sum(row["mask_count"] for row in selected),
        }
        assert len(selected) == specs[split][1]
    bank_audit = {
        "regime": "detector",
        "expected_images": len(image_rows),
        "processed_images": len(rows),
        "roi_success": len(rows),
        "roi_failed": 0,
        "reused_from_exp4": reused,
        "generated_for_component_clean": generated,
        "splits": split_audit,
        "index_sha256": common.sha(folder / "index.json"),
    }
    common.write(folder / "audit.json", bank_audit)
    common.write(ROOT / "roi_bank/COMPLETE.json", {"roi_failed": 0, "detector": bank_audit})
    common.status("roi_preflight", "complete", reused=reused, generated=generated)
    print("ROI BANK COMPLETE", bank_audit, flush=True)


if __name__ == "__main__":
    main()
