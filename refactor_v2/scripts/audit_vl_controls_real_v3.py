"""TRAIN-only frozen-weight ROI/pixel mechanics; not matched efficacy."""
import copy
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.diagnose_p_controls_real_v3 import state_hash
from scripts.run_grounding_interventions_v3 import box_mask, intervene, matched_boxes
from src.data import collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset, FRACATLAS_ROOT
from src.data.sealed_manifest import load_sealed_records
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.preprocessing import canonicalize_raster_and_boxes
from src.provenance import validate_detector_proposals
from src.train import seed_training_epoch
from src.utils import load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("refactor_v2/reports/vl_controls_real_mechanics_v3.json"))
    output = parser.parse_args().output
    if output.exists():
        raise RuntimeError("Existing control audit must be inspected, not overwritten")
    root = Path.cwd()
    loader = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")["data"]["loader_kwargs"]
    parent = Path(loader["sealed_records_by_split"]["train"])
    source = load_sealed_records(parent, loader["seal_path"], root)
    proposal_path = Path("refactor_v2/detector_fold0_train_only_smoke_v3/boxes.json")
    validate_detector_proposals(proposal_path, require_complete=False)
    proposals = json.loads(proposal_path.read_text())
    rows = [row for row in source if row["image_id"] in proposals]
    if {row["image_id"] for row in rows} != set(proposals):
        raise RuntimeError("Smoke proposals include non-TRAIN images")
    # Explicit derivative of the sealed source; original manifests unchanged.
    subset = root / "refactor_v2/reports/roi_control_train_only_rows_v3.json"
    subset.write_text(json.dumps(rows, indent=2) + "\n")
    seal = root / "refactor_v2/reports/roi_control_train_only_data_seal_v3.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "scope": "technical_smoke_train_only",
                               "parent_manifest": str(parent), "parent_manifest_sha256": sha(parent),
                               "parent_seal_sha256": sha(loader["seal_path"]),
                               "artifacts": {subset.relative_to(root).as_posix(): sha(subset)}}, indent=2) + "\n")
    indices = json.loads(Path("refactor_v2/reports/vl_detector_routes_real_v3.json").read_text())["train_indices"]
    chosen_keys = [(source[i]["image_id"], source[i]["question"]) for i in indices]
    chosen = [next(i for i, row in enumerate(rows) if (row["image_id"], row["question"]) == key) for key in chosen_keys]
    batches, coverage, datasets = {}, {}, {}
    for control in ("real", "shuffled", "generic", "matched", "matched_all"):
        dataset = FracAtlasVQADataset(root=FRACATLAS_ROOT, sealed_records_path=subset, seal_path=seal,
                                     project_root=root, with_masks=False, with_boxes=False,
                                     boxes_from=proposal_path, boxes_as_masks=True, roi_control=control)
        datasets[control] = dataset
        unique = {row["image_id"]: i for i, row in enumerate(rows)}
        counts = {image: len(dataset[index]["boxes"] or []) for image, index in unique.items()}
        coverage[control] = {"images": len(counts), "with_roi": sum(value > 0 for value in counts.values()),
                             "proposal_counts": counts}
        batches[control] = normalize_batch(collate_fn([dataset[i] for i in chosen]))
    reference = batches["real"]
    matched = batches["matched"]
    geometry = []
    for original, shifted in zip(reference["boxes"], matched["boxes"]):
        for a, b in zip(original or [], shifted or []):
            geometry.append({"width_error": abs((round(a[2]) - round(a[0])) - (round(b[2]) - round(b[0]))),
                             "height_error": abs((round(a[3]) - round(a[1])) - (round(b[3]) - round(b[1]))),
                             "comparison_space": "clipped image-bounded raster pixels, not float coordinate differences",
                             "position_changed": a[:4] != b[:4]})
    pixel_audit = []
    gt_boxes = []
    for index, image in zip(chosen, reference["images"]):
        row = rows[index]
        with Image.open(FRACATLAS_ROOT / row["image"]) as raw:
            _, boxes = canonicalize_raster_and_boxes(raw, row.get("boxes") or [])
        gt_boxes.append(boxes)
        lesion = box_mask(boxes, image.width, image.height)
        relocation = box_mask(matched_boxes(boxes, image.width, image.height), image.width, image.height)
        pixel_audit.append({"image_id": row["image_id"], "lesion_pixels": int(lesion.sum()),
                            "matched_pixels": int(relocation.sum()),
                            "matched_overlap_pixels": int((lesion & relocation).sum())})
    for condition in ("lesion_mask", "matched_nonlesion_mask", "background_replacement"):
        changed = dict(reference)
        changed["images"] = [intervene(im, boxes, condition) for im, boxes in zip(reference["images"], gt_boxes)]
        batches[condition] = changed
    neutral = dict(reference)
    neutral["images"] = [intervene(im, boxes or [], "lesion_mask")
                          for im, boxes in zip(reference["images"], reference["boxes"])]
    batches["proposal_pixels_neutralized"] = neutral
    gray = dict(reference)
    gray["images"] = [Image.new("RGB", image.size, (128, 128, 128)) for image in reference["images"]]
    batches["geometry_only"] = gray
    results = {}
    for variant in ("v", "l"):
        checkpoint = Path(f"refactor_v2/checkpoints/gate_{variant}_detector_route_real_v3/smoke.pt")
        model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
        before = state_hash(model)

        @torch.no_grad()
        def evaluate(batch):
            model.eval()
            masks = model._prepare_masks(batch["images"], batch["masks"], batch["boxes"], batch["image_ids"])
            core = model._encode_core(batch["images"], batch["questions"], batch["regions"], masks,
                                      model.cfg["prompt_kind"], False)
            idx = [1, 3]
            embeds, attention, labels = model._build_llm_inputs(model._llm_prefix(core)[idx],
                [core["q_texts"][i] for i in idx], [batch["answers"][i] for i in idx])
            logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
            return {"closed": core["logits_closed"].detach(), "fracture": core["fracture_logit"].detach(),
                    "open": logits[:, :-1][labels[:, 1:] != -100].detach().float()}

        baseline = evaluate(reference)
        differences = {name: {key: float((baseline[key] - value).abs().max()) for key, value in evaluate(batch).items()}
                       for name, batch in batches.items()}
        alignments = {}
        if variant == "l":
            for control in ("real", "matched"):
                model.train()
                seed_training_epoch(42, 0)
                with torch.no_grad():
                    alignments[control] = float(model(batches[control], stage=3)["loss_lens_align"])
        results[variant.upper()] = {"endpoint_max_differences": differences,
                                   "lens_alignment_diagnostics": alignments,
                                   "trainable_state_unchanged": before == state_hash(model)}
        print(json.dumps({"variant": variant, **results[variant.upper()]}), flush=True)
        del model, baseline
        gc.collect()
    passed = (all(item["width_error"] < 1e-6 and item["height_error"] < 1e-6 and item["position_changed"] for item in geometry)
              and all(result["trainable_state_unchanged"] for result in results.values())
              and all(value == 0 for value in results["L"]["endpoint_max_differences"]["matched"].values())
              and any(value > 0 for value in results["V"]["endpoint_max_differences"]["matched"].values()))
    passed = passed and all(item["lesion_pixels"] == item["matched_pixels"]
                            and item["matched_overlap_pixels"] == 0 for item in pixel_audit)
    report = {"scope": "frozen two-step smoke models, TRAIN-only derivative, mechanics NOT quality",
              "parent_manifest_sha256": sha(parent), "subset_manifest_sha256": sha(subset),
              "images": len(proposals), "selected_qa": 4, "roi_control_coverage": coverage,
              "same_area_geometry_checks": geometry, "annotation_pixel_audit": pixel_audit,
              "models": results, "passed_technical_mechanics": passed, "seconds": time.perf_counter() - start,
              "limitations": ["ROI annotation used ONLY for audit pixel mutations, never prompt masks/boxes",
                              "pixel conditions keep detector proposals fixed; no detector re-inference",
                              "generic/matched_all change view counts; not compute-matched comparisons",
                              "no accuracy, clinical, lesion-grounding or superiority conclusion",
                              "matched nonlesion label valid only when recorded overlap is zero"]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
