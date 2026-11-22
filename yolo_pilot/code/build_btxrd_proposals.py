"""Lesion-class predictions of the joint YOLOv8 detector -> proposal file for the BTXRD images.

    venv/bin/python -I code/build_btxrd_proposals.py

Same contract as the FracAtlas files (vg-proposals-v1): at most 3 boxes per image, highest
score first, score >= one threshold.  The threshold follows the rule used for the fracture
class: the smallest threshold at which the share of NORMAL BTXRD val images that get a lesion
proposal is no more than the target rate stored in the FracAtlas proposal file (the rate of the
sealed V4 Faster R-CNN on non-fractured FracAtlas val images).  Only val is used to set it.

Proposals are question-routed: BTXRD rows (tumour questions) get lesion-class boxes, FracAtlas
rows (fracture questions) keep their fracture-class boxes.  Every BTXRD image has EXIF
orientation 1, so detector coordinates are already canonical-image coordinates.

TRAIN proposals here are IN-SAMPLE (the detector was trained on these images); there is no
cross-fitted key yet, so the file has ``train_insample_audit_only`` and no ``train``.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_detectors import contract, per_image, summarise, threshold_for_rate  # noqa: E402

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
PILOT = ROOT / "yolo_pilot"
CFJ = ROOT / "v4_grounded_next/runs/crossfit_yolo_joint"
ROLE = {"val": "model_val", "test": "test", "external": "external"}


def main(fold=0, level=None):
    # fold 0 keeps its original names; fold k: predictions f<k>_..., output btxrd_fold<k>/
    # level (added 2026-11-20, final configuration): target normal rate of YOLOv8's own threshold rule (e.g. 0.02)
    # instead of the Faster R-CNN rate; output crossfit_yolo_joint_sweep/r<level>/btxrd_fold<k>/
    pre = "" if fold == 0 else f"f{fold}_"
    OUT = CFJ / ("btxrd" if fold == 0 else f"btxrd_fold{fold}")
    if level is not None:
        OUT = ROOT / f"v4_grounded_next/runs/crossfit_yolo_joint_sweep/r{level}/btxrd_fold{fold}"
    FA_PROPOSALS = CFJ / f"fold_{fold}/proposals_fold{fold}_insample.json"
    raw = json.loads((PILOT / "datasets/btxrd_pilot/gt.json").read_text())
    # eval_detectors speaks of "fractured" images; here the positive class is "has a tumour box".
    gt = {i: {"fractured": bool(g["tumor"]), "boxes": [b[:4] for b in g["boxes"]], "role": g["role"]} for i, g in raw.items()}
    ids = {r: sorted(i for i, g in gt.items() if g["role"] == r) for r in ("train", "val", "test", "external")}
    held = json.loads((PILOT / f"runs/pred/{pre}fa_btxrd_cls1.json").read_text())["boxes"]
    train = json.loads((PILOT / f"runs/pred/{pre}fa_btxrd_bttrain_cls1.json").read_text())["boxes"]
    assert set(ids["train"]) <= set(train) and all(set(ids[r]) <= set(held) for r in ROLE)
    target = json.loads(FA_PROPOSALS.read_text())["target_nonfractured_rate_model_val"] if level is None else float(level)
    thr = threshold_for_rate(held, gt, ids["val"], target)
    doc = {"schema": "vg-proposals-v1", "dataset": "btxrd", "detector": "yolov8s_fracatlas_os3_btxrd_lesion_class",
           "contract": "top-3 lesion-class boxes with score >= threshold, canonical-image xyxy + score; outer YOLOv8s "
                       f"(trained on FracAtlas fold-{fold} TRAIN + BTXRD TRAIN). train_insample_audit_only is in-sample.",
           "target_normal_rate_val": target, "thresholds": {"outer": thr},
           "train_insample_audit_only": {i: contract(train, i, thr) for i in ids["train"]}}
    audit = {"target_normal_rate_val": target, "thresholds": doc["thresholds"]}
    for short, pred in (("train", train), ("val", held), ("test", held), ("external", held)):
        if short in ROLE:
            doc[ROLE[short]] = {i: contract(pred, i, thr) for i in ids[short]}
        s = summarise(per_image(pred, gt, ids[short], thr), ids[short])
        s["images"], s["tumor"] = len(ids[short]), sum(gt[i]["fractured"] for i in ids[short])
        audit["train_insample" if short == "train" else ROLE[short]] = s
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "proposals_btxrd_insample.json").open("x") as f:
        json.dump(doc, f)
    with (OUT / "audit_insample.json").open("x") as f:
        json.dump(audit, f, indent=2)
    print("threshold", round(thr, 4), "target normal rate", round(target, 4))
    for k, v in audit.items():
        if isinstance(v, dict) and "m_rate_fractured" in v:
            print(f"{k:16s} images {v['images']:5d} tumour {v['tumor']:4d}  proposal rate tumour {v['m_rate_fractured']:.3f} "
                  f"normal {v['m_rate_nonfractured']:.3f}  recall top-3 IoU>=0.3 {v['recall_top3_iou0.3']:.3f} "
                  f"IoU>=0.5 {v['recall_top3_iou0.5']:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, sys.argv[2] if len(sys.argv) > 2 else None)
