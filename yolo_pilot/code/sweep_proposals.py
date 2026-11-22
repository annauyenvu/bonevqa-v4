"""Proposal-threshold sweep of the joint YOLOv8 detectors (fold 0), approved 2026-11-19.

    python3 -I code/sweep_proposals.py 0 0.02      # target non-fractured proposal rate on model_val
    python3 -I code/sweep_proposals.py 0 max       # every box YOLOv8 outputs (score >= 0.001) on the outer detector

Same contract and same rule as build_v4_proposals.py (top-3 boxes, one threshold per detector, set on model_val so
that the share of non-fractured model_val images with a proposal is <= the target), only the target changes: it is
no longer the rate of the sealed Faster R-CNN.  "max" = the rate of the outer detector with no threshold; the inner
(cross-fit) detectors are matched to that same rate, so TRAIN and the held-out roles see proposals at the same rate.
Writes v4_grounded_next/runs/crossfit_yolo_joint_sweep/r<target>/fold_<k>/{proposals_fold<k>.json, audit_full.json}.
The level is chosen later on model_val (V4 AUPRC there); test is only reported.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_v4_proposals as B  # noqa: E402
from eval_detectors import contract, iou, per_image, summarise, threshold_for_rate  # noqa: E402


def main(fold, level):
    B.PREFIX = "" if fold == 0 else f"f{fold}_"
    V = B.VARIANTS["joint"]
    gt = json.loads((B.PILOT / f"datasets/fa_fold{fold}/gt.json").read_text())
    ids = {r: sorted(i for i, g in gt.items() if g["role"] == r) for r in ("train", "val", "cal", "test")}
    outer = B.boxes(V["outer"])
    if level == "max":
        target = summarise(per_image(outer, gt, ids["val"], 0.0), ids["val"])["m_rate_nonfractured"]
    else:
        target = float(level)
    out_dir = B.ROOT / f"v4_grounded_next/runs/crossfit_yolo_joint_sweep/r{level}/fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_outer = threshold_for_rate(outer, gt, ids["val"], target)
    doc = {"schema": "vg-proposals-v1", "outer_fold": str(fold), "detector": V["detector"],
           "contract": "top-3 boxes with score >= threshold, canonical-image xyxy + score. model_val/calibration/test: "
                       "outer YOLOv8s (full TRAIN). train: out-of-fold inner YOLOv8s. Threshold sweep level " + level,
           "sweep_level": level, "target_nonfractured_rate_model_val": target, "thresholds": {"outer": t_outer}}
    insample = B.boxes(V["train"])   # read by FoldData under the "insample_v4" key (eval_arm), as in every proposal file
    assert set(ids["train"]) <= set(insample) and all(set(ids[r]) <= set(outer) for r in B.ROLE)
    doc["train_insample_audit_only"] = {i: contract(insample, i, t_outer) for i in ids["train"]}
    for short, role in B.ROLE.items():
        doc[role] = {i: contract(outer, i, t_outer) for i in ids[short]}
    audit = {"sweep_level": level, "target_nonfractured_rate_model_val": target, "thresholds": doc["thresholds"],
             **{role: B.stats(outer, gt, ids[short], t_outer) for short, role in B.ROLE.items()}}
    split = json.loads((B.ROOT / f"v4_grounded_next/runs/crossfit_yolo/fold_{fold}/inner_split.json").read_text())["inner"]
    train_cf = {}
    for k, held in split.items():
        pk = B.boxes(V["cf"].format(k=k))
        assert set(held) <= set(pk) and set(ids["val"]) <= set(pk)
        tk = threshold_for_rate(pk, gt, ids["val"], target)
        doc["thresholds"][f"inner{k}"] = tk
        audit[f"inner{k}_on_model_val"] = B.stats(pk, gt, ids["val"], tk)
        audit[f"inner{k}_on_heldout"] = B.stats(pk, gt, sorted(held), tk)
        for i in held:
            assert i not in train_cf
            train_cf[i] = contract(pk, i, tk)
    assert sorted(train_cf) == ids["train"]
    doc["train"] = train_cf
    fx = [i for i in ids["train"] if gt[i]["fractured"]]
    nf = [i for i in ids["train"] if not gt[i]["fractured"]]
    best = [max((iou(p, g) for p in train_cf[i] for g in gt[i]["boxes"]), default=0.0) for i in fx]
    audit["train_cf"] = {"images": len(ids["train"]), "fractured": len(fx),
                         "m_rate_fractured": float(np.mean([bool(train_cf[i]) for i in fx])),
                         "m_rate_nonfractured": float(np.mean([bool(train_cf[i]) for i in nf])),
                         "mean_props_nonfractured": float(np.mean([len(train_cf[i]) for i in nf])),
                         **{f"recall_top3_iou{t}": float(np.mean([b >= t for b in best])) for t in (0.1, 0.3, 0.5)}}
    with (out_dir / f"proposals_fold{fold}.json").open("x") as f:
        json.dump(doc, f)
    with (out_dir / "audit_full.json").open("x") as f:
        json.dump(audit, f, indent=2)
    keys = ("m_rate_fractured", "m_rate_nonfractured", "recall_top3_iou0.3", "recall_top3_iou0.5")
    print("level", level, "target", round(target, 4), "thresholds", {k: round(v, 4) for k, v in doc["thresholds"].items()})
    for k, v in audit.items():
        if isinstance(v, dict) and "m_rate_fractured" in v:
            print(f"{k:24s}", {a: round(v[a], 3) for a in keys})


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
