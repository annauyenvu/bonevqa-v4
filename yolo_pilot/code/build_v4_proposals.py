"""YOLOv8 predictions -> proposal file in the V4 schema (vg-proposals-v1).

    python3 -I code/build_v4_proposals.py 0 insample [fa|joint]   # held-out roles + in-sample TRAIN
    python3 -I code/build_v4_proposals.py 0 full [fa|joint]       # + cross-fitted TRAIN, + audit

``fa`` is the FracAtlas-only detector, ``joint`` the FracAtlas + BTXRD detector (only its
fracture class is used).  Both use the same inner split.

Contract (the one V4 uses): at most 3 boxes per image, highest score first, score >= a
threshold.  One rule sets every threshold: the smallest threshold at which the share of
non-fractured model_val images that get a proposal is no more than that of the sealed V4
Faster R-CNN at 0.05 on the same images.  It is applied to the outer detector and to
each inner (cross-fit) detector separately, so every detector hands the VQA model
proposals at the same rate on images it has not seen.  Only model_val is used.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_detectors import contract, per_image, summarise, threshold_for_rate  # noqa: E402

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
PILOT = ROOT / "yolo_pilot"
ROLE = {"val": "model_val", "cal": "calibration", "test": "test"}
VARIANTS = {"fa": {"outer": "fa_only_cls0", "train": "fa_only_train_cls0", "cf": "cf{k}_cls0", "dir": "crossfit_yolo",
                   "detector": "yolov8s_fracatlas_os3"},
            "joint": {"outer": "fa_btxrd_cls0", "train": "fa_btxrd_train_cls0", "cf": "cfj{k}_cls0",
                      "dir": "crossfit_yolo_joint", "detector": "yolov8s_fracatlas_os3_btxrd_fracture_class"}}


PREFIX = ""   # prediction files of fold 0 have no prefix; those of fold k carry "f<k>_" (set in main)


def boxes(name):
    return json.loads((PILOT / f"runs/pred/{PREFIX}{name}.json").read_text())["boxes"]


def stats(pred, gt, ids, thr):
    rows = per_image(pred, gt, ids, thr)
    s = summarise(rows, ids)
    fx = [i for i in ids if gt[i]["fractured"]]
    s["best_iou_quantiles_fractured"] = np.quantile([rows[i]["best_iou"] for i in fx], [.1, .25, .5, .75, .9]).round(3).tolist()
    s["images"], s["fractured"] = len(ids), len(fx)
    return s


def main(fold, stage, variant="fa"):
    global PREFIX
    PREFIX = "" if fold == 0 else f"f{fold}_"
    V = VARIANTS[variant]
    gt = json.loads((PILOT / f"datasets/fa_fold{fold}/gt.json").read_text())
    ids = {r: sorted(i for i, g in gt.items() if g["role"] == r) for r in ("train", "val", "cal", "test")}
    out_dir = ROOT / f"v4_grounded_next/runs/{V['dir']}/fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = json.loads((PILOT / f"runs/frcnn/fold{fold}_ep2_sealed.json").read_text())["boxes"]
    target = summarise(per_image(ref, gt, ids["val"], 0.05), ids["val"])["m_rate_nonfractured"]

    outer = boxes(V["outer"])
    t_outer = threshold_for_rate(outer, gt, ids["val"], target)
    insample = boxes(V["train"])
    assert set(ids["train"]) <= set(insample) and all(set(ids[r]) <= set(outer) for r in ROLE)
    doc = {"schema": "vg-proposals-v1", "outer_fold": str(fold), "detector": V["detector"],
           "contract": "top-3 boxes with score >= threshold, canonical-image xyxy + score. model_val/calibration/test "
                       "and train_insample_audit_only: outer YOLOv8s (full TRAIN). train: out-of-fold inner YOLOv8s.",
           "target_nonfractured_rate_model_val": target, "thresholds": {"outer": t_outer},
           "train_insample_audit_only": {i: contract(insample, i, t_outer) for i in ids["train"]}}
    for short, role in ROLE.items():
        doc[role] = {i: contract(outer, i, t_outer) for i in ids[short]}
    audit = {"target_nonfractured_rate_model_val": target, "thresholds": doc["thresholds"],
             "train_insample": stats(insample, gt, ids["train"], t_outer),
             **{role: stats(outer, gt, ids[short], t_outer) for short, role in ROLE.items()}}

    if stage == "full":
        split = json.loads((ROOT / f"v4_grounded_next/runs/crossfit_yolo/fold_{fold}/inner_split.json").read_text())["inner"]
        train_cf, merged_scores = {}, {}
        for k, held in split.items():
            pk = boxes(V["cf"].format(k=k))
            assert set(held) <= set(pk) and set(ids["val"]) <= set(pk)
            tk = threshold_for_rate(pk, gt, ids["val"], target)
            doc["thresholds"][f"inner{k}"] = tk
            audit[f"inner{k}_on_model_val"] = stats(pk, gt, ids["val"], tk)
            audit[f"inner{k}_on_heldout"] = stats(pk, gt, sorted(held), tk)
            for i in held:
                assert i not in train_cf
                train_cf[i] = contract(pk, i, tk)
        assert sorted(train_cf) == ids["train"]
        doc["train"] = train_cf
        fx = [i for i in ids["train"] if gt[i]["fractured"]]
        nf = [i for i in ids["train"] if not gt[i]["fractured"]]
        from eval_detectors import iou
        best = [max((iou(p, g) for p in train_cf[i] for g in gt[i]["boxes"]), default=0.0) for i in fx]
        audit["train_cf"] = {"images": len(ids["train"]), "fractured": len(fx),
                             "m_rate_fractured": float(np.mean([bool(train_cf[i]) for i in fx])),
                             "m_rate_nonfractured": float(np.mean([bool(train_cf[i]) for i in nf])),
                             "mean_props_nonfractured": float(np.mean([len(train_cf[i]) for i in nf])),
                             **{f"recall_top3_iou{t}": float(np.mean([b >= t for b in best])) for t in (0.1, 0.3, 0.5)},
                             "best_iou_quantiles_fractured": np.quantile(best, [.1, .25, .5, .75, .9]).round(3).tolist()}
    name = f"proposals_fold{fold}.json" if stage == "full" else f"proposals_fold{fold}_insample.json"
    with (out_dir / name).open("x") as f:
        json.dump(doc, f)
    with (out_dir / f"audit_{stage}.json").open("x") as f:
        json.dump(audit, f, indent=2)
    keys = ("m_rate_fractured", "m_rate_nonfractured", "recall_top3_iou0.3", "recall_top3_iou0.5")
    print("thresholds", {k: round(v, 4) for k, v in doc["thresholds"].items()})
    for k, v in audit.items():
        if isinstance(v, dict) and "m_rate_fractured" in v:
            print(f"{k:24s}", {a: round(v[a], 3) for a in keys})


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "fa")
