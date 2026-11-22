"""Cross-dataset checks for the two-class detector (fracture = 0, bone_lesion = 1).

    python eval_cross.py <fa_report.json> <out.json>

* lesion box AP50 on BTXRD test (centre 1, pseudo-patient split) and external (centres 2-3);
* how often the FRACTURE class fires (top-3 contract, threshold matched on FracAtlas val)
  on BTXRD tumour / normal films, for the FracAtlas-only and the joint detector;
* how often the LESION class fires (conf >= 0.25) on FracAtlas test fractured / non-fractured;
* source shortcut: image-level AUROC of the lesion score for 'image comes from BTXRD'
  among NORMAL images only (BTXRD normal vs FracAtlas non-fractured, test roles).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_detectors import auroc, box_ap, iou  # noqa: E402

ROOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")


def load(name):
    return json.loads((ROOT / f"runs/pred/{name}.json").read_text())["boxes"]


def smax(pred, iid):
    return max((b[4] for b in pred.get(iid, [])), default=0.0)


def main():
    fa_report = json.loads(Path(sys.argv[1]).read_text())
    fa = json.loads((ROOT / "datasets/fa_fold0/gt.json").read_text())
    bt = json.loads((ROOT / "datasets/btxrd_pilot/gt.json").read_text())
    for g in bt.values():
        g["boxes"] = [b[:4] for b in g["boxes"]]
    out = {}

    lesion = load("fa_btxrd_cls1")
    out["lesion_box_AP50"] = {}
    for role in ("val", "test", "external"):
        ids = sorted(i for i, g in bt.items() if g["role"] == role)
        tum = [i for i in ids if bt[i]["tumor"]]
        rec = float(np.mean([max((iou(p, g) for p in sorted(lesion.get(i, []), key=lambda b: -b[4])[:3]
                                  if p[4] >= 0.25 for g in bt[i]["boxes"]), default=0.0) >= 0.5 for i in tum]))
        out["lesion_box_AP50"][role] = {
            "images": len(ids), "tumor_images": len(tum), "AP50": box_ap(lesion, bt, ids, 0.5),
            "recall_top3_iou0.5_conf0.25": rec,
            "fire_rate_normal_conf0.25": float(np.mean([smax(lesion, i) >= 0.25 for i in ids if not bt[i]["tumor"]])),
            "image_AUROC_tumor": auroc([smax(lesion, i) for i in ids], [bt[i]["tumor"] for i in ids])}

    out["fracture_class_on_btxrd"] = {}
    for det, pred_name in (("yolo_fa", "fa_only_on_btxrd_cls0"), ("yolo_fa_btxrd", "fa_btxrd_cls0")):
        pred = load(pred_name)
        thr = fa_report["detectors"][det]["threshold_matched_on_val"]
        entry = {"threshold": thr}
        for role in ("test", "external"):
            ids = [i for i, g in bt.items() if g["role"] == role]
            entry[role] = {
                "fire_rate_tumor": float(np.mean([smax(pred, i) >= thr for i in ids if bt[i]["tumor"]])),
                "fire_rate_normal": float(np.mean([smax(pred, i) >= thr for i in ids if not bt[i]["tumor"]])),
                "fracture_box_on_lesion_iou0.3": float(np.mean([
                    any(iou(p, g) >= 0.3 for p in pred.get(i, []) if p[4] >= thr for g in bt[i]["boxes"])
                    for i in ids if bt[i]["tumor"]]))}
        out["fracture_class_on_btxrd"][det] = entry

    lesion_fa = lesion
    test = [i for i, g in fa.items() if g["role"] == "test"]
    out["lesion_class_on_fracatlas_test_conf0.25"] = {
        "fire_rate_fractured": float(np.mean([smax(lesion_fa, i) >= 0.25 for i in test if fa[i]["fractured"]])),
        "fire_rate_nonfractured": float(np.mean([smax(lesion_fa, i) >= 0.25 for i in test if not fa[i]["fractured"]])),
        "lesion_box_on_fracture_iou0.3": float(np.mean([
            any(iou(p, g) >= 0.3 for p in lesion_fa.get(i, []) if p[4] >= 0.25 for g in fa[i]["boxes"])
            for i in test if fa[i]["fractured"]]))}

    normal_bt = [i for i, g in bt.items() if g["role"] == "test" and not g["tumor"]]
    normal_fa = [i for i in test if not fa[i]["fractured"]]
    frac = load("fa_btxrd_cls0")
    out["source_shortcut_normals_only"] = {
        "n_btxrd_normal": len(normal_bt), "n_fracatlas_nonfractured": len(normal_fa),
        "AUROC_lesion_score_predicts_btxrd": auroc([smax(lesion, i) for i in normal_bt + normal_fa],
                                                   [1] * len(normal_bt) + [0] * len(normal_fa)),
        "AUROC_fracture_score_predicts_fracatlas": auroc([smax(frac, i) for i in normal_bt + normal_fa],
                                                         [0] * len(normal_bt) + [1] * len(normal_fa))}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
