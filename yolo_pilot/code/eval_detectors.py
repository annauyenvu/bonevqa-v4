"""Compare fracture detectors on one FracAtlas fold.

Predictions: {image_id: [[x1, y1, x2, y2, score], ...]} in canonical-image pixels,
fracture class only, any order.  Thresholds for the V4 proposal contract (top-3 boxes
above a score threshold) are chosen on model_val only; test is read once per detector.

    python eval_detectors.py <gt.json> <out.json> name=path[:fixed_threshold] ...
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
MANIFEST = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"
IOUS = (0.1, 0.3, 0.5)
TOPK = 3


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores, labels):
    """Area under the precision-recall curve (step-wise, as sklearn average_precision_score)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * labels).sum() / labels.sum())


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    order = np.argsort(allv, kind="stable")
    sorted_v = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def box_ap(pred, gt, ids, thr):
    """COCO-style single-class AP at one IoU threshold, all-point interpolation."""
    dets = [(b[4], iid, b) for iid in ids for b in pred.get(iid, [])]
    dets.sort(key=lambda d: -d[0])
    n_gt = sum(len(gt[iid]["boxes"]) for iid in ids)
    used = {iid: [False] * len(gt[iid]["boxes"]) for iid in ids}
    tps = []
    for _, iid, b in dets:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt[iid]["boxes"]):
            v = iou(b, g)
            if v > best:
                best, best_j = v, j
        if best >= thr and not used[iid][best_j]:
            used[iid][best_j] = True
            tps.append(1)
        else:
            tps.append(0)
    if n_gt == 0 or not dets:
        return 0.0
    tp = np.cumsum(tps)
    recall = tp / n_gt
    precision = tp / (np.arange(len(tps)) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def contract(pred, iid, thr):
    boxes = sorted(pred.get(iid, []), key=lambda b: -b[4])
    return [b for b in boxes if b[4] >= thr][:TOPK]


def per_image(pred, gt, ids, thr):
    """Per-image quantities under the proposal contract."""
    rows = {}
    for iid in ids:
        props = contract(pred, iid, thr)
        best = max((iou(p, g) for p in props for g in gt[iid]["boxes"]), default=0.0)
        smax = max((b[4] for b in pred.get(iid, [])), default=0.0)
        rows[iid] = {"m": int(bool(props)), "n": len(props), "best_iou": best, "smax": smax,
                     "y": gt[iid]["fractured"]}
    return rows


def summarise(rows, ids):
    fx = [i for i in ids if rows[i]["y"] == 1]
    nf = [i for i in ids if rows[i]["y"] == 0]
    out = {"m_rate_fractured": float(np.mean([rows[i]["m"] for i in fx])),
           "m_rate_nonfractured": float(np.mean([rows[i]["m"] for i in nf])),
           "mean_props_nonfractured": float(np.mean([rows[i]["n"] for i in nf]))}
    for t in IOUS:
        out[f"recall_top{TOPK}_iou{t}"] = float(np.mean([rows[i]["best_iou"] >= t for i in fx]))
    y = [rows[i]["y"] for i in ids]
    s = [rows[i]["smax"] for i in ids]
    out["image_AP_smax"] = average_precision(s, y)
    out["image_AUROC_smax"] = auroc(s, y)
    return out


def threshold_for_rate(pred, gt, ids, target):
    """Smallest threshold whose non-fractured proposal rate on ``ids`` is <= target."""
    nf = [i for i in ids if gt[i]["fractured"] == 0]
    smax = np.array([max((b[4] for b in pred.get(i, [])), default=0.0) for i in nf])
    cands = np.unique(np.concatenate([smax[smax > 0], [1.0]]))
    for t in cands:  # ascending
        if float(np.mean(smax >= t)) <= target + 1e-12:
            return float(t)
    return 1.0


def main():
    gt_all = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    comps = {}
    for gid, images in enumerate(json.loads(MANIFEST.read_text())["components"]):
        for iid in images:
            comps[iid] = gid
    ids = {r: sorted(i for i, g in gt_all.items() if g["role"] == r) for r in ("val", "cal", "test")}
    specs = []
    for arg in sys.argv[3:]:
        name, rest = arg.split("=", 1)
        path, _, fixed = rest.partition(":")
        raw = json.loads(Path(path).read_text())
        specs.append((name, raw.get("boxes", raw), float(fixed) if fixed else None))

    ref_name, ref_pred, ref_thr = specs[0]
    assert ref_thr is not None, "first detector is the reference and needs a fixed threshold"
    ref_val_rate = summarise(per_image(ref_pred, gt_all, ids["val"], ref_thr), ids["val"])["m_rate_nonfractured"]

    report = {"reference": ref_name, "reference_val_nonfractured_rate": ref_val_rate,
              "n": {r: {"images": len(v), "fractured": sum(gt_all[i]["fractured"] for i in v)} for r, v in ids.items()},
              "detectors": {}}
    test_rows = {}
    for name, pred, fixed in specs:
        matched = fixed if name == ref_name else threshold_for_rate(pred, gt_all, ids["val"], ref_val_rate)
        entry = {"threshold_matched_on_val": matched,
                 "box_AP50_test": box_ap(pred, gt_all, ids["test"], 0.5),
                 "box_AP30_test": box_ap(pred, gt_all, ids["test"], 0.3),
                 "box_AP50_val": box_ap(pred, gt_all, ids["val"], 0.5)}
        rows = per_image(pred, gt_all, ids["test"], matched)
        entry["contract_matched_test"] = summarise(rows, ids["test"])
        entry["contract_matched_val"] = summarise(per_image(pred, gt_all, ids["val"], matched), ids["val"])
        if fixed is not None and name != ref_name:
            entry["threshold_fixed"] = fixed
            entry["contract_fixed_test"] = summarise(per_image(pred, gt_all, ids["test"], fixed), ids["test"])
        report["detectors"][name] = entry
        test_rows[name] = rows

    # paired component bootstrap of (detector - reference) on test, matched thresholds
    rng = np.random.default_rng(20261114)
    groups = {}
    for iid in ids["test"]:
        groups.setdefault(comps.get(iid, iid), []).append(iid)
    keys = list(groups)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(2000)]
    metrics = ["recall_top3_iou0.3", "recall_top3_iou0.5", "m_rate_fractured", "m_rate_nonfractured",
               "image_AP_smax", "image_AUROC_smax"]
    report["bootstrap_vs_reference"] = {}
    for name, _, _ in specs[1:]:
        diffs = {m: [] for m in metrics}
        for d in draws:
            sample = [i for k in d for i in groups[keys[k]]]
            if not any(gt_all[i]["fractured"] for i in sample):
                continue
            a = summarise(test_rows[name], sample)
            b = summarise(test_rows[ref_name], sample)
            for m in metrics:
                diffs[m].append(a[m] - b[m])
        report["bootstrap_vs_reference"][name] = {
            m: {"diff": report["detectors"][name]["contract_matched_test"][m]
                        - report["detectors"][ref_name]["contract_matched_test"][m],
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for m, v in diffs.items()}
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
