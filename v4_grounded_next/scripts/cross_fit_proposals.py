"""Cross-fitted TRAIN proposals (removes V4's in-sample detector shift).

For outer fold f:
  plan       TRAIN -> 5 inner folds, StratifiedGroupKFold on (fractured, component_id)
  inner K    train detector on TRAIN \\ inner_K with the exact V4 recipe
  oof K      that detector proposes (with scores) on inner_K only
  heldout    sealed V4 outer detector (full TRAIN) re-run WITH scores on
             model_val/calibration/test (+ TRAIN in-sample, audit only);
             boxes must match sealed boxes.json
  merge      train_cf.json (OOF) + eval.json (outer) -> proposals_fold{f}.json
  audit      m(I), IoU and score distributions: TRAIN-CF vs TRAIN-in-sample vs TEST
  all        everything above, resumable (completed steps are sha-checked and skipped)

Usage:  source v4_grounded_next/env.sh
        $PY -m scripts.cross_fit_proposals all --fold 0
        $PY -m scripts.cross_fit_proposals smoke          # 3-minute pipeline check
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vg import geometry
from vg.detector import load_detector, open_canonical, predict_scored, train_detector_v4_recipe
from vg.paths import (FRACATLAS, ROLES, RUNS, V4_REPORT, component_lookup, dump_json, image_table,
                      manifest_fold, read_json, sha256)

K_INNER = 5
SPLIT_SEED = 20261111


def out_dir(fold, smoke=False):
    return RUNS / ("smoke_crossfit" if smoke else "crossfit") / f"fold_{fold}"


def plan(fold, smoke=False):
    from sklearn.model_selection import StratifiedGroupKFold
    d = out_dir(fold, smoke)
    p = d / "inner_split.json"
    if p.exists():
        return read_json(p)
    roles = manifest_fold(fold)
    table = image_table(read_json(roles["train"]))
    lookup = component_lookup()
    ids = sorted(table)
    missing = [i for i in ids if i not in lookup]
    # Images without a known component are their own group (V4 'known_components' < images).
    groups = [lookup.get(i, f"solo:{i}") for i in ids]
    y = [int(table[i]["fractured"]) for i in ids]
    gid = {g: n for n, g in enumerate(sorted(set(map(str, groups))))}
    sgkf = StratifiedGroupKFold(n_splits=K_INNER, shuffle=True, random_state=SPLIT_SEED)
    inner = {}
    for k, (_, held) in enumerate(sgkf.split(ids, y, [gid[str(g)] for g in groups])):
        inner[str(k)] = sorted(ids[j] for j in held)
    allheld = [i for v in inner.values() for i in v]
    assert sorted(allheld) == ids and len(allheld) == len(set(allheld))
    comp_fold = {}
    for k, v in inner.items():
        for i in v:
            g = str(groups[ids.index(i)])
            assert comp_fold.setdefault(g, k) == k, f"component {g} straddles inner folds"
    if smoke:
        inner = {k: v[:6] for k, v in inner.items()}
    spec = {"outer_fold": fold, "train_manifest": str(roles["train"]), "train_sha256": sha256(roles["train"]),
            "k_inner": K_INNER, "seed": SPLIT_SEED, "images": len(ids), "unknown_component_images": len(missing),
            "inner": inner,
            "inner_stats": {k: {"images": len(v), "fractured": sum(table[i]["fractured"] for i in v)} for k, v in inner.items()}}
    dump_json(p, spec, exclusive=True)
    return spec


def inner_detector(fold, k, smoke=False):
    d = out_dir(fold, smoke) / f"inner_{k}"
    receipt = d / "receipt.json"
    if receipt.exists():
        r = read_json(receipt); assert sha256(r["checkpoint"]) == r["sha256"]; return r
    spec = plan(fold, smoke)
    table = image_table(read_json(manifest_fold(fold)["train"]))
    held = set(spec["inner"][str(k)])
    if smoke:
        pos = [i for i in sorted(table) if table[i]["fractured"] and i not in held][:4]
        neg = [i for i in sorted(table) if not table[i]["fractured"] and i not in held][:8]
        train_ids = pos + neg
    else:
        train_ids = [i for i in sorted(table) if i not in held]
    from src.detector_protocol import verify_detector_annotation_parity
    from scripts.train_detector_v2 import BoPhatHienDataset
    # Same check V4 ran: detector annotations == sealed VQA GT for every inner-train image.
    verify_detector_annotation_parity(BoPhatHienDataset("train", canh_dai=640, tang_cuong=False, allowed_ids=set(train_ids)),
                                      str(manifest_fold(fold)["train"]))
    ckpt = d / "LAST.pt"
    info = train_detector_v4_recipe(train_ids, ckpt, epochs=1 if smoke else 2, max_updates=3 if smoke else None,
                                    progress_path=d / "progress.json")
    assert not (set(train_ids) & held), "inner detector saw its held-out images"
    r = {"checkpoint": str(ckpt), "sha256": sha256(ckpt), "train_ids": len(train_ids), "held_out": len(held), **info}
    dump_json(receipt, r, exclusive=True)
    return r


def _propose(model, ids, table, progress=None):
    out, start = {}, time.time()
    for n, iid in enumerate(ids):
        im = open_canonical(table[iid]["image"])
        out[iid], _ = predict_scored(model, im)
        if progress and n % 50 == 0:
            dump_json(progress, {"done": n, "total": len(ids), "wall": time.time() - start})
    return out


def oof(fold, k, smoke=False):
    d = out_dir(fold, smoke) / f"inner_{k}"
    p = d / "oof.json"
    if p.exists():
        return read_json(p)
    r = inner_detector(fold, k, smoke)
    model, payload = load_detector(r["checkpoint"])
    held = plan(fold, smoke)["inner"][str(k)]
    assert not (set(payload["training_image_ids"]) & set(held))
    table = image_table(read_json(manifest_fold(fold)["train"]))
    props = _propose(model, held, table, d / "oof_progress.json")
    dump_json(p, props, exclusive=True)
    return props


def heldout(fold, smoke=False):
    d = out_dir(fold, smoke)
    p = d / "outer_scored.json"
    if p.exists():
        return read_json(p)
    receipt = read_json(V4_REPORT / f"fold_{fold}/detector/receipt.json")
    assert sha256(receipt["checkpoint"]) == receipt["sha256"], "sealed V4 detector changed"
    model, payload = load_detector(receipt["checkpoint"])
    sealed = read_json(V4_REPORT / f"fold_{fold}/boxes.json")
    roles = manifest_fold(fold)
    out, parity = {}, {"checked": 0, "max_abs_diff": 0.0, "count_mismatch": 0}
    for role in ROLES:
        table = image_table(read_json(roles[role]))
        ids = sorted(table)
        if role == "train":
            assert set(payload["training_image_ids"]) == set(ids)
        if smoke:
            ids = ids[:5]
        props = _propose(model, ids, table, d / f"outer_{role}_progress.json")
        for iid, pr in props.items():
            ref = sealed[iid]
            parity["checked"] += 1
            if len(ref) != len(pr):
                parity["count_mismatch"] += 1
                continue
            for a, b in zip(ref, pr):
                parity["max_abs_diff"] = max(parity["max_abs_diff"], max(abs(x - y) for x, y in zip(a, b[:4])))
        out[role] = props
    assert parity["count_mismatch"] == 0 and parity["max_abs_diff"] < 1e-2, f"sealed proposal parity failed {parity}"
    out["_parity_with_sealed_boxes_json"] = parity
    dump_json(p, out, exclusive=True)
    return out


def merge(fold, smoke=False):
    d = out_dir(fold, smoke)
    p = d / f"proposals_fold{fold}.json"
    if p.exists():
        return read_json(p)
    spec = plan(fold, smoke)
    train_cf = {}
    for k in map(str, range(K_INNER)):
        o = oof(fold, k, smoke)
        assert not (set(o) & set(train_cf))
        train_cf.update(o)
    if not smoke:
        assert len(train_cf) == spec["images"]
    outer = heldout(fold, smoke)
    merged = {"schema": "vg-proposals-v1", "outer_fold": fold,
              "contract": "TRAIN: out-of-fold inner detectors (source=detector_cf). model_val/calibration/test: "
                          "sealed V4 full-TRAIN detector (source=detector_full). Boxes canonical-image xyxy + score.",
              "train": train_cf, "train_insample_audit_only": outer["train"],
              **{r: outer[r] for r in ROLES if r != "train"},
              "parity": outer["_parity_with_sealed_boxes_json"]}
    dump_json(p, merged, exclusive=True)
    return merged


def _stats(props, table, orient_cache):
    frac = [i for i in props if table[i]["fractured"]]
    neg = [i for i in props if not table[i]["fractured"]]
    ious, smax = [], []
    for i in frac:
        o, raw = orient_cache.setdefault(i, geometry.orientation_of(FRACATLAS / table[i]["image"]))
        gts = geometry.canonical_boxes(table[i]["boxes"], o, raw)
        ious.append(geometry.best_iou(props[i], gts))
        smax.append(max([p[4] for p in props[i]], default=0.0))
    ious, smax = np.array(ious), np.array(smax)
    m_frac = np.array([bool(props[i]) for i in frac])
    return {"fractured_images": len(frac), "negative_images": len(neg),
            "m_rate_fractured": float(m_frac.mean()) if len(frac) else None,
            "m_rate_negative": float(np.mean([bool(props[i]) for i in neg])) if neg else None,
            "recall_iou25": float((ious >= .25).mean()) if len(frac) else None,
            "recall_iou50": float((ious >= .5).mean()) if len(frac) else None,
            "best_iou_quantiles": np.quantile(ious, [.1, .25, .5, .75, .9]).round(4).tolist() if len(frac) else None,
            "smax_fractured_quantiles": np.quantile(smax, [.1, .25, .5, .75, .9]).round(4).tolist() if len(frac) else None,
            "_ious": ious.tolist(), "_m": m_frac.astype(int).tolist()}


def _two_prop(a, b):
    if not len(a) or not len(b):
        return None
    p1, n1, p2, n2 = np.mean(a), len(a), np.mean(b), len(b)
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    return {"diff": float(p1 - p2), "z": float(z), "p_two_sided": float(math.erfc(abs(z) / math.sqrt(2)))}


def audit(fold, smoke=False):
    from scipy.stats import ks_2samp
    d = out_dir(fold, smoke)
    merged = merge(fold, smoke)
    roles = manifest_fold(fold)
    tables = {r: image_table(read_json(roles[r])) for r in ROLES}
    orient = {}
    s = {"train_cf": _stats(merged["train"], tables["train"], orient),
         "train_insample": _stats(merged["train_insample_audit_only"], tables["train"], orient),
         "test": _stats(merged["test"], tables["test"], orient)}
    comp = {}
    for a in ("train_cf", "train_insample"):
        comp[f"{a}_vs_test"] = {"m_rate": _two_prop(s[a]["_m"], s["test"]["_m"]),
                                "best_iou_ks": (dict(zip(("statistic", "p"), map(float, ks_2samp(s[a]["_ious"], s["test"]["_ious"]))))
                                                if s[a]["_ious"] and s["test"]["_ious"] else None)}
    for v in s.values():
        v.pop("_ious"); v.pop("_m")
    rep = {"fold": fold, "stats": s, "comparisons": comp,
           "criterion": "train_cf m_rate and IoU distribution should be statistically indistinguishable from test "
                        "(|diff| small, p>0.05), unlike train_insample (V4: 82.0% vs 75.9%)."}
    dump_json(d / "audit.json", rep)
    return rep


def run_all(fold, smoke=False):
    plan(fold, smoke)
    for k in map(str, range(K_INNER)):
        oof(fold, k, smoke)
    heldout(fold, smoke)
    merge(fold, smoke)
    rep = audit(fold, smoke)
    print({k: {kk: vv for kk, vv in v.items() if kk.startswith("m_rate") or kk.startswith("recall")} for k, v in rep["stats"].items()})
    print(rep["comparisons"])


if __name__ == "__main__":
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("PYTHONPYCACHEPREFIX"), "source v4_grounded_next/env.sh"
    import torch
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "inner", "oof", "heldout", "merge", "audit", "all", "smoke"])
    ap.add_argument("--fold", default="0")
    ap.add_argument("--inner", default=None)
    a = ap.parse_args()
    if a.step == "smoke":
        run_all(a.fold, smoke=True)
    elif a.step in ("inner", "oof"):
        ks = [a.inner] if a.inner is not None else list(map(str, range(K_INNER)))
        for k in ks:
            (inner_detector if a.step == "inner" else oof)(a.fold, k)
    elif a.step == "all":
        run_all(a.fold)
    else:
        globals()[a.step](a.fold)
