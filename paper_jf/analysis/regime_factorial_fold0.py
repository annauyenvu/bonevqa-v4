"""Localization-regime factorial on FracAtlas fold 0 (post hoc, stored predictions only, no retraining).

Factors: detector {Faster R-CNN, YOLOv8 FracAtlas-only (Y1), YOLOv8 joint (Y2)} x training regime
{no retraining (inference-time swap), retrained on in-sample proposals, retrained on cross-fitted proposals},
plus the proposal level for the Y2 cross-fitted model. Contrasts use a paired bootstrap over FracAtlas
components (2,000 resamples), the same unit as the main study.

Run: yolo_pilot/venv/bin/python -I analysis/regime_factorial_fold0.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
F0 = ROOT / "v4_grounded_next/runs/yolo_ablation/fold_0"
MANIFEST = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"
OUT = Path(__file__).resolve().parents[1] / "results/regime_factorial_fold0.json"

ARMS = {
    "A": "A_v4_sealed_frcnn",          # Faster R-CNN, trained on in-sample proposals (sealed V4)
    "B": "B_v4cf_frcnn",               # Faster R-CNN, cross-fitted
    "E": "E_v4_sealed_yolo",           # sealed V4 + Y1 proposals at inference only
    "E2": "E2_v4_sealed_yolojoint",    # sealed V4 + Y2 proposals at inference only
    "C": "C_v4_yolo_insample",         # retrained, Y1 in-sample
    "C2": "C2_v4_yolojoint_insample",  # retrained, Y2 in-sample
    "D": "D_v4cf_yolo",                # retrained, Y1 cross-fitted
    "D2": "D2_v4cf_yolojoint",         # retrained, Y2 cross-fitted, level 0.284
    "D2_015": "D2_sweep_r0.15",
    "D2_002": "D2_sweep_r0.02",
    "D2_max": "D2_sweep_rmax",
    "D2_plc": "D2_v4cf_yolojoint_placebo", "D2_orc": "D2_v4cf_yolojoint_oracleloc",
    "D2_002_plc": "D2_sweep_r0.02_placebo", "D2_002_orc": "D2_sweep_r0.02_oracleloc",
    "JF": "JF_v4cf_yolojoint_btxrd_r0.02", "J": "J_v4cf_yolojoint_btxrd",
}


def ap(y, p):
    o = np.argsort(-p, kind="mergesort"); y = y[o]; p = p[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(p))[0], len(p) - 1]  # thresholds at distinct scores
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auroc(y, p):
    r = np.argsort(np.argsort(p, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    s = np.sort(p); _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    first = np.cumsum(np.r_[0, cnt[:-1]]); r = first[inv] + (cnt[inv] - 1) / 2.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


rows = {k: {r["image_id"]: r for r in json.load(open(F0 / f"{v}_test.json"))["rows"]} for k, v in ARMS.items()}
ids = sorted(rows["A"])
assert all(set(r) == set(ids) for r in rows.values())
comp = {}
for gid, imgs in enumerate(json.load(open(MANIFEST))["components"]):
    for i in imgs:
        comp[i] = gid
y = np.array([rows["A"][i]["y"] for i in ids])
P = {k: np.array([rows[k][i]["p_final"] for i in ids]) for k in ARMS}
M = {k: np.array([bool(rows[k][i]["m"]) for i in ids]) for k in ARMS}
cid = np.array([comp[i] for i in ids]); ucomp = np.unique(cid)
members = [np.where(cid == c)[0] for c in ucomp]

stored = {k: json.load(open(F0 / f"{v}_test.json"))["metrics"]["auprc"] for k, v in ARMS.items()}
point = {k: ap(y, P[k]) for k in ARMS}
for k in ARMS:
    assert abs(point[k] - stored[k]) < 1e-9, (k, point[k], stored[k])

rng = np.random.default_rng(20261122)
B = 2000
boot_idx = [np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) for _ in range(B)]
boot = {k: np.array([ap(y[ix], P[k][ix]) for ix in boot_idx]) for k in ARMS}


def contrast(weights):
    est = sum(w * point[k] for k, w in weights.items())
    dist = sum(w * boot[k] for k, w in weights.items())
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = 2 * min((dist <= 0).mean(), (dist >= 0).mean()); p = max(p, 1 / B)
    return {"estimate": est, "ci95": [float(lo), float(hi)], "p_boot": float(p)}


C = {
    # inference-time localization quality: swap detector without retraining
    "swapY1_minus_A": {"E": 1, "A": -1},
    "swapY2_minus_A": {"E2": 1, "A": -1},
    # adaptation: retrain on the new regime (cross-fitted) vs inference-only swap
    "D_minus_E": {"D": 1, "E": -1},
    "D2_minus_E2": {"D2": 1, "E2": -1},
    # detector effect after adaptation (cross-fitted vs cross-fitted)
    "D_minus_B": {"D": 1, "B": -1},
    "D2_minus_B": {"D2": 1, "B": -1},
    # interaction: detector effect after adaptation minus detector effect without adaptation
    "interaction_Y2": {"D2": 1, "B": -1, "E2": -1, "A": 1},
    "interaction_Y1": {"D": 1, "B": -1, "E": -1, "A": 1},
    # in-sample vs cross-fitted training proposals
    "C2_minus_D2": {"C2": 1, "D2": -1},
    "C_minus_D": {"C": 1, "D": -1},
    # operating level (Y2, cross-fitted, retrained at each level)
    "level002_minus_level0284": {"D2_002": 1, "D2": -1},
    "level015_minus_level0284": {"D2_015": 1, "D2": -1},
    "levelmax_minus_level0284": {"D2_max": 1, "D2": -1},
    # additive path A -> E2 -> D2 -> D2_002 (sums to D2_002 - A)
    "path_total": {"D2_002": 1, "A": -1},
    # location interventions at two levels
    "placebo_0284": {"D2_plc": 1, "D2": -1}, "oracle_0284": {"D2_orc": 1, "D2": -1},
    "placebo_002": {"D2_002_plc": 1, "D2_002": -1}, "oracle_002": {"D2_002_orc": 1, "D2_002": -1},
    "JF_minus_J": {"JF": 1, "J": -1}, "JF_minus_D2_002": {"JF": 1, "D2_002": -1},
}
res = {"n_images": int(len(ids)), "n_fractured": int(y.sum()), "n_components": int(len(ucomp)), "resamples": B,
       "seed": 20261122,
       "arms": {k: {"file": v, "auprc": point[k], "auroc": auroc(y, P[k]),
                    "sens_m0": float(((P[k] >= 0.5) & (y == 1) & ~M[k]).sum() / max(((y == 1) & ~M[k]).sum(), 1)),
                    "n_pos_m0": int(((y == 1) & ~M[k]).sum()),
                    "m_rate_pos": float(M[k][y == 1].mean()), "m_rate_neg": float(M[k][y == 0].mean())}
                for k, v in ARMS.items()},
       "contrasts": {k: contrast(w) for k, w in C.items()}}
OUT.write_text(json.dumps(res, indent=1))
for k in ("A", "B", "E", "E2", "C", "C2", "D", "D2", "D2_015", "D2_002", "D2_max", "J", "JF"):
    a = res["arms"][k]; print(f"{k:7s} AUPRC {a['auprc']:.3f} AUROC {a['auroc']:.3f} sens|m0 {a['sens_m0']:.3f} (n={a['n_pos_m0']})")
for k, v in res["contrasts"].items():
    print(f"{k:28s} {v['estimate']:+.3f} [{v['ci95'][0]:+.3f}; {v['ci95'][1]:+.3f}] p={v['p_boot']:.4f}")
