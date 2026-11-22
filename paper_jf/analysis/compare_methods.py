"""Same-protocol comparison of every fracture-presence scorer available on the five FracAtlas folds.

All arms score the canonical question "Is there a fracture in this X-ray?" (VQA arms) or the image
(image-only classifiers) for the same 4,013 test images (each image is test in exactly one fold).
Reported: pooled out-of-fold AP (the primary endpoint of the V4 study), mean of per-fold AP (the
primary endpoint of the final-system ladder), AUROC, and sensitivity / specificity at 0.5.
Contrasts JF - X use a paired bootstrap that resamples FracAtlas components within each fold
(2,000 resamples), so both pooled and per-fold summaries get intervals from the same resamples.

Run: yolo_pilot/venv/bin/python -I analysis/compare_methods.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
R = ROOT / "refactor_v2/reports"
S = R / "local_noaux_refinement_v1"            # sealed Base and V4, seed 42
M = R / "v5_studies/access_v1_methods"         # FAVP-, LaPA-, Lens-style integrations, seed 42
A43 = R / "v5_studies/access_v1"               # seed-43 replication and mechanism controls
IB = R / "access_v1/image_baselines"
Y = ROOT / "v4_grounded_next/runs/yolo_ablation"
OUT = Path(__file__).resolve().parents[1] / "results/compare_methods.json"
CAN = "Is there a fracture in this X-ray?"
FOLDS = "01234"
comp = {im: k for k, c in enumerate(json.load(open(ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"))["components"]) for im in c}


def read(p):
    return json.load(open(p))


def can(path):
    return {r["image_id"]: float(r["p_yes"]) for r in read(path)["records"] if r["question"] == CAN}


def ladder(f, name):
    return {r["image_id"]: float(r["p_final"]) for r in read(Y / f"fold_{f}/{name}_test.json")["rows"]}


def ap(y, p):
    o = np.argsort(-p, kind="mergesort"); y = y[o]; p = p[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); last = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tp, fp = tp[last], fp[last]
    return float(np.sum(np.diff(np.r_[0.0, tp / y.sum()]) * tp / (tp + fp)))


def auroc(y, p):
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    r = np.cumsum(np.r_[0, cnt[:-1]])[inv] + (cnt[inv] - 1) / 2.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


SOURCES = {
    "BiomedCLIP-LR": lambda f: read(IB / "biomedclip_lr.json")[f],
    "MobileNetV3": lambda f: read(IB / f"mobilenet_fold{f}.json")["p_yes"],
    "Base": lambda f: can(S / f"fold_{f}/B/test_predictions.json"),
    "FAVP-style": lambda f: can(M / f"fold_{f}/V_seed42/test_predictions.json"),
    "LaPA-style": lambda f: can(M / f"fold_{f}/P_seed42/test_predictions.json"),
    "Lens-style": lambda f: can(M / f"fold_{f}/L_seed42/test_predictions.json"),
    "V4": lambda f: can(S / f"fold_{f}/local_real_noaux/test_predictions.json"),
    "Base_s43": lambda f: can(A43 / f"fold_{f}/B_seed43/test_predictions.json"),
    "V4_s43": lambda f: can(A43 / f"fold_{f}/V4_seed43/test_predictions.json"),
    "V4_flag_s43": lambda f: can(A43 / f"fold_{f}/V4_flag_seed43/test_predictions.json"),
    "V4_grid_s43": lambda f: can(A43 / f"fold_{f}/V4_grid_seed43/test_predictions.json"),
    "V4_placebo_s43": lambda f: can(A43 / f"fold_{f}/V4_placebo_seed43/test_predictions.json"),
    "L_A": lambda f: ladder(f, "A_v4_sealed_frcnn"),
    "L_B": lambda f: ladder(f, "B_v4cf_frcnn"),
    "L_D2": lambda f: ladder(f, "D2_v4cf_yolojoint"),
    "L_J": lambda f: ladder(f, "J_v4cf_yolojoint_btxrd"),
    "JF": lambda f: ladder(f, "JF_v4cf_yolojoint_btxrd_r0.02"),
}

ids, fold_of, yv = [], [], []
P = {k: [] for k in SOURCES}
for f in FOLDS:
    got = {k: g(f) for k, g in SOURCES.items()}
    lab = {r["image_id"]: int(r["y"]) for r in read(Y / f"fold_{f}/A_v4_sealed_frcnn_test.json")["rows"]}
    keys = sorted(lab)
    for k, d in got.items():
        assert set(d) == set(keys), (f, k, len(d), len(keys))
    for im in keys:
        ids.append(im); fold_of.append(int(f)); yv.append(lab[im])
        for k in SOURCES:
            P[k].append(got[k][im])
y = np.array(yv); fold_of = np.array(fold_of); P = {k: np.array(v) for k, v in P.items()}
# the ladder's A is the sealed V4 checkpoint re-scored in the final-system pipeline
agree = float(np.max(np.abs(P["L_A"] - P["V4"])))

groups = [[np.where((fold_of == f) & (np.array([comp[i] for i in ids]) == c))[0]
           for c in np.unique([comp[i] for i, ff in zip(ids, fold_of) if ff == f])] for f in range(5)]


def summarize(p, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    yy, pp, ff = y[idx], p[idx], fold_of[idx]
    per = [ap(yy[ff == f], pp[ff == f]) for f in range(5)]
    return ap(yy, pp), float(np.mean(per)), per


rng = np.random.default_rng(20261122); NB = 2000
boots = [np.concatenate([g[j] for g in groups for j in rng.integers(0, len(g), len(g))]) for _ in range(NB)]
BP = {k: np.array([summarize(P[k], ix)[:2] for ix in boots]) for k in SOURCES}

res = {"n": int(len(y)), "fractured": int(y.sum()), "max_abs_diff_ladderA_vs_sealedV4": agree, "arms": {}, "JF_minus": {}}
for k in SOURCES:
    pooled, mean_fold, per = summarize(P[k]); z = P[k] >= 0.5
    res["arms"][k] = {"pooled_ap": pooled, "mean_fold_ap": mean_fold, "fold_ap": per, "pooled_auroc": auroc(y, P[k]),
                      "sens": float(z[y == 1].mean()), "spec": float((~z[y == 0]).mean())}
for k in SOURCES:
    if k == "JF":
        continue
    out = {}
    for j, name in ((0, "pooled_ap"), (1, "mean_fold_ap")):
        d = BP["JF"][:, j] - BP[k][:, j]
        est = res["arms"]["JF"][name] - res["arms"][k][name]
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / NB)
        out[name] = {"diff": est, "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])], "p_boot": float(p)}
    res["JF_minus"][k] = out
OUT.write_text(json.dumps(res, indent=1))
print("max |ladder A - sealed V4| =", agree)
for k, a in res["arms"].items():
    d = res["JF_minus"].get(k)
    s = "" if d is None else f"  JF-X pooled {d['pooled_ap']['diff']:+.3f} [{d['pooled_ap']['ci95'][0]:+.3f};{d['pooled_ap']['ci95'][1]:+.3f}]  mean {d['mean_fold_ap']['diff']:+.3f} [{d['mean_fold_ap']['ci95'][0]:+.3f};{d['mean_fold_ap']['ci95'][1]:+.3f}]"
    print(f"{k:16s} pooled {a['pooled_ap']:.3f} meanfold {a['mean_fold_ap']:.3f} AUROC {a['pooled_auroc']:.3f} sens {a['sens']:.3f} spec {a['spec']:.3f}{s}")
