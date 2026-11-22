"""Two inference-free uncertainty checks on the stored scores of a joint run (no model, no training).

    source v4_grounded_next/env.sh; cd v4_grounded_next
    $PY -m scripts.uncertainty_probe J_v4cf_yolojoint_btxrd

A  Grounding-aware confidence for the yes / no questions: P(correct) from the head confidence alone (Platt) versus the head
   confidence plus the detector's top box score (the complementary-signal idea of hallucination-aware calibration).
   Reports AUROC of the confidence for correctness, ECE, error in the most confident half, and the "safe yield": the share
   of cases answered when the threshold is set on val for an error budget.
B  Split-conformal prediction sets (LAC) for the 9-class diagnosis: a shortlist with a target coverage instead of one answer.
Everything is fitted on calibration / model_val and read on test / external.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts import reference_analyze as A
from vg.paths import RUNS, read_json

NAME = sys.argv[1]
J, ABL, CF = RUNS / "joint" / "fold_0", RUNS / "yolo_ablation" / "fold_0", RUNS / "crossfit_yolo_joint"
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ece(conf, ok, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return e


def yield_at(conf_v, ok_v, conf_t, ok_t, budget):
    """Lowest val threshold whose answered part has error <= budget; coverage and error of that threshold on test."""
    best = None
    for tau in np.unique(conf_v):
        keep = conf_v >= tau
        if keep.sum() >= 20 and 1 - ok_v[keep].mean() <= budget:
            best = tau
            break
    if best is None:
        return 0.0, float("nan")
    keep = conf_t >= best
    return float(keep.mean()), float(1 - ok_t[keep].mean()) if keep.any() else float("nan")


def part_a(tag, pv, yv, sv, pt, yt, st):
    predv, predt = (pv >= 0.5).astype(int), (pt >= 0.5).astype(int)
    okv, okt = (predv == yv), (predt == yt)
    f1 = lambda p: np.abs(logit(p))[:, None]
    f2 = lambda p, s, pred: np.c_[np.abs(logit(p)), pred, s, pred * s]
    out = {}
    for lab, Xv, Xt in (("head confidence, Platt", f1(pv), f1(pt)), ("+ detector score", f2(pv, sv, predv), f2(pt, st, predt))):
        m = LogisticRegression(C=10.0, max_iter=1000).fit(Xv, okv)
        cv, ct = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
        half = ct >= np.median(ct)
        y5, y10 = yield_at(cv, okv, ct, okt, 0.05), yield_at(cv, okv, ct, okt, 0.10)
        out[lab] = ct
        print(f"  {tag:26s} {lab:24s} acc {okt.mean():.3f} | AUROC(correct) {roc_auc_score(okt, ct):.3f} | ECE {ece(ct, okt):.3f} | "
              f"error in confident half {1 - okt[half].mean():.3f} | yield@5% {y5[0]:.2f} (err {y5[1]:.3f}) | yield@10% {y10[0]:.2f} (err {y10[1]:.3f})")
    raw = np.maximum(pt, 1 - pt)
    print(f"  {tag:26s} {'raw head confidence':24s} AUROC(correct) {roc_auc_score(okt, raw):.3f} | ECE {ece(raw, okt):.3f}")
    rng = np.random.default_rng(20261114)
    d = [roc_auc_score(okt[ix], out["+ detector score"][ix]) - roc_auc_score(okt[ix], out["head confidence, Platt"][ix])
         for ix in (rng.integers(0, len(okt), len(okt)) for _ in range(2000)) if 0 < okt[ix].sum() < len(ix)]
    print(f"  {tag:26s} AUROC gain from the detector score {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] (image bootstrap)")


top = lambda props, i: max((b[4] for b in props.get(i, [])), default=0.0)
print("A. grounding-aware confidence")
fa = read_json(CF / "fold_0" / "proposals_fold0.json")
c, t = read_json(ABL / f"{NAME}_calibration.json")["rows"], read_json(ABL / f"{NAME}_test.json")["rows"]
g = lambda rows, k: np.array([r[k] for r in rows])
part_a("FracAtlas fracture, test", g(c, "p_final"), g(c, "y"), np.array([top(fa["calibration"], r["image_id"]) for r in c]),
       g(t, "p_final"), g(t, "y"), np.array([top(fa["test"], r["image_id"]) for r in t]))
bt = read_json(CF / "btxrd" / "proposals_btxrd_insample.json")
R = {role: read_json(J / f"ref_{NAME}_{role}.json") for role in ("model_val", "test", "external")}
yy = lambda rows: np.array([int(r["truth"] == "yes") for r in rows])
for ep in ("tumor", "malignancy"):
    v = R["model_val"]["endpoints"][ep]
    for role in ("test", "external"):
        x = R[role]["endpoints"][ep]
        part_a(f"BTXRD {ep}, {role}", g(v, "plain"), yy(v), np.array([top(bt["model_val"], r["image_id"]) for r in v]),
               g(x, "plain"), yy(x), np.array([top(bt[role], r["image_id"]) for r in x]))

print("B. conformal shortlists for the diagnosis (LAC, calibrated on model_val)")
vqa = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot/datasets/btxrd_vqa")
train = read_json(vqa / "vqa_train.json")
per_image = {r["image_id"]: r for r in train if r.get("canonical") == "tumor"}
dx = [(per_image[r["image_id"]], r["answer"]) for r in train if r.get("canonical") == "diagnosis"]
nb = A.NaiveBayes([r for r, _ in dx], [y for _, y in dx], ("site",))
cands = R["test"]["candidates"]


def probs(role, site):
    rows = R[role]["endpoints"]["diagnosis"]
    y = np.array([cands.index(r["truth"]) for r in rows])
    z = np.array([r["plain"] for r in rows])
    if site:
        z = z + 0.5 * np.array([nb.loglik(r) for r in rows])
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True), y


for site in (False, True):
    pv, yv = probs("model_val", site)
    s = np.sort(1 - pv[np.arange(len(yv)), yv])
    for role in ("test", "external"):
        p, y = probs(role, site)
        line = f"  {'image + site' if site else 'image only':12s} {role:8s} n={len(y)} top-1 {np.mean(p.argmax(1) == y):.3f} top-2 {np.mean([y[i] in np.argsort(-p[i])[:2] for i in range(len(y))]):.3f} top-3 {np.mean([y[i] in np.argsort(-p[i])[:3] for i in range(len(y))]):.3f}"
        for alpha in (0.2, 0.1):
            q = s[min(len(s) - 1, int(np.ceil((len(s) + 1) * (1 - alpha))) - 1)]
            sets = (1 - p) <= q
            cov = sets[np.arange(len(y)), y].mean()
            size = sets.sum(1)
            rare = np.isin(y, [cands.index(c) for c in ("giant cell tumor", "osteofibroma", "other benign bone tumor", "other malignant bone tumor")])
            line += f" | target {1 - alpha:.0%}: coverage {cov:.3f}, mean size {size.mean():.2f}, single answer {np.mean(size == 1):.2f}, empty {np.mean(size == 0):.2f}, rare-class coverage {sets[np.arange(len(y)), y][rare].mean():.2f}"
        print(line)
