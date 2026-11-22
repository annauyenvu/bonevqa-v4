"""Selective prediction (abstention) on the stored scores of a joint run.  No model, no training.

    source v4_grounded_next/env.sh; cd v4_grounded_next
    $PY -m scripts.selective_report J_v4cf_yolojoint_btxrd

Calibration (Platt / temperature) and abstention thresholds are fitted on calibration / model_val only, then applied to
test and external.  Needs runs/yolo_ablation/fold_0/<name>_{calibration,test}.json and runs/joint/fold_0/ref_<name>_<role>.json.
"""
import json, sys, numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.optimize import minimize
from scripts import reference_analyze as A
from vg.paths import RUNS, read_json
NAME = sys.argv[1]
J = RUNS / "joint" / "fold_0"; ABL = RUNS / "yolo_ablation" / "fold_0"
rng = np.random.default_rng(20261114)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

def platt(p, y):
    z = logit(p)
    f = lambda w: np.mean(np.log1p(np.exp(-np.where(y == 1, 1, -1) * (w[0] * z + w[1]))))
    w = minimize(f, [1.0, 0.0], method="Nelder-Mead").x
    return lambda q: sig(w[0] * logit(q) + w[1])

def ece(conf, ok, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return e

def ci(groups, fn):
    mem = defaultdict(list)
    for i, g in enumerate(groups): mem[g].append(i)
    gl = list(mem.values()); out = []
    for _ in range(2000):
        ix = np.array([i for k in rng.integers(0, len(gl), len(gl)) for i in gl[k]])
        v = fn(ix)
        if v is not None: out.append(v)
    return np.percentile(out, [2.5, 97.5])

def report(tag, conf_v, conf_t, ok_t, groups, y_t=None, pred_t=None, extra=""):
    full = ok_t.mean()
    line = f"{tag:38s} n={len(ok_t):4d} full acc {full:.3f}"
    for cov in (0.9, 0.8):
        tau = np.quantile(conf_v, 1 - cov)
        keep = conf_t >= tau
        gain = lambda ix: (ok_t[ix][keep[ix]].mean() - ok_t[ix].mean()) if keep[ix].any() else None
        lo, hi = ci(groups, gain)
        caught = (~ok_t & ~keep).sum() / max((~ok_t).sum(), 1)
        line += f" | val-cov {cov:.0%}: test cov {keep.mean():.2f} acc {ok_t[keep].mean():.3f} (+{ok_t[keep].mean()-full:.3f} [{lo:+.3f},{hi:+.3f}]) errors flagged {caught:.0%}"
        if y_t is not None and cov == 0.8:
            pos = y_t == 1
            fn_all = (pos & (pred_t == 0)).sum(); fn_kept = (pos & (pred_t == 0) & keep).sum()
            line += f" FN {fn_all}->{fn_kept}"
    print(line + extra)

def binary(tag, pv, yv, pt, yt, groups, mv=None, mt=None):
    cal = platt(pv, yv)
    qv, qt = cal(pv), cal(pt)
    for lab, a, b in (("raw", pv, pt), ("calibrated", qv, qt)):
        pred = (b >= 0.5).astype(int); ok = pred == yt; conf = np.maximum(b, 1 - b)
        report(f"{tag} [{lab}]", np.maximum(a, 1 - a), conf, ok, groups, yt, pred, f" | ECE {ece(conf, ok):.3f}")
    if mt is not None:   # rule: also abstain on "no" answers without any proposal
        pred = (qt >= 0.5).astype(int); ok = pred == yt
        flag = (pred == 0) & (~mt)
        print(f"   rule 'negative answer without a proposal': flagged {flag.mean():.2f} of images; of them wrong {(~ok[flag]).mean():.2f} (n={flag.sum()}); "
              f"missed positives in flagged {(flag & (yt==1)).sum()} of {((pred==0)&(yt==1)).sum()} FN; acc on the rest {ok[~flag].mean():.3f}")

# FracAtlas fracture
comp = {}
for gid, images in enumerate(read_json(A.FA_DATA / "manifest.json")["components"]):
    for im in images: comp[str(im).rsplit(".", 1)[0]] = gid
c = read_json(ABL / f"{NAME}_calibration.json")["rows"]; t = read_json(ABL / f"{NAME}_test.json")["rows"]
g = lambda rows, k: np.array([r[k] for r in rows])
binary("FracAtlas fracture, test", g(c, "p_final"), g(c, "y"), g(t, "p_final"), g(t, "y"), [comp[r["image_id"]] for r in t], g(c, "m").astype(bool), g(t, "m").astype(bool))

R = {role: read_json(J / f"ref_{NAME}_{role}.json") for role in ("model_val", "test", "external")}
for ep in ("tumor", "malignancy"):
    v = R["model_val"]["endpoints"][ep]
    for role in ("test", "external"):
        t = R[role]["endpoints"][ep]
        yy = lambda rows: np.array([int(r["truth"] == "yes") for r in rows])
        binary(f"BTXRD {ep}, {role}", g(v, "plain"), yy(v), g(t, "plain"), yy(t), [(r["center"], r["group"]) for r in t],
               g(v, "has_proposal").astype(bool) if ep == "tumor" else None, g(t, "has_proposal").astype(bool) if ep == "tumor" else None)

# diagnosis: confidence = max softmax of temperature-scaled candidate scores
vqa = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot/datasets/btxrd_vqa")
train = read_json(vqa / "vqa_train.json")
per_image = {r["image_id"]: r for r in train if r.get("canonical") == "tumor"}
dx = [(per_image[r["image_id"]], r["answer"]) for r in train if r.get("canonical") == "diagnosis"]
nb = A.NaiveBayes([r for r, _ in dx], [y for _, y in dx], ("site",))
cands = R["test"]["candidates"]
def soft(z, T):
    z = z / T; z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
def pack(role, site):
    rows = R[role]["endpoints"]["diagnosis"]; y = np.array([cands.index(r["truth"]) for r in rows])
    z = np.array([r["plain"] for r in rows])
    if site: z = z + 0.5 * np.array([nb.loglik(r) for r in rows])
    return rows, y, z
for site in (False, True):
    _, vy, vz = pack("model_val", site)
    T = min(np.linspace(0.3, 5, 48), key=lambda T: -np.mean(np.log(soft(vz, T)[np.arange(len(vy)), vy] + 1e-12)))
    for role in ("test", "external"):
        rows, y, z = pack(role, site)
        p = soft(z, T); ok = p.argmax(1) == y; conf = p.max(1)
        report(f"BTXRD diagnosis{' + site' if site else ''}, {role} (T={T:.1f})", soft(vz, T).max(1), conf, ok, [(r["center"], r["group"]) for r in rows], extra=f" | ECE {ece(conf, ok):.3f}")
