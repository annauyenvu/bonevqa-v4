"""Choose a post-training fracture scorer for V4 WITHOUT reading test.  Two-way cross-fit on calibration / model_val inside each fold,
scores pooled over the 5 folds.  Simplest candidate within 0.005 AP of the best is locked in selection.json before test is opened."""
import json
from pathlib import Path
import numpy as np

V4 = Path("reports/local_noaux_refinement_v1"); OUT = Path("reports/v4_poststack_v1"); CAN = "Is there a fracture in this X-ray?"
ARMS = dict(V4="local_real_noaux", V4h="local_real_noaux/head_refit", B="B", Bh="B/head_refit")
# ordered simplest first; the rule walks this order
CAND = [("V4", ["V4"]), ("V4h", ["V4h"]), ("V4+V4h", ["V4", "V4h"]), ("V4+V4h+flag", ["V4", "V4h", "flag"]),
        ("V4+V4h+score", ["V4", "V4h", "score"]), ("V4+V4h+flag+score", ["V4", "V4h", "flag", "score"]),
        ("V4+V4h+B+Bh", ["V4", "V4h", "B", "Bh"]), ("V4+V4h+B+Bh+flag+score", ["V4", "V4h", "B", "Bh", "flag", "score"])]


def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]; tp = np.cumsum(y)
    return float((tp[y == 1] / (np.nonzero(y == 1)[0] + 1)).mean())


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit(X, y, ridge=1.0):
    w = np.zeros(X.shape[1]); R = ridge * np.eye(len(w)); R[0, 0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w)); H = (X * (p * (1 - p) + 1e-9)[:, None]).T @ X + R
        w -= np.linalg.solve(H, X.T @ (p - y) + R @ w)
    return w


def load(fold, role):
    f = V4 / f"fold_{fold}"; sc = json.load(open(OUT / f"scores_fold{fold}.json")); assert sc["mismatch"] == 0
    rows = {a: [r for r in json.load(open(f / d / f"{role}_predictions.json"))["records"] if r["question"] == CAN] for a, d in ARMS.items()}
    ids = [r["image_id"] for r in rows["V4"]]; assert all([r["image_id"] for r in v] == ids for v in rows.values())
    F = {a: logit([r["p_yes"] for r in v]) for a, v in rows.items()}
    s = [sc["images"][i]["scores"] for i in ids]
    F["flag"] = np.array([bool(x) for x in s], float)
    F["score"] = np.array([logit(max(x)) if x else logit(.05) - 1 for x in s])
    return ids, np.array([r["gt"] == "yes" for r in rows["V4"]], float), F


def design(F, names, mu_sd):
    return np.column_stack([np.ones(len(F[names[0]]))] + [(F[n] - mu_sd[n][0]) / mu_sd[n][1] for n in names])


def fit_apply(Ff, yf, Fa, names):
    ms = {n: (float(Ff[n].mean()), float(Ff[n].std()) or 1.0) for n in names}
    return design(Fa, names, ms) @ fit(design(Ff, names, ms), yf)


if __name__ == "__main__":
    res = {}; Y = []; S = {n: [] for n, _ in CAND}; per = {n: [] for n, _ in CAND}
    for k in range(5):
        _, yc, Fc = load(k, "calibration"); _, ym, Fm = load(k, "model_val")
        y = np.r_[yc, ym]; Y.append(y)
        for n, names in CAND:
            s = np.r_[fit_apply(Fm, ym, Fc, names), fit_apply(Fc, yc, Fm, names)]
            S[n].append(s); per[n].append(ap(y, s))
    Y = np.concatenate(Y); S = {n: np.concatenate(v) for n, v in S.items()}
    rng = np.random.default_rng(20261101); draws = [rng.integers(0, len(Y), len(Y)) for _ in range(1000)]
    for n, _ in CAND:
        d = [ap(Y[i], S[n][i]) - ap(Y[i], S["V4"][i]) for i in draws]
        res[n] = dict(AP_pooled=ap(Y, S[n]), AP_fold_mean=float(np.mean(per[n])), per_fold=per[n], minus_V4_CI=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        print(n, round(res[n]["AP_pooled"], 4), round(res[n]["AP_fold_mean"], 4), np.round(per[n], 3), np.round(res[n]["minus_V4_CI"], 3), flush=True)
    best = max(v["AP_pooled"] for v in res.values())
    pick = next(n for n, _ in CAND if res[n]["AP_pooled"] >= best - .005)
    print("n", len(Y), "pos", int(Y.sum()), "locked:", pick)
    json.dump(dict(rule="simplest candidate within 0.005 pooled cross-fit AP of the best; test unread", locked=pick, n=len(Y), pos=int(Y.sum()), results=res), open(OUT / "selection.json", "w"), indent=1)
