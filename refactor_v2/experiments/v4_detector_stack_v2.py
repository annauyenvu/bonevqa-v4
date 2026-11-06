"""Choose (phase select: no test) or evaluate (phase test) a fracture scorer from the extended detectors.  Same cross-fit and bootstrap as v4_poststack."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.v4_poststack_select_v1 import V4, CAN, ap, logit, fit_apply
O = Path("reports/v4_detector_ext_v1"); ARMS = dict(V4="local_real_noaux", V4h="local_real_noaux/head_refit", B="B", Bh="B/head_refit")
EPS = [int(x) for x in sys.argv[2].split(",")] if __name__ == "__main__" and len(sys.argv) > 2 else [2, 3, 4, 5, 6]
# simplest first: no crops and one detector before V4 crops, which also need the old detector for their boxes
SETS = [("B+det", ["B", "Bh", "flag", "s1"]), ("B+det+", ["B", "Bh", "flag", "s1", "s2", "n"]), ("V4+det", ["V4", "V4h", "flag", "s1"]), ("V4+det+", ["V4", "V4h", "flag", "s1", "s2", "n"])]


def load(fold, role, ep):
    f = V4 / f"fold_{fold}"; sc = json.load(open(O / f"fold_{fold}/scores_ep{ep}.json"))
    rows = {a: [r for r in json.load(open(f / d / f"{role}_predictions.json"))["records"] if r["question"] == CAN] for a, d in ARMS.items()}
    ids = [r["image_id"] for r in rows["V4"]]; F = {a: logit([r["p_yes"] for r in v]) for a, v in rows.items()}
    S = [sc[i]["scores"] for i in ids]; low = logit(.05) - 1
    F["flag"] = np.array([bool(x) for x in S], float); F["n"] = np.array([len(x) for x in S], float)
    F["s1"] = np.array([logit(x[0]) if x else low for x in S]); F["s2"] = np.array([logit(x[1]) if len(x) > 1 else low for x in S])
    return ids, np.array([r["gt"] == "yes" for r in rows["V4"]], float), F


def thr90(s, y):
    n = np.sort(s[y == 0]); return float(np.nextafter(n[int(np.ceil(.9 * len(n))) - 1], np.inf))


if __name__ != "__main__":
    pass
elif sys.argv[1] == "select":
    res = {}
    for ep in EPS:
        for name, cols in SETS:
            Y, S, per = [], [], []
            for k in range(5):
                _, yc, Fc = load(k, "calibration", ep); _, ym, Fm = load(k, "model_val", ep)
                y = np.r_[yc, ym]; s = np.r_[fit_apply(Fm, ym, Fc, cols), fit_apply(Fc, yc, Fm, cols)]; Y.append(y); S.append(s); per.append(ap(y, s))
            a = ap(np.concatenate(Y), np.concatenate(S)); res[f"{name}|ep{ep}"] = dict(AP=a, per_fold=per, name=name, ep=ep)
            print(name, ep, round(a, 4), np.round(per, 3), flush=True)
    best = max(v["AP"] for v in res.values())
    # fewest columns, then the earliest epoch, within 0.005 of the best
    order = sorted(res, key=lambda k: ([n for n, _ in SETS].index(res[k]["name"]), res[k]["ep"]))
    pick = next(k for k in order if res[k]["AP"] >= best - .005)
    print("locked", pick, "best", round(best, 4))
    json.dump(dict(locked=pick, best=best, results=res), open(O / "selection.json", "w"), indent=1)
else:
    lock = json.load(open(O / "selection.json"))["locked"]; name, ep = lock.split("|ep"); ep = int(ep); cols = dict(SETS)[name]
    from src.context_controls import load_component_lookup
    cfg = json.load(open(V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
    P = {k: [] for k in ("locked", "V4", "B", "old_B+det", "V4+det")}; D = {k: [] for k in P}; Y, FO, CO = [], [], []; per = {}; dec = {}
    for k in range(5):
        ic, yc, Fc = load(k, "calibration", ep); im, ym, Fm = load(k, "model_val", ep); it, y, Ft = load(k, "test", ep)
        F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]; row = {}
        oof = np.r_[fit_apply(Fm, ym, Fc, cols), fit_apply(Fc, yc, Fm, cols)]; s = fit_apply(F, yf, Ft, cols); P["locked"].append(s); D["locked"].append(s >= thr90(oof, yf)); row["locked"] = ap(y, s)
        dec[k] = {i: bool(v >= 0) for i, v in zip(it, s)}
        _, yc2, Fc2 = load(k, "calibration", 2); _, ym2, Fm2 = load(k, "model_val", 2); _, _, Ft2 = load(k, "test", 2); F2 = {n: np.r_[Fc2[n], Fm2[n]] for n in Fc2}
        c0 = ["B", "Bh", "flag", "s1"]; oof = np.r_[fit_apply(Fm2, ym2, Fc2, c0), fit_apply(Fc2, yc2, Fm2, c0)]; s = fit_apply(F2, yf, Ft2, c0); P["old_B+det"].append(s); D["old_B+det"].append(s >= thr90(oof, yf)); row["old_B+det"] = ap(y, s)
        c1 = dict(SETS)["V4+det"]; oof = np.r_[fit_apply(Fm, ym, Fc, c1), fit_apply(Fc, yc, Fm, c1)]; s = fit_apply(F, yf, Ft, c1); P["V4+det"].append(s); D["V4+det"].append(s >= thr90(oof, yf)); row["V4+det"] = ap(y, s)
        for n, d in (("V4", "local_real_noaux"), ("B", "B")):
            th = json.load(open(V4 / f"fold_{k}" / d / "operating.json"))["operating_point"]["threshold"]
            P[n].append(Ft[n]); D[n].append(Ft[n] >= np.log(th / (1 - th))); row[n] = ap(y, Ft[n])
        per[k] = row; Y.append(y); FO += [k] * len(y); CO += [lookup[i] for i in it]; print(k, {a: round(b, 3) for a, b in row.items()}, flush=True)
    Y = np.concatenate(Y).astype(bool); FO = np.array(FO); CO = np.array(CO); P = {k: np.concatenate(v) for k, v in P.items()}; D = {k: np.concatenate(v) for k, v in D.items()}
    sig = lambda s: 1 / (1 + np.exp(-s)); res = {}
    for k in P:
        se, sp = float(D[k][Y].mean()), float((~D[k][~Y]).mean()); res[k] = dict(AP=ap(Y, P[k]), AUROC=float(roc_auc_score(Y, P[k])), sens90=se, spec90=sp, BA90=(se + sp) / 2, brier=float(((sig(P[k]) - Y) ** 2).mean())); print(k, {a: round(b, 4) for a, b in res[k].items()})
    rng = np.random.default_rng(20261105); G = [[np.where((FO == f) & (CO == c))[0] for c in np.unique(CO[FO == f])] for f in range(5)]
    pairs = [("locked", "V4"), ("locked", "B"), ("locked", "old_B+det"), ("locked", "V4+det")]; dr = {p: dict(AP=[], BA=[], sens=[], spec=[]) for p in pairs}
    for _ in range(2000):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in G]); y = Y[ix]
        for a, b in pairs:
            se = D[a][ix][y].mean() - D[b][ix][y].mean(); sp = (~D[a][ix][~y]).mean() - (~D[b][ix][~y]).mean(); dr[(a, b)]["AP"].append(ap(y, P[a][ix]) - ap(y, P[b][ix])); dr[(a, b)]["sens"].append(se); dr[(a, b)]["spec"].append(sp); dr[(a, b)]["BA"].append((se + sp) / 2)
    ci = {f"{a}-{b}": {m: [float(x) for x in np.percentile(v, [2.5, 97.5])] for m, v in d.items()} for (a, b), d in dr.items()}
    for k, v in ci.items(): print(k, {m: np.round(x, 3).tolist() for m, x in v.items()})
    json.dump(dict(locked=lock, perfold=per, pooled=res, ci=ci), open(O / "test_results.json", "w"), indent=1); json.dump(dec, open(O / "test_decisions.json", "w"))
