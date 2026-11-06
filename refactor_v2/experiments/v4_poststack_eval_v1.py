"""Leakage checks, then the locked scorer on outer test, 5 folds.  Same pairing and component bootstrap as the v4 study.
Controls (not selection candidates): detector score alone, and B's two heads + flag + score, which uses no crops."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.v4_poststack_select_v1 import V4, OUT, CAN, ap, fit_apply, load
from src.context_controls import load_component_lookup

ROOT = Path(sys.argv[1]).resolve(); LOCK = json.load(open(OUT / "selection.json"))["locked"]; assert LOCK == "V4+V4h+flag+score"
SETS = {"locked": ["V4", "V4h", "flag", "score"], "det_only": ["flag", "score"], "nocrop": ["B", "Bh", "flag", "score"]}
cfg = json.load(open(V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], ROOT)
meta = json.load(open(V4 / "metadata.json")); leak = {}


def thr90(s, y):
    n = np.sort(s[y == 0]); return float(np.nextafter(n[int(np.ceil(.9 * len(n))) - 1], np.inf))


A = {k: dict(p=[], d=[]) for k in list(SETS) + ["V4", "V4h", "B", "Bh"]}; Y, FOLD, COMP = [], [], []; Q = {}; perfold = {}
for k in range(5):
    tr = {r["image_id"] for r in json.load(open(meta[str(k)]["roles"]["train"]["path"]))}
    ic, yc, Fc = load(k, "calibration"); im, ym, Fm = load(k, "model_val"); it, y, Ft = load(k, "test")
    other = set(ic) | set(im) | set(it)
    leak[k] = dict(train_overlap=len(tr & other), component_overlap=len({lookup[i] for i in tr} & {lookup[i] for i in other}))
    F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]; row = {}
    for n, names in SETS.items():
        oof = np.r_[fit_apply(Fm, ym, Fc, names), fit_apply(Fc, yc, Fm, names)]
        s = fit_apply(F, yf, Ft, names); th = thr90(oof, yf)
        A[n]["p"].append(s); A[n]["d"].append(s >= th); row[n] = ap(y, s)
    for n, d in dict(V4="local_real_noaux", V4h="local_real_noaux/head_refit", B="B", Bh="B/head_refit").items():
        th = json.load(open(V4 / f"fold_{k}" / d / "operating.json"))["operating_point"]["threshold"]
        s = Ft[n]; A[n]["p"].append(s); A[n]["d"].append(s >= np.log(th / (1 - th))); row[n] = ap(y, s)
    perfold[k] = row; Y.append(y); FOLD += [k] * len(y); COMP += [lookup[i] for i in it]
    print(k, leak[k], {a: round(b, 3) for a, b in row.items()}, flush=True)
assert all(v == dict(train_overlap=0, component_overlap=0) for v in leak.values()), leak
Y = np.concatenate(Y).astype(bool); FOLD = np.array(FOLD); COMP = np.array(COMP)
P = {k: np.concatenate(v["p"]) for k, v in A.items()}; D = {k: np.concatenate(v["d"]) for k, v in A.items()}
sig = lambda s: 1 / (1 + np.exp(-s))
res = {}
for k in P:
    se, sp = float(D[k][Y].mean()), float((~D[k][~Y]).mean())
    res[k] = dict(AP=ap(Y, P[k]), AUROC=float(roc_auc_score(Y, P[k])), sens90=se, spec90=sp, BA90=(se + sp) / 2, brier=float(((sig(P[k]) - Y) ** 2).mean()))
    print(k, {a: round(b, 4) for a, b in res[k].items()}, flush=True)
rng = np.random.default_rng(20261105)
groups = [[np.where((FOLD == f) & (COMP == c))[0] for c in np.unique(COMP[FOLD == f])] for f in range(5)]
pairs = [("locked", "V4"), ("locked", "V4h"), ("locked", "B"), ("V4", "B"), ("locked", "nocrop"), ("nocrop", "B"), ("nocrop", "V4"), ("locked", "det_only")]
dr = {p: dict(AP=[], BA=[], sens=[], spec=[]) for p in pairs}
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in groups]); y = Y[ix]
    for a, b in pairs:
        se = D[a][ix][y].mean() - D[b][ix][y].mean(); sp = (~D[a][ix][~y]).mean() - (~D[b][ix][~y]).mean()
        dr[(a, b)]["AP"].append(ap(y, P[a][ix]) - ap(y, P[b][ix])); dr[(a, b)]["sens"].append(se); dr[(a, b)]["spec"].append(sp); dr[(a, b)]["BA"].append((se + sp) / 2)
ci = {f"{a}-{b}": {m: [float(x) for x in np.percentile(v, [2.5, 97.5])] for m, v in d.items()} for (a, b), d in dr.items()}
for k, v in ci.items(): print(k, {m: np.round(x, 3).tolist() for m, x in v.items()}, flush=True)
json.dump(dict(locked=LOCK, leak=leak, perfold=perfold, pooled=res, ci=ci), open(OUT / "test_results.json", "w"), indent=1)
