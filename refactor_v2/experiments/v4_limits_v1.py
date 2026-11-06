"""Limits of the locked system that need no new training: selection-aware nested estimate, localisation, body-region subgroups, calibration,
per-fold operating point, and an export for an independent audit."""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.v4_detector_stack_v2 import V4, O, load, SETS, thr90
from experiments.v4_poststack_select_v1 import ap, fit, fit_apply, design, CAN
from experiments.v4_consistency_v1 import fix, WHERE, DESC
from src.context_controls import load_component_lookup
from src.utils import normalize_answer as na
EPS = [2, 3, 4, 5, 6]; NAME, EP = "B+det", 6; COLS = dict(SETS)[NAME]; out = {}; rng = np.random.default_rng(20261105)
cfg = json.load(open(V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
L = {(k, e): (load(k, "calibration", e), load(k, "model_val", e), load(k, "test", e)) for k in range(5) for e in EPS}


def crossfit(k, e, cols):
    (_, yc, Fc), (_, ym, Fm), _ = L[(k, e)]; return np.r_[yc, ym], np.r_[fit_apply(Fm, ym, Fc, cols), fit_apply(Fc, yc, Fm, cols)]


def final(k, e, cols):
    (_, yc, Fc), (_, ym, Fm), (it, y, Ft) = L[(k, e)]; F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]
    oy, os_ = crossfit(k, e, cols); return it, y, fit_apply(F, yf, Ft, cols), thr90(os_, oy), Ft


# 1. nested: each fold's configuration is chosen from the other four folds' non-test data, by the same rule
CF = {(k, e, n): crossfit(k, e, c) for k in range(5) for e in EPS for n, c in SETS}; picks = {}; NP, NY, NV, NF, NC = [], [], [], [], []
names = [n for n, _ in SETS]
for k in range(5):
    A = {(n, e): ap(np.concatenate([CF[(j, e, n)][0] for j in range(5) if j != k]), np.concatenate([CF[(j, e, n)][1] for j in range(5) if j != k])) for n in names for e in EPS}
    best = max(A.values()); pick = next(c for c in sorted(A, key=lambda c: (names.index(c[0]), c[1])) if A[c] >= best - .005); picks[k] = pick
    it, y, s, _, Ft = final(k, pick[1], dict(SETS)[pick[0]]); NP.append(s); NY.append(y); NV.append(Ft["V4"]); NF += [k] * len(y); NC += [lookup[i] for i in it]
NP, NY, NV, NF, NC = map(np.concatenate, (NP, NY, NV)) + (np.array(NF), np.array(NC)) if False else (np.concatenate(NP), np.concatenate(NY), np.concatenate(NV), np.array(NF), np.array(NC))
G = [[np.where((NF == f) & (NC == c))[0] for c in np.unique(NC[NF == f])] for f in range(5)]; D = []
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in G]); D.append(ap(NY[ix], NP[ix]) - ap(NY[ix], NV[ix]))
out["nested"] = dict(picks={str(k): list(v) for k, v in picks.items()}, AP=ap(NY, NP), AP_V4=ap(NY, NV), minus_V4_CI=[float(x) for x in np.percentile(D, [2.5, 97.5])]); print(out["nested"], flush=True)

# 2. localisation on test, epoch 2 vs 6, against boxes from a random same-fold image of the same size
def iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.
meta = json.load(open(V4 / "metadata.json")); loc = {e: defaultdict(float) for e in (2, 6)}; r2 = np.random.default_rng(0)
for k in range(5):
    gt = {r["image_id"]: [b[:4] for b in r["boxes"]] for r in json.load(open(meta[str(k)]["roles"]["test"]["path"])) if r.get("boxes")}
    for e in loc:
        sc = json.load(open(O / f"fold_{k}/scores_ep{e}.json")); by = defaultdict(list); ts = set(L[(k, 2)][2][0])
        for i in ts: by[tuple(sc[i]["size"])].append(i)
        for i, g in gt.items():
            if i not in ts: continue
            c = by[tuple(sc[i]["size"])]; other = sc[c[r2.integers(len(c))]]["boxes"]; me = sc[i]["boxes"]; d = loc[e]; d["n"] += 1
            for nm, bx in (("own", me), ("ctl", other)):
                for t in (.1, .3, .5): d[f"{nm}_iou{t}"] += any(iou(x, y) >= t for x in bx for y in g)
                d[f"{nm}_top1_iou.3"] += bool(bx) and any(iou(bx[0], y) >= .3 for y in g)
out["localisation"] = {str(e): {a: (b if a == "n" else b / v["n"]) for a, b in v.items()} for e, v in loc.items()}; print(out["localisation"], flush=True)

# 3. Where-question location source on non-test, and body-region subgroups + calibration + operating point on test
W = defaultdict(int)
for k in range(5):
    sc = json.load(open(O / f"fold_{k}/scores_ep{EP}.json"))
    for role in ("calibration", "model_val"):
        for r in json.load(open(V4 / f"fold_{k}/B/{role}_predictions.json"))["records"]:
            if r["question"] == WHERE and r["gt"]: 
                W["n"] += 1; W["model"] += na(r["pred"]) == na(r["gt"]); W["detector"] += na(fix(r, True, sc.get(r["image_id"]), "detector")) == na(r["gt"])
out["where_nontest_given_yes"] = dict(W); print(dict(W), flush=True)
AY, AS, AV, AR, AF, op = [], [], [], [], [], {}; export = {}; params = {}
for k in range(5):
    it, y, s, th, Ft = final(k, EP, COLS); (_, yc, Fc), (_, ym, Fm), _ = L[(k, EP)]; F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]
    ms = {n: (float(F[n].mean()), float(F[n].std()) or 1.) for n in COLS}; params[k] = dict(columns=COLS, mean_sd=ms, weights=fit(design(F, COLS, ms), yf).tolist(), threshold_spec90=th, threshold_raw=0.0, detector=f"reports/v4_detector_ext_v1/fold_{k}/ep6.pt")
    rows = json.load(open(V4 / f"fold_{k}/B/test_predictions.json"))["records"]; reg = {}
    for r in rows:
        if r["question"] == "Which body region is shown in this X-ray?": reg[r["image_id"]] = r["gt"]
        elif r["question"] == DESC: reg.setdefault(r["image_id"], re.sub(r".* in the (.*?)( at .*)?\.$", r"\1", r["gt"]))
    sc = json.load(open(O / f"fold_{k}/scores_ep{EP}.json")); dmap = dict(zip(it, s >= 0)); pmap = dict(zip(it, s))
    export[k] = [dict(image_id=r["image_id"], question=r["question"], answer_type=r["answer_type"], gt=r["gt"], pred=fix(r, bool(dmap[r["image_id"]]), sc.get(r["image_id"]), "detector"), s=float(pmap[r["image_id"]]) if r["question"] == CAN else None) for r in rows]
    d = s >= th; op[k] = dict(sens=float(d[y == 1].mean()), spec=float((~d[y == 0]).mean()), proposal_neg=float(np.mean([bool(sc[i]["scores"]) for i, t in zip(it, y) if t == 0])))
    AY.append(y); AS.append(s); AV.append(Ft["V4"]); AR += [reg.get(i, "unknown") for i in it]
AY, AS, AV, AR = np.concatenate(AY), np.concatenate(AS), np.concatenate(AV), np.array(AR); out["operating_by_fold"] = op; print(op, flush=True)
out["regions"] = {g: dict(n=int((AR == g).sum()), pos=int(AY[AR == g].sum()), AP=ap(AY[AR == g], AS[AR == g]), AP_V4=ap(AY[AR == g], AV[AR == g])) for g in np.unique(AR) if AY[AR == g].sum() >= 20}; print(out["regions"], flush=True)
p = 1 / (1 + np.exp(-AS)); bins = np.minimum((p * 10).astype(int), 9); ece = sum(abs(p[bins == b].mean() - AY[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
w = fit(np.column_stack([np.ones(len(AS)), AS]), AY, ridge=0.)
out["calibration"] = dict(ECE=float(ece), intercept=float(w[0]), slope=float(w[1]), mean_p=float(p.mean()), prevalence=float(AY.mean())); print(out["calibration"], flush=True)
json.dump(out, open(O / "limits.json", "w"), indent=1); json.dump(params, open(O / "stack_params.json", "w"), indent=1); json.dump(export, open(O / "final_test_answers.json", "w"))
