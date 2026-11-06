"""Post hoc, 2026-11-06: V4 nhận đúng các nâng cấp của hệ thống khóa (head-refit + cờ + điểm detector epoch 6 + lớp nhất quán, vị trí từ detector).
Cùng cross-fit, cùng bộ chấm, cùng bootstrap thành phần như v4_consistency_ci_v1 và v4_detector_stack_v2. Test đã mở từ trước: không phải kiểm định.
Ghi reports/v4_detector_ext_v1/v4arm_full.json; không ghi đè file cũ."""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiments.v4_detector_stack_v2 import V4, O, load, SETS, thr90
from experiments.v4_poststack_select_v1 import fit_apply, ap
from experiments.v4_consistency_v1 import fix, contradictions, CAN, DESC, WHERE, CNT, ABN
from src.context_controls import load_component_lookup
from src.utils import normalize_answer as na

EP = 6; ARMS = {"locked": ("B+det", "B"), "V4full": ("V4+det", "local_real_noaux")}
cfg = json.load(open(V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
QA = defaultdict(list); G, FO, CO = [], [], []; contra = defaultdict(int); P = defaultdict(list); D = defaultdict(list); Y, FI, CI_ = [], [], []
for k in range(5):
    ic, yc, Fc = load(k, "calibration", EP); im, ym, Fm = load(k, "model_val", EP); it, y, Ft = load(k, "test", EP)
    F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]; sc = json.load(open(O / f"fold_{k}/scores_ep{EP}.json")); dec = {}
    for arm, (name, _) in ARMS.items():
        cols = dict(SETS)[name]; oof = np.r_[fit_apply(Fm, ym, Fc, cols), fit_apply(Fc, yc, Fm, cols)]; s = fit_apply(F, yf, Ft, cols)
        P[arm].append(s); D[arm].append(s >= thr90(oof, yf)); dec[arm] = dict(zip(it, s >= 0))
    th = json.load(open(V4 / f"fold_{k}/local_real_noaux/operating.json"))["operating_point"]["threshold"]
    P["V4"].append(Ft["V4"]); D["V4"].append(Ft["V4"] >= np.log(th / (1 - th))); Y.append(y); FI += [k] * len(y); CI_ += [lookup[i] for i in it]
    rec = {d: json.load(open(V4 / f"fold_{k}/{d}/test_predictions.json"))["records"] for d in ("B", "local_real_noaux")}
    by = defaultdict(lambda: defaultdict(list))
    for rb, rv in zip(rec["B"], rec["local_real_noaux"]):
        assert (rb["image_id"], rb["question"]) == (rv["image_id"], rv["question"]); i = rb["image_id"]
        rows = dict(B=rb, V4=rv, locked=dict(rb, pred=fix(rb, bool(dec["locked"][i]), sc.get(i), "detector")), V4full=dict(rv, pred=fix(rv, bool(dec["V4full"][i]), sc.get(i), "detector")))
        for a, r in rows.items(): QA[a].append(na(r["pred"]) == na(r["gt"])); by[a][i].append(r)
        q = rb["question"]; G.append("canonical" if q == CAN else q if q in (DESC, WHERE, CNT, ABN) else "fracture_closed" if "fractur" in q.lower() else "other"); FO.append(k); CO.append(lookup[i])
    for a in by:
        for i in by[a]: contra[a] += contradictions(by[a][i])
    print("fold", k, "AP", {a: round(ap(y, P[a][-1]), 3) for a in ("locked", "V4full", "V4")}, flush=True)
QA = {a: np.array(v) for a, v in QA.items()}; G, FO, CO = map(np.array, (G, FO, CO)); assert len(G) == 17998
Y = np.concatenate(Y).astype(bool); FI, CI_ = np.array(FI), np.array(CI_); P = {a: np.concatenate(v) for a, v in P.items()}; D = {a: np.concatenate(v) for a, v in D.items()}
sig = lambda s: 1 / (1 + np.exp(-s)); frac = {}
for a in P:
    se, sp = float(D[a][Y].mean()), float((~D[a][~Y]).mean()); frac[a] = dict(AP=ap(Y, P[a]), sens90=se, spec90=sp, BA90=(se + sp) / 2, brier=float(((sig(P[a]) - Y) ** 2).mean()))
rng = np.random.default_rng(20261105); gi = [[np.where((FI == f) & (CI_ == c))[0] for c in np.unique(CI_[FI == f])] for f in range(5)]
gq = [[np.where((FO == f) & (CO == c))[0] for c in np.unique(CO[FO == f])] for f in range(5)]; pairs = [("V4full", "locked"), ("V4full", "V4")]; R = {p: defaultdict(list) for p in pairs}
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in gi]); y = Y[ix]
    jx = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in gq])
    for a, b in pairs:
        se = D[a][ix][y].mean() - D[b][ix][y].mean(); sp = (~D[a][ix][~y]).mean() - (~D[b][ix][~y]).mean()
        R[(a, b)]["AP"].append(ap(y, P[a][ix]) - ap(y, P[b][ix])); R[(a, b)]["BA"].append((se + sp) / 2); R[(a, b)]["QA"].append(QA[a][jx].mean() - QA[b][jx].mean())
ci = {f"{a}-{b}": {m: [float(x) for x in np.percentile(v, [2.5, 97.5])] for m, v in d.items()} for (a, b), d in R.items()}
out = dict(note="post hoc; test opened earlier; V4full = V4 + V4 head-refit + flag + detector ep6 max score + consistency layer (detector location)", fracture=frac,
           QA={a: float(v.mean()) for a, v in QA.items()}, contradictions=dict(contra), ci=ci,
           groups={g: {a: float(QA[a][G == g].mean()) for a in QA} | dict(n=int((G == g).sum())) for g in np.unique(G)})
print(json.dumps(out, indent=1, ensure_ascii=False)); json.dump(out, open(O / "v4arm_full.json", "w"), indent=1, ensure_ascii=False)
