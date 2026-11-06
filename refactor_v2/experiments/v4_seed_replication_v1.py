"""Protocol reports/v4_detector_ext_v1/seed43/PROTOCOL.md: the locked system with the seed-43 detectors on pooled test."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import experiments.v4_detector_stack_v2 as m
from experiments.v4_poststack_select_v1 import ap, fit_apply
from experiments.v4_consistency_v1 import fix, contradictions
from src.context_controls import load_component_lookup
from src.utils import normalize_answer as na
cfg = json.load(open(m.V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
COLS = dict(m.SETS)["B+det"]; ROOT = m.O; R = {}
for seed, base in ((42, ROOT), (43, ROOT / "seed43")):
    m.O = base; S, Y, V, F, C, A, W, QF, QC = [], [], [], [], [], [], [], [], []; bad = 0; per = []
    for k in range(5):
        _, yc, Fc = m.load(k, "calibration", 6); _, ym, Fm = m.load(k, "model_val", 6); it, y, Ft = m.load(k, "test", 6); Fa = {n: np.r_[Fc[n], Fm[n]] for n in Fc}
        s = fit_apply(Fa, np.r_[yc, ym], Ft, COLS); S.append(s); Y.append(y); V.append(Ft["V4"]); F += [k] * len(y); C += [lookup[i] for i in it]; per.append(ap(y, s)); d = dict(zip(it, s >= 0)); sc = json.load(open(base / f"fold_{k}/scores_ep6.json"))
        by = defaultdict(list); Vr = json.load(open(m.V4 / f"fold_{k}/local_real_noaux/test_predictions.json"))["records"]
        for r, x in zip(json.load(open(m.V4 / f"fold_{k}/B/test_predictions.json"))["records"], Vr):
            n = dict(r, pred=fix(r, bool(d[r["image_id"]]), sc.get(r["image_id"]), "detector")); by[r["image_id"]].append(n); A.append(na(n["pred"]) == na(n["gt"])); W.append(na(x["pred"]) == na(x["gt"])); QF.append(k); QC.append(lookup[r["image_id"]])
        bad += sum(contradictions(v) for v in by.values())
    R[seed] = dict(S=np.concatenate(S), Y=np.concatenate(Y).astype(bool), V=np.concatenate(V), F=np.array(F), C=np.array(C), A=np.array(A), W=np.array(W), QF=np.array(QF), QC=np.array(QC), bad=bad, per=per)
a, b = R[43], R[42]; rng = np.random.default_rng(20261105); keys = [(f, c) for f in range(5) for c in np.unique(a["C"][a["F"] == f])]
ig = {x: np.where((a["F"] == x[0]) & (a["C"] == x[1]))[0] for x in keys}; qg = {x: np.where((a["QF"] == x[0]) & (a["QC"] == x[1]))[0] for x in keys}; byf = {f: [x for x in keys if x[0] == f] for f in range(5)}; D1, D2, D3 = [], [], []
for _ in range(2000):
    pick = [fk[j] for fk in byf.values() for j in rng.integers(0, len(fk), len(fk))]; ix = np.concatenate([ig[p] for p in pick]); qx = np.concatenate([qg[p] for p in pick]); y = a["Y"][ix]
    D1.append(ap(y, a["S"][ix]) - ap(y, a["V"][ix])); D2.append(ap(y, a["S"][ix]) - ap(y, b["S"][ix])); D3.append(a["A"][qx].mean() - a["W"][qx].mean())
c = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]; A43, A42 = ap(a["Y"], a["S"]), ap(a["Y"], b["S"])
out = dict(AP_seed43=A43, AP_seed42=A42, AP_V4=ap(a["Y"], a["V"]), per_fold_43=a["per"], per_fold_42=b["per"], minus_V4_CI=c(D1), minus_seed42_CI=c(D2), QA_seed43=float(a["A"].mean()), QA_seed42=float(b["A"].mean()), QA_V4=float(a["W"].mean()), QA_minus_V4_CI=c(D3), contradictions=a["bad"],
    criteria=dict(AP_lower_gt_0=c(D1)[0] > 0, AP_abs_diff_le_0015=abs(A43 - A42) <= .015, QA_lower_gt_0=c(D3)[0] > 0))
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT / "seed43/results.json", "w"), indent=1)
