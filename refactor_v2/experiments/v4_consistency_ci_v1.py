"""Test QA of the locked system (B + extended detector + consistency layer, detector location) against sealed V4, by question group, with the study's component bootstrap."""
import json
import numpy as np
from collections import defaultdict
from experiments.v4_detector_stack_v2 import V4, O, load, SETS
from experiments.v4_poststack_select_v1 import fit_apply
from experiments.v4_consistency_v1 import fix, contradictions, CAN, DESC, WHERE, CNT, ABN
from src.context_controls import load_component_lookup
from src.utils import normalize_answer as na
from pathlib import Path
name, ep = json.load(open(O / "selection.json"))["locked"].split("|ep"); ep = int(ep); cols = dict(SETS)[name]
cfg = json.load(open(V4 / "fold_0/B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
A, Bb, W, FO, CO, G, contra = [], [], [], [], [], [], defaultdict(int); per = {}
for k in range(5):
    _, yc, Fc = load(k, "calibration", ep); _, ym, Fm = load(k, "model_val", ep); it, _, Ft = load(k, "test", ep); F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}
    d = dict(zip(it, fit_apply(F, np.r_[yc, ym], Ft, cols) >= 0)); sc = json.load(open(O / f"fold_{k}/scores_ep{ep}.json"))
    B = json.load(open(V4 / f"fold_{k}/B/test_predictions.json"))["records"]; v = json.load(open(V4 / f"fold_{k}/local_real_noaux/test_predictions.json"))["records"]
    by = defaultdict(list); byb = defaultdict(list); byv = defaultdict(list); fold_a = fold_w = 0
    for r, x in zip(B, v):
        assert (r["image_id"], r["question"]) == (x["image_id"], x["question"]); n = dict(r, pred=fix(r, bool(d[r["image_id"]]), sc.get(r["image_id"]), "detector"))
        by[r["image_id"]].append(n); byb[r["image_id"]].append(r); byv[r["image_id"]].append(x)
        a = na(n["pred"]) == na(n["gt"]); w = na(x["pred"]) == na(x["gt"]); A.append(a); W.append(w); Bb.append(na(r["pred"]) == na(r["gt"])); FO.append(k); CO.append(lookup[r["image_id"]]); fold_a += a; fold_w += w
        q = r["question"]; G.append("canonical" if q == CAN else q if q in (DESC, WHERE, CNT, ABN) else "fracture_closed" if "fractur" in q.lower() else "other")
    per[k] = dict(locked=fold_a / sum(1 for f in FO if f == k), V4=fold_w / sum(1 for f in FO if f == k))
    for i in by: contra["locked"] += contradictions(by[i]); contra["B"] += contradictions(byb[i]); contra["V4"] += contradictions(byv[i])
A, W, Bb, FO, CO, G = map(np.array, (A, W, Bb, FO, CO, G)); assert len(A) == 17998
rng = np.random.default_rng(20261105); grp = [[np.where((FO == f) & (CO == c))[0] for c in np.unique(CO[FO == f])] for f in range(5)]; D = []
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in grp]); D.append(A[ix].mean() - W[ix].mean())
out = dict(QA=dict(locked=float(A.mean()), V4=float(W.mean()), B=float(Bb.mean())), minus_V4_CI=[float(x) for x in np.percentile(D, [2.5, 97.5])], contradictions=dict(contra), perfold=per,
    groups={g: dict(n=int((G == g).sum()), locked=float(A[G == g].mean()), V4=float(W[G == g].mean()), B=float(Bb[G == g].mean())) for g in np.unique(G)})
print(json.dumps(out, indent=1)); json.dump(out, open(O / "test_qa.json", "w"), indent=1)
