"""Plan reports/v4_detector_ext_v1/seed43/B_RETRAIN_PLAN.md: B retrain sensitivity on fold 0. Argument dry = fit on calibration, read model_val, no test."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import experiments.v4_detector_stack_v2 as m
from experiments.v4_poststack_select_v1 import ap, fit_apply, logit
from experiments.v4_consistency_v1 import fix, contradictions
from src.context_controls import load_component_lookup
from src.utils import normalize_answer as na
DRY = sys.argv[1:] == ["dry"]; FIT, READ = (["calibration"], "model_val") if DRY else (["calibration", "model_val"], "test")
V5 = Path("reports/v5_studies/factorial_fold0/fold_0"); SEALED = m.V4 / "fold_0"; ROOT = m.O; COLS = ["B", "flag", "s1"]
RUNS = dict(sealed=SEALED / "B", seed42=V5 / "B_seed42") | ({} if DRY else dict(seed43=V5 / "B_seed43")); DET = {42: ROOT, 43: ROOT / "seed43"}
cfg = json.load(open(SEALED / "B.json")); lookup, _ = load_component_lookup(cfg["train"]["component_batching"]["manifest"], cfg["train"]["component_batching"]["seal"], Path("/Users/uyen/Downloads/BONEVQA"))
rec = lambda d, role: json.load(open(d / f"{role}_predictions.json"))["records"]
can = lambda rs: [r for r in rs if r["question"] == m.CAN]
right = lambda rs: np.array([na(r["pred"]) == na(r["gt"]) for r in rs])


def feats(d, role, sc):
    rows = can(rec(d, role)); S = [sc[r["image_id"]]["scores"] for r in rows]; low = logit(.05) - 1
    return [r["image_id"] for r in rows], np.array([r["gt"] == "yes" for r in rows], float), dict(B=logit([r["p_yes"] for r in rows]), flag=np.array([bool(x) for x in S], float), s1=np.array([logit(x[0]) if x else low for x in S]))


Vr = rec(SEALED / "local_real_noaux", READ); Vc = can(Vr); out = dict(role=READ, V4=dict(AP=ap(np.array([r["gt"] == "yes" for r in Vc], float), logit([r["p_yes"] for r in Vc])), QA=float(right(Vr).mean()))); keep = {}
for name, d in RUNS.items():
    for ds, base in DET.items():
        sc = json.load(open(base / "fold_0/scores_ep6.json")); parts = [feats(d, r, sc) for r in FIT]; it, y, Ft = feats(d, READ, sc)
        Fa = {c: np.concatenate([p[2][c] for p in parts]) for c in COLS}; s = fit_apply(Fa, np.concatenate([p[1] for p in parts]), Ft, COLS); yes = dict(zip(it, s >= 0)); by = defaultdict(list); rows = rec(d, READ)
        assert [r["image_id"] for r in rows] == [r["image_id"] for r in Vr] and [r["question"] for r in rows] == [r["question"] for r in Vr]
        for r in rows:
            by[r["image_id"]].append(dict(r, pred=fix(r, bool(yes[r["image_id"]]), sc.get(r["image_id"]), "detector")))
        fixed = [n for v in by.values() for n in v]
        A = np.array([na(n["pred"]) == na(n["gt"]) for n in fixed]); out[f"{name}|det{ds}"] = dict(AP=ap(y, s), QA=float(A.mean()), contradictions=sum(contradictions(v) for v in by.values()), plain_AP=ap(y, Ft["B"]), plain_QA=float(right(rows).mean()))
        keep[name, ds] = (it, y, s, fixed)
if not DRY:
    it, y, s, fixed = keep["seed43", 42]; V = logit([r["p_yes"] for r in Vc]); C = np.array([lookup[i] for i in it]); keys = np.unique(C); rng = np.random.default_rng(20261105); D1, D2 = [], []
    A = np.array([na(n["pred"]) == na(n["gt"]) for n in fixed]); QC = np.array([lookup[n["image_id"]] for n in fixed]); W = {(r["image_id"], r["question"]): na(r["pred"]) == na(r["gt"]) for r in Vr}; Wa = np.array([W[n["image_id"], n["question"]] for n in fixed])
    ig = {c: np.where(C == c)[0] for c in keys}; qg = {c: np.where(QC == c)[0] for c in keys}
    for _ in range(2000):
        p = rng.choice(keys, len(keys)); ix = np.concatenate([ig[c] for c in p]); qx = np.concatenate([qg[c] for c in p]); D1.append(ap(y[ix], s[ix]) - ap(y[ix], V[ix])); D2.append(A[qx].mean() - Wa[qx].mean())
    out["seed43_det42_minus_V4"] = dict(AP_CI=[float(x) for x in np.percentile(D1, [2.5, 97.5])], QA_CI=[float(x) for x in np.percentile(D2, [2.5, 97.5])])
print(json.dumps(out, indent=1))
if not DRY:
    json.dump(out, open(ROOT / "seed43/B_retrain_fold0.json", "w"), indent=1)
