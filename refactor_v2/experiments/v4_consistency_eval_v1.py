"""Non-test role by default: contradictions and QA before and after the consistency layer, for each location source, with the fold's cross-fit decision."""
import json, sys
import numpy as np
from collections import defaultdict
from experiments.v4_detector_stack_v2 import V4, O, load, SETS, thr90
from experiments.v4_poststack_select_v1 import fit_apply
from experiments.v4_consistency_v1 import fix, contradictions, CAN, DESC, WHERE, CNT, ABN
from src.utils import normalize_answer as na
name, ep = sys.argv[2], int(sys.argv[3]); cols = dict(SETS)[name]; base = "B" if name.startswith("B") else "local_real_noaux"
tot = defaultdict(int)
for k in range(5):
    ic, yc, Fc = load(k, "calibration", ep); im, ym, Fm = load(k, "model_val", ep); sc = json.load(open(O / f"fold_{k}/scores_ep{ep}.json"))
    if sys.argv[1] == "test":
        it, _, Ft = load(k, "test", ep); F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; dec = [("test", dict(zip(it, fit_apply(F, np.r_[yc, ym], Ft, cols) >= 0)))]
    else:
        dec = [("calibration", dict(zip(ic, fit_apply(Fm, ym, Fc, cols) >= 0))), ("model_val", dict(zip(im, fit_apply(Fc, yc, Fm, cols) >= 0)))]
    for role, d in dec:
        rows = json.load(open(V4 / f"fold_{k}/{base}/{role}_predictions.json"))["records"]
        by = defaultdict(list)
        for r in rows:
            by[r["image_id"]].append(r); tot["n"] += 1; ok = na(r["pred"]) == na(r["gt"]); tot["base_ok"] += ok
            if r["question"] in (DESC, WHERE, CNT, ABN) or "fractur" in r["question"].lower(): tot["dep_n"] += 1; tot["dep_base_ok"] += ok
        for i, rs in by.items():
            tot["base_contra"] += contradictions(rs)
            tot["stack_only_contra"] += contradictions([dict(r, pred=("yes" if d[i] else "no") if r["question"] == CAN else r["pred"]) for r in rs])
            for loc in ("model", "detector"):
                new = [dict(r, pred=fix(r, bool(d[i]), sc.get(i), loc)) for r in rs]
                tot[f"{loc}_contra"] += contradictions(new)
                for r in new:
                    ok = na(r["pred"]) == na(r["gt"]); tot[f"{loc}_ok"] += ok
                    if r["question"] in (DESC, WHERE, CNT, ABN) or "fractur" in r["question"].lower(): tot[f"{loc}_dep_ok"] += ok
print(sys.argv[1:], {k: (v if "contra" in k or k.endswith("n") else round(v / (tot["dep_n"] if "dep" in k else tot["n"]), 4)) for k, v in tot.items()})
json.dump(tot, open(O / f"consistency_{sys.argv[1]}.json", "w"), indent=1)
