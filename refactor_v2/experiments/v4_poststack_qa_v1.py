"""QA accuracy and answer consistency of the locked scorer, plus cross-fit non-test AP of the no-crop control.  Only the canonical answer changes."""
import json
import numpy as np
from experiments.v4_poststack_select_v1 import V4, OUT, CAN, ap, fit_apply, load
from src.utils import normalize_answer as na
L = ["V4", "V4h", "flag", "score"]; N = ["B", "Bh", "flag", "score"]
tot = cor_base = cor_raw = 0; cf = {"V4": [], "locked": [], "nocrop": []}; Yv = []; flips = up = dn = 0; contra_base = contra_new = opens = 0
for k in range(5):
    _, yc, Fc = load(k, "calibration"); _, ym, Fm = load(k, "model_val"); it, y, Ft = load(k, "test")
    F = {n: np.r_[Fc[n], Fm[n]] for n in Fc}; yf = np.r_[yc, ym]; Yv.append(yf)
    for n, s in (("V4", ["V4"]), ("locked", L), ("nocrop", N)): cf[n].append(np.r_[fit_apply(Fm, ym, Fc, s), fit_apply(Fc, yc, Fm, s)])
    new = dict(zip(it, fit_apply(F, yf, Ft, L) >= 0))
    rows = json.load(open(V4 / f"fold_{k}/local_real_noaux/test_predictions.json"))["records"]
    bypos = {r["image_id"]: r["pred"] == "yes" for r in rows if r["question"] == CAN}
    for r in rows:
        tot += 1; c = na(r["pred"]) == na(r["gt"]); cor_base += c
        if r["question"] == CAN:
            d = new[r["image_id"]]; cor_raw += d == (r["gt"] == "yes"); flips += d != (r["pred"] == "yes")
            up += d and r["pred"] != "yes"; dn += (not d) and r["pred"] == "yes"
        else:
            cor_raw += c
            if r["answer_type"] == "OPEN" and r["question"] == "Describe the finding.":
                said_none = na(r["pred"]).startswith("no fracture"); opens += 1
                contra_base += said_none == bypos[r["image_id"]]; contra_new += said_none == bool(new[r["image_id"]])
Yv = np.concatenate(Yv); out = dict(QA_V4_raw=cor_base / tot, QA_locked_raw=cor_raw / tot, n=tot, canonical_flips=int(flips), no_to_yes=int(up), yes_to_no=int(dn),
    open_describe=opens, contradict_V4=contra_base, contradict_locked=contra_new, nontest_AP={n: ap(Yv, np.concatenate(v)) for n, v in cf.items()})
print(json.dumps(out, indent=1)); json.dump(out, open(OUT / "qa.json", "w"), indent=1)
