"""Leave-one-out, cumulative ablation and question-type breakdown from saved predictions only (v4 5-fold, v5 fold 0)."""
import json, re
from pathlib import Path
import numpy as np

R2 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2")
V5 = R2 / "reports/v5_studies/factorial_fold0/fold_0"
V4 = R2 / "reports/local_noaux_refinement_v1"
CAN = "Is there a fracture in this X-ray?"


def family(q):
    if q == CAN: return "fracture_presence"
    if re.match(r"Is the .* fractured\?", q): return "region_fracture"
    if q.startswith("Is this an X-ray of"): return "region_polar"
    if "view" in q: return "view_polar"
    if "hardware" in q: return "hardware"
    if "more than one scan" in q: return "multi_scan"
    if "multiple fractures" in q: return "multiple_fractures"
    if "How many" in q: return "count_open"
    if "body region" in q: return "region_open"
    if "Where" in q: return "location_open"
    if "Describe" in q: return "describe_open"
    if "abnormality" in q: return "abnormality_open"
    if "imaging" in q: return "modality_open"
    return "other"


def norm(a): return re.sub(r"\s+", " ", str(a).strip().lower().rstrip("."))
def load(p): return json.load(open(p))["records"]


def table(ref, arm, seed=20261101, n=2000):
    """Paired image bootstrap of accuracy difference, per question family."""
    rng = np.random.default_rng(seed)
    assert [r["question"] for r in ref] == [r["question"] for r in arm]
    fam = np.array([family(r["question"]) for r in ref])
    ids = np.array([r["image_id"] for r in ref])
    a = np.array([norm(r["pred"]) == norm(r["gt"]) for r in ref], float)
    b = np.array([norm(r["pred"]) == norm(r["gt"]) for r in arm], float)
    out = {}
    for f in sorted(set(fam)):
        s = fam == f
        un, inv = np.unique(ids[s], return_inverse=True)
        sa, sb, cnt = np.bincount(inv, a[s], len(un)), np.bincount(inv, b[s], len(un)), np.bincount(inv, minlength=len(un))
        d = []
        for _ in range(n):
            w = rng.integers(0, len(un), len(un))
            d.append(sb[w].sum() / cnt[w].sum() - sa[w].sum() / cnt[w].sum())
        out[f] = dict(n=int(s.sum()), base=float(a[s].mean()), arm=float(b[s].mean()), diff=float(b[s].mean() - a[s].mean()),
                      ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], gained=int(((b - a)[s] > 0).sum()), lost=int(((b - a)[s] < 0).sum()),
                      pred_changed=int(sum(ref[i]["pred"] != arm[i]["pred"] for i in np.nonzero(s)[0])))
    return out


res = {}
for arm in ("T", "P", "PT", "V4", "V", "L", "VLPT"):
    for role in ("model_val", "test"):
        res[f"v5_fold0_{arm}_vs_B_{role}"] = table(load(V5 / f"B_seed42/{role}_predictions.json"), load(V5 / f"{arm}_seed42/{role}_predictions.json"))
A, B = [], []
for k in range(5):
    B += load(V4 / f"fold_{k}/B/test_predictions.json")
    A += load(V4 / f"fold_{k}/local_real_noaux/test_predictions.json")
res["v4_5fold_real_vs_B_test"] = table(B, A)
res["v4_fold0_placebo_vs_B_test"] = table(load(V4 / "fold_0/B/test_predictions.json"), load(V4 / "fold_0/local_placebo_noaux/test_predictions.json"))
res["v4_fold0_real_vs_placebo_test"] = table(load(V4 / "fold_0/local_placebo_noaux/test_predictions.json"), load(V4 / "fold_0/local_real_noaux/test_predictions.json"))
json.dump(res, open(R2 / "reports/route_audit_v1/question_type.json", "w"), indent=1)
for k in ("v4_5fold_real_vs_B_test", "v5_fold0_V4_vs_B_test", "v4_fold0_real_vs_placebo_test", "v5_fold0_T_vs_B_test", "v5_fold0_P_vs_B_test", "v5_fold0_L_vs_B_test", "v5_fold0_V_vs_B_test", "v5_fold0_VLPT_vs_B_test"):
    print("\n" + k)
    for f, v in res[k].items():
        print(f"{f:20s} n={v['n']:5d} base={v['base']:.3f} arm={v['arm']:.3f} d={v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] +{v['gained']}/-{v['lost']} chg={v['pred_changed']}")
