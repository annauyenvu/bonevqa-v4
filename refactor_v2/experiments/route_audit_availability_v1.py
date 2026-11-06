"""Does V4's fracture gain reduce to a late fusion of B with the detector's 'any proposal' flag?  Saved predictions only.
Stacker fitted on the fold's calibration+model_val canonical rows, read on test; 5 folds pooled the way the v4 study pools."""
import json
from pathlib import Path
import numpy as np

R2 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2")
V4 = R2 / "reports/local_noaux_refinement_v1"
CAN = "Is there a fracture in this X-ray?"


def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]; tp = np.cumsum(y)
    return float((tp[y == 1] / (np.nonzero(y == 1)[0] + 1)).mean())


def can(path):
    return [r for r in json.load(open(path))["records"] if r["question"] == CAN]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit(x, y, steps=400):
    w = np.zeros(x.shape[1]); X = x
    for _ in range(steps):  # Newton, plain logistic regression with a small ridge
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + 1e-4 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def feats(rows, boxes, mode):
    b = logit(np.array([r["p_yes"] for r in rows]))
    a = np.array([bool(boxes.get(r["image_id"])) for r in rows], float)
    n = np.array([len(boxes.get(r["image_id"]) or []) for r in rows], float)
    cols = [np.ones(len(rows)), b]
    if mode == "flag": cols.append(a)
    if mode == "flag_inter": cols += [a, a * b]
    if mode == "count": cols += [a, n]
    return np.stack(cols, 1)


out, pool = {}, {k: [] for k in ("y", "B", "V4", "flag", "flag_inter", "count", "flag_only_rank")}
for k in range(5):
    f = V4 / f"fold_{k}"
    boxes = json.load(open(f / "boxes.json"))
    fitrows = can(f / "B/calibration_predictions.json") + can(f / "B/model_val_predictions.json")
    test, v = can(f / "B/test_predictions.json"), can(f / "local_real_noaux/test_predictions.json")
    y_fit = np.array([r["gt"] == "yes" for r in fitrows], float)
    y = np.array([r["gt"] == "yes" for r in test], float)
    assert [r["image_id"] for r in test] == [r["image_id"] for r in v]
    row = dict(n=len(y), pos=int(y.sum()), AP_B=ap(y, np.array([r["p_yes"] for r in test])), AP_V4=ap(y, np.array([r["p_yes"] for r in v])))
    pool["y"].append(y); pool["B"].append(np.array([r["p_yes"] for r in test])); pool["V4"].append(np.array([r["p_yes"] for r in v]))
    a = np.array([bool(boxes.get(r["image_id"])) for r in test], bool)
    row["active_pos"], row["active_neg"] = float(a[y == 1].mean()), float(a[y == 0].mean())
    for m in ("flag", "flag_inter", "count"):
        w = fit(feats(fitrows, boxes, m), y_fit)
        s = 1 / (1 + np.exp(-feats(test, boxes, m) @ w))
        row[f"AP_B_plus_{m}"] = ap(y, s); row[f"w_{m}"] = [float(x) for x in w]
        pool[m].append(s)
    out[f"fold_{k}"] = row
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items() if not a.startswith("w_")}, flush=True)
Y = np.concatenate(pool["y"]); P = {k: np.concatenate(v) for k, v in pool.items() if k not in ("y", "flag_only_rank") and v}
rng = np.random.default_rng(20261101)
draws = [rng.integers(0, len(Y), len(Y)) for _ in range(2000)]
def ci(a, b):
    d = [ap(Y[s], a[s]) - ap(Y[s], b[s]) for s in draws]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
# fold-mean AP is what the study reports next to pooled AP; both are printed
out["pooled"] = {f"AP_{k}": ap(Y, v) for k, v in P.items()}
out["pooled"]["V4_minus_B"] = ci(P["V4"], P["B"])
for m in ("flag", "flag_inter", "count"):
    out["pooled"][f"B_plus_{m}_minus_B"] = ci(P[m], P["B"])
    out["pooled"][f"V4_minus_B_plus_{m}"] = ci(P["V4"], P[m])
print(json.dumps(out["pooled"], indent=1))
# fold-0 placebo has the same availability
f = V4 / "fold_0"; pl = can(f / "local_placebo_noaux/test_predictions.json"); B0 = can(f / "B/test_predictions.json"); re0 = can(f / "local_real_noaux/test_predictions.json")
y = np.array([r["gt"] == "yes" for r in pl], float)
out["fold0_placebo"] = dict(AP_B=ap(y, np.array([r["p_yes"] for r in B0])), AP_placebo=ap(y, np.array([r["p_yes"] for r in pl])), AP_real=ap(y, np.array([r["p_yes"] for r in re0])), AP_B_flag=out["fold_0"]["AP_B_plus_flag"])
print(out["fold0_placebo"])
json.dump(out, open(R2 / "reports/route_audit_v1/availability.json", "w"), indent=1)
