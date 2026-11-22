"""Compare eval_arm outputs on the same canonical rows: AUPRC / AUROC, operating point at 0.5,
and paired component-bootstrap differences against the first arm.

    source v4_grounded_next/env.sh
    $PY -m scripts.compare_arms --role test --out compare_test A_v4_sealed_frcnn E_v4_sealed_yolo ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from vg.paths import RUNS, component_lookup, dump_json, read_json

N_BOOT, SEED = 2000, 20261114


def point(y, p, m):
    pos, neg = y == 1, y == 0
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "sens@0.5": float((p[pos] >= .5).mean()), "spec@0.5": float((p[neg] < .5).mean()),
            "m_rate_fractured": float(m[pos].mean()), "m_rate_nonfractured": float(m[neg].mean()),
            "sens@0.5_m1": float((p[pos & m] >= .5).mean()) if (pos & m).any() else None,
            "sens@0.5_m0": float((p[pos & ~m] >= .5).mean()) if (pos & ~m).any() else None,
            "n_pos_m0": int((pos & ~m).sum())}


def main(a):
    d = RUNS / "yolo_ablation" / f"fold_{a.fold}"
    arms = {n: read_json(d / f"{n}_{a.role}.json") for n in a.names}
    ids = [r["image_id"] for r in arms[a.names[0]]["rows"]]
    y = np.array([r["y"] for r in arms[a.names[0]]["rows"]])
    P, M = {}, {}
    for n, rep in arms.items():
        rows = {r["image_id"]: r for r in rep["rows"]}
        assert sorted(rows) == sorted(ids) and all(rows[i]["y"] == yy for i, yy in zip(ids, y)), n
        P[n] = np.array([rows[i]["p_final"] for i in ids])
        M[n] = np.array([bool(rows[i]["m"]) for i in ids])
    lookup = component_lookup()
    comp = np.array([lookup[i] for i in ids])
    uniq = np.unique(comp)
    members = [np.where(comp == c)[0] for c in uniq]
    rng = np.random.default_rng(SEED)
    ref = a.names[0]
    diffs = {n: {"auprc": [], "auroc": []} for n in a.names[1:]}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        base = (average_precision_score(yy, P[ref][idx]), roc_auc_score(yy, P[ref][idx]))
        for n in a.names[1:]:
            diffs[n]["auprc"].append(average_precision_score(yy, P[n][idx]) - base[0])
            diffs[n]["auroc"].append(roc_auc_score(yy, P[n][idx]) - base[1])
    out = {"role": a.role, "fold": a.fold, "n": len(ids), "fractured": int(y.sum()), "components": len(uniq),
           "reference": ref, "bootstrap": {"resamples": N_BOOT, "seed": SEED, "unit": "component"},
           "arms": {n: {**point(y, P[n], M[n]), "proposals_file": arms[n]["proposals_file"],
                        "checkpoint_sha256": arms[n]["checkpoint_sha256"]} for n in a.names},
           "diff_vs_reference": {n: {k: {"diff": float(point(y, P[n], M[n])[k] - point(y, P[ref], M[ref])[k]),
                                         "ci95": np.quantile(v, [.025, .975]).round(4).tolist()}
                                     for k, v in dv.items()} for n, dv in diffs.items()}}
    dump_json(d / f"{a.out}.json", out)
    for n in a.names:
        r = out["arms"][n]
        print(f"{n:28s} AUPRC {r['auprc']:.3f}  AUROC {r['auroc']:.3f}  sens {r['sens@0.5']:.3f}  spec {r['spec@0.5']:.3f}  "
              f"m+ {r['m_rate_fractured']:.3f}  m- {r['m_rate_nonfractured']:.3f}")
    print(json.dumps(out["diff_vs_reference"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="test")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
