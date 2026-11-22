"""Paired comparison of checkpoints on the BTXRD endpoints (outputs of scripts/eval_joint.py).

    source v4_grounded_next/env.sh
    $PY -m scripts.compare_btxrd --role test --out compare_btxrd_test_kg --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        J_v4cf_yolojoint_btxrd K_v4cf_kg K0_v4cf_kgempty

The first name is the reference.  Rows are paired by image; the bootstrap resamples pseudo-patient
groups (centre, group), 2,000 times, seed 20261114, like the FracAtlas component bootstrap.
tumor / malignancy: AUPRC and AUROC of P(yes).  diagnosis: exact-match accuracy and balanced accuracy.
Inference outputs only; no model is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from vg.joint import BT_ROLE
from vg.paths import RUNS, dump_json, read_json

N_BOOT, SEED = 2000, 20261114


def binary(rows_by_arm, idx):
    out = {}
    for name, rows in rows_by_arm.items():
        y = np.array([rows[i]["y"] for i in idx])
        p = np.array([rows[i]["p_yes"] for i in idx])
        if not 0 < y.sum() < len(y):
            return None
        out[name] = {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p))}
    return out


def diagnosis(rows_by_arm, idx):
    out = {}
    for name, rows in rows_by_arm.items():
        gt = np.array([rows[i]["gt"] for i in idx])
        ok = np.array([rows[i]["correct"] for i in idx], dtype=float)
        out[name] = {"accuracy": float(ok.mean()),
                     "balanced_accuracy": float(np.mean([ok[gt == c].mean() for c in sorted(set(gt))]))}
    return out


def main(a):
    folder = Path(a.dir) if a.dir else RUNS / "joint" / f"fold_{a.fold}"
    out = folder / f"{a.out}.json"
    assert not out.exists(), out
    group = {r["image_id"]: (r["center"], r["group"]) for r in read_json(Path(a.btxrd_vqa) / f"vqa_{BT_ROLE[a.role]}.json")}
    reports = {n: read_json(folder / f"{n}_btxrd_{a.role}.json") for n in a.names}
    ref = a.names[0]
    result = {"role": a.role, "reference": ref, "n_boot": N_BOOT, "seed": SEED,
              "checkpoints": {n: r["checkpoint_sha256"] for n, r in reports.items()}, "endpoints": {}}
    rng = np.random.default_rng(SEED)
    for ep, metric_fn, truth in (("tumor", binary, "y"), ("malignancy", binary, "y"), ("diagnosis", diagnosis, "gt")):
        rows = {n: r["rows"][ep] for n, r in reports.items()}
        ids = [x["image_id"] for x in rows[ref]]
        assert len(set(ids)) == len(ids), f"{ep}: an image appears twice"
        for n in a.names:   # same images, same order, same labels
            assert [x["image_id"] for x in rows[n]] == ids and [x[truth] for x in rows[n]] == [x[truth] for x in rows[ref]], (ep, n)
        point = metric_fn(rows, range(len(ids)))
        assert point is not None, f"{ep}: one class only"
        members = {}
        for i, iid in enumerate(ids):
            members.setdefault(group[iid], []).append(i)
        groups = list(members.values())
        diffs = {n: {k: [] for k in point[ref]} for n in a.names[1:]}
        for _ in range(N_BOOT):
            idx = [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]
            m = metric_fn(rows, idx)
            if m is None:
                continue
            for n in a.names[1:]:
                for k in m[ref]:
                    diffs[n][k].append(m[n][k] - m[ref][k])
        result["endpoints"][ep] = {
            "n": len(ids), "groups": len(groups), "metrics": point,
            "diff_vs_reference": {n: {k: {"diff": point[n][k] - point[ref][k],
                                          "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                          "resamples": len(v)} for k, v in d.items()} for n, d in diffs.items()}}
        for n in a.names:
            line = "  ".join(f"{k} {v:.3f}" for k, v in point[n].items())
            extra = ""
            if n != ref:
                d = result["endpoints"][ep]["diff_vs_reference"][n]
                extra = "  | " + "  ".join(f"d{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in d.items())
            print(f"{ep:11s} n={len(ids):4d}  {n:30s} {line}{extra}", flush=True)
    dump_json(out, result, exclusive=True)
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="evaluation names; the first is the reference")
    ap.add_argument("--role", default="test", choices=["model_val", "test", "external"])
    ap.add_argument("--fold", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--btxrd-vqa", required=True)
    ap.add_argument("--dir", default=None, help="folder with the evaluation files (default runs/joint/fold_<fold>)")
    main(ap.parse_args())
