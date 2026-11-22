"""Score one checkpoint on the canonical fracture rows of a role, with a chosen proposal file.

    source v4_grounded_next/env.sh
    $PY -m scripts.eval_arm --name A_v4_sealed_frcnn --sealed-v4 --role test
    $PY -m scripts.eval_arm --name E_v4_sealed_yolo --sealed-v4 --role test --proposals-file <yolo proposals>
    $PY -m scripts.eval_arm --name D_v4cf_yolo --tag yolo_fold0/V4-CF_yolo --role test --proposals-file <yolo proposals>

Inference only, CPU.  Rows (image_id, y, m, p_final) are written so that arms can be
compared with a paired component bootstrap afterwards.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from scripts.mechanism_tests import load_arm, load_sealed_v4
from vg.paths import CANONICAL_Q, RUNS, V4_REPORT, dump_json, read_json
from vg.trainer import FoldData, evaluate_canonical


def main(a):
    torch.set_num_threads(a.threads)
    out = RUNS / "yolo_ablation" / f"fold_{a.fold}" / f"{a.name}_{a.role}.json"
    assert not out.exists(), out
    if a.sealed_v4:
        model, info, ck_sha = load_sealed_v4(a.fold, "cpu")
    else:
        model, info, ck_sha = load_arm(a.tag, a.fold, "cpu")
    # Held-out roles are the same under either TRAIN key; "insample_v4" is the key every proposal file has.
    data = FoldData(a.fold, "insample_v4", a.proposals_file)
    t0 = time.time()
    metrics, rows = evaluate_canonical(model, data, a.role, limit=a.limit)
    rep = {"name": a.name, "role": a.role, "fold": a.fold, "checkpoint_sha256": ck_sha, "arm": info.get("arm"),
           "tag": a.tag, "sealed_v4": a.sealed_v4, "proposals_file": a.proposals_file,
           "proposals_sha256": data.proposals_sha256, "metrics": metrics, "seconds": time.time() - t0}
    if a.sealed_v4 and a.proposals_file is None and a.role in ("model_val", "test") and a.limit is None:
        f = V4_REPORT / f"fold_{a.fold}/local_real_noaux/{a.role}_predictions.json"
        sealed = {r["image_id"]: r["p_yes"] for r in read_json(f)["records"] if r["question"] == CANONICAL_Q}
        d = [abs(r["p_final"] - sealed[r["image_id"]]) for r in rows if r["image_id"] in sealed]
        rep["parity_with_sealed_predictions"] = {"n": len(d), "max_abs_dp": max(d) if d else None}
    dump_json(out, {**rep, "rows": [{k: r[k] for k in ("image_id", "y", "m", "p_final")} for r in rows]}, exclusive=True)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sealed-v4", action="store_true")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="test")
    ap.add_argument("--proposals-file", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    assert a.sealed_v4 != bool(a.tag), "give exactly one of --sealed-v4 / --tag"
    main(a)
