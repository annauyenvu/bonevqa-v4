"""Score a checkpoint on the BTXRD endpoints (tumor, malignancy, diagnosis).  Inference only, CPU.

Works for a joint checkpoint and for a FracAtlas-only one (the reference "never saw BTXRD"):

    source v4_grounded_next/env.sh
    $PY -m scripts.eval_joint --name J_v4_joint_insample --tag joint_fold0/V4_joint_insample --role test \
        --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0_insample.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json

The FracAtlas fracture endpoint of the same checkpoint is scored by scripts/eval_arm.py, so its
output can go straight into scripts/compare_arms.py.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from scripts.mechanism_tests import load_arm
from vg.joint import JointFoldData, evaluate_btxrd
from vg.paths import RUNS, dump_json


def main(a):
    torch.set_num_threads(a.threads)
    out = RUNS / "joint" / f"fold_{a.fold}" / f"{a.name}_btxrd_{a.role}.json"
    assert not out.exists(), out
    model, info, ck_sha = load_arm(a.tag, a.fold, "cpu")
    data = JointFoldData(a.fold, "insample_v4", a.proposals_file, a.btxrd_vqa, a.btxrd_proposals)
    t0 = time.time()
    metrics, rows = evaluate_btxrd(model, data, a.role, limit=a.limit)
    rep = {"name": a.name, "role": a.role, "fold": a.fold, "tag": a.tag, "arm": info.get("arm"), "joint_training": bool(info.get("joint")),
           "checkpoint_sha256": ck_sha, "btxrd_vqa_sha256": data.btxrd_vqa_sha256[a.role],
           "btxrd_proposals_sha256": data.btxrd_proposals_sha256, "metrics": metrics, "seconds": time.time() - t0}
    dump_json(out, {**rep, "rows": rows}, exclusive=True)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="test", choices=["model_val", "test", "external"])
    ap.add_argument("--proposals-file", required=True)
    ap.add_argument("--btxrd-vqa", required=True)
    ap.add_argument("--btxrd-proposals", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args())
