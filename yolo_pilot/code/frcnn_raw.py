"""Raw (score >= 0.001, no top-3 cap) predictions of the sealed Faster R-CNN detectors
on fold-0 val / calibration / test.  Run with the sealed interpreter:

    source v4_grounded_next/env.sh && $PY yolo_pilot/code/frcnn_raw.py

Reads refactor_v2 and v4_grounded_next read-only; writes only under yolo_pilot/runs.
"""
import json
import sys
import time
from pathlib import Path

import torch

from vg.detector import load_detector, open_canonical, predict_raw
from vg.paths import V4_REPORT, image_table, manifest_fold, read_json, sha256

OUT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot/runs/frcnn")
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # fold 0 also ran the ep6 detector; other folds only ep2_sealed
torch.set_num_threads(8)


def run(tag, ckpt):
    dest = OUT / f"fold{FOLD}_{tag}.json"
    if dest.exists():
        print("exists", dest)
        return
    model, _ = load_detector(ckpt)
    model.eval()
    roles = manifest_fold(FOLD)
    out, start = {}, time.time()
    for role in ("model_val", "calibration", "test"):
        table = image_table(read_json(roles[role]))
        for n, iid in enumerate(sorted(table)):
            boxes, _ = predict_raw(model, open_canonical(table[iid]["image"]))
            out[iid] = boxes
            if n % 200 == 0:
                print(tag, role, n, len(table), round(time.time() - start), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"checkpoint": str(ckpt), "sha256": sha256(ckpt), "floor": 0.001, "boxes": out}))
    print("wrote", dest, round(time.time() - start), "s")


if __name__ == "__main__":
    receipt = read_json(V4_REPORT / f"fold_{FOLD}/detector/receipt.json")
    assert sha256(receipt["checkpoint"]) == receipt["sha256"]
    run("ep2_sealed", receipt["checkpoint"])
    if FOLD != 0:
        sys.exit(0)
    ep6 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/v4_detector_ext_v1") / f"fold_{FOLD}/ep6.pt"
    try:
        run("ep6", ep6)
    except Exception as e:  # ep6 checkpoints may use a different payload layout
        print("ep6 failed:", repr(e), file=sys.stderr)
