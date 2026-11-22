"""How far does each arm's fracture probability move from V4 at PREtask init?

An auxiliary module that changes the forward function before any training
starts the arm from a different point than V4, which would confound the
ablation.  Reports max |delta p| vs the V4 arm and model_val AUROC at init.

    $PY -m scripts.init_shift_check --n 96 --device mps
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from vg.paths import RUNS, dump_json
from vg.trainer import FoldData, build_model, evaluate_canonical

VARIANTS = [("V4", {}), ("V4-G", {"vg_pool_init": "legacy"}), ("V4-G", {}), ("V4-GC", {}),
            ("V4-GCR", {}), ("V4-GCR-flag", {})]


def main(a):
    torch.set_num_threads(4)
    data = FoldData(a.fold, "crossfit")
    ref, out = None, []
    for arm, ov in VARIANTS:
        model, _ = build_model(arm, a.fold, device=a.device, overrides=ov)
        m, rows = evaluate_canonical(model, data, "model_val", limit=a.n)
        p = np.array([r["p_final"] for r in rows])
        if ref is None:
            ref = p
        rec = {"arm": arm, "overrides": ov, "max_abs_dp_vs_V4": float(np.abs(p - ref).max()),
               "mean_p": float(p.mean()), "auroc": m.get("auroc"), "spec@0.5": m.get("spec@0.5")}
        out.append(rec); print(json.dumps(rec), flush=True)
        del model
        if a.device == "mps":
            torch.mps.empty_cache()
    dump_json(RUNS / "lambda" / f"init_shift_fold{a.fold}.json", {"n": a.n, "device": a.device, "variants": out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--device", default="mps")
    main(ap.parse_args())
