"""Eval-path parity: sealed trained V4 (fold f, local_real_noaux/LAST.pt) loaded into
the V4 arm of VNextGroundedModel and scored by ``evaluate_canonical`` must
reproduce the sealed per-image p_yes on model_val canonical rows.

    $PY -m scripts.eval_parity_v4 --fold 0 --device mps
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from vg.paths import RUNS, V4_REPORT, dump_json, read_json, sha256
from vg.paths import CANONICAL_Q
from vg.trainer import FoldData, build_model, evaluate_canonical


def main(a):
    torch.set_num_threads(4)
    folder = V4_REPORT / f"fold_{a.fold}/local_real_noaux"
    ck = folder / "LAST.pt"
    assert sha256(ck) == read_json(folder / "completed.json")["checkpoint_sha256"]
    model, _ = build_model("V4", a.fold, device="cpu")
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    own = model.state_dict()
    res = model.load_state_dict({k: v for k, v in payload["state"].items() if k in own}, strict=False)
    assert not res.unexpected_keys and not set(payload["state"]) - set(own)
    model.to(a.device)
    data = FoldData(a.fold, "crossfit")
    metrics, rows = evaluate_canonical(model, data, "model_val")
    sealed = [r for r in read_json(folder / "model_val_predictions.json")["records"] if r["question"] == CANONICAL_Q]
    sp = {r["image_id"]: r["p_yes"] for r in sealed}
    d = np.array([abs(r["p_final"] - sp[r["image_id"]]) for r in rows])
    rep = {"fold": a.fold, "device": a.device, "n": len(rows), "n_sealed": len(sealed),
           "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
           "decision_flips@0.5": int(sum((r["p_final"] >= .5) != (sp[r["image_id"]] >= .5) for r in rows)),
           "auprc_ours": metrics.get("auprc"),
           "auprc_sealed": read_json(folder / "model_val_predictions.json")["metrics"]["fracture_auprc"],
           "metrics": metrics}
    dump_json(RUNS / "gates" / f"eval_parity_v4_fold{a.fold}_{a.device}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "metrics"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--device", default="mps")
    main(ap.parse_args())
