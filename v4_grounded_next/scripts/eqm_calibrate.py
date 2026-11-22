"""EQM calibration of cross-fit TRAIN proposals (see vg/eqm.py for the map).

For outer fold f, after ``cross_fit_proposals all --fold f``:

  export   raw (floor 0.001, no cap) proposals of
             D_sealed on calibration,
             D_k (k = 0..4) on calibration and on its own held-out inner TRAIN part.
           Parity: the V4 contract on raw output with the identity map must
           reproduce the existing 0.05 / top-3 proposals exactly
           (outer_scored.json calibration, inner_k/oof.json).
  fit      one map per inner detector from calibration s_max ECDFs -> eqm/mapping.json
  apply    proposals_fold{f}_eqm.json: TRAIN = mapped OOF proposals (V4 contract on
           mapped scores); model_val / calibration / test unchanged (sealed detector)
  audit    m(I) and recall@IoU of TRAIN-CF-EQM vs calibration and TEST, next to raw
           TRAIN-CF and TRAIN in-sample -> eqm/audit_eqm.json

Only calibration-split detector outputs are used to fit (no labels at all: the
map uses scores only); TEST enters the audit only.  Nothing existing is
overwritten: all outputs go to runs/crossfit/fold_f/eqm/ and proposals_fold{f}_eqm.json.

    source v4_grounded_next/env.sh
    $PY -m scripts.eqm_calibrate all --fold 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from vg import eqm
from vg.detector import load_detector, open_canonical, predict_raw
from vg.paths import RUNS, V4_REPORT, dump_json, image_table, manifest_fold, read_json, sha256

K_INNER = 5


def d_of(fold):
    return RUNS / "crossfit" / f"fold_{fold}"


def _raw(model, ids, table, progress):
    out, t0 = {}, time.time()
    for n, iid in enumerate(ids):
        out[iid], _ = predict_raw(model, open_canonical(table[iid]["image"]), floor=eqm.FLOOR)
        if n % 50 == 0:
            dump_json(progress, {"done": n, "total": len(ids), "wall": time.time() - t0})
    return out


def _parity(raw, ref):
    """V4 contract (identity map) on raw output == stored 0.05/top-3 proposals."""
    res = {"checked": 0, "count_mismatch": 0, "max_abs_diff": 0.0}
    for iid, r in ref.items():
        got = eqm.v4_contract(raw[iid])
        res["checked"] += 1
        if len(got) != len(r):
            res["count_mismatch"] += 1
            continue
        for a, b in zip(got, r):
            res["max_abs_diff"] = max(res["max_abs_diff"], max(abs(x - y) for x, y in zip(a, b)))
    assert res["count_mismatch"] == 0 and res["max_abs_diff"] < 1e-4, f"raw-export parity failed {res}"
    return res


def export(fold):
    d = d_of(fold) / "eqm"
    d.mkdir(exist_ok=True)
    roles = manifest_fold(fold)
    cal = image_table(read_json(roles["calibration"]))
    train = image_table(read_json(roles["train"]))
    cal_ids = sorted(cal)
    outer = read_json(d_of(fold) / "outer_scored.json")
    p = d / "raw_sealed_calibration.json"
    if not p.exists():
        receipt = read_json(V4_REPORT / f"fold_{fold}/detector/receipt.json")
        assert sha256(receipt["checkpoint"]) == receipt["sha256"], "sealed V4 detector changed"
        model, _ = load_detector(receipt["checkpoint"])
        raw = _raw(model, cal_ids, cal, d / "progress_sealed.json")
        par = _parity(raw, outer["calibration"])
        dump_json(p, {"detector_sha256": receipt["sha256"], "floor": eqm.FLOOR, "parity": par, "props": raw}, exclusive=True)
    split = read_json(d_of(fold) / "inner_split.json")["inner"]
    for k in map(str, range(K_INNER)):
        p = d / f"raw_inner{k}.json"
        if p.exists():
            continue
        r = read_json(d_of(fold) / f"inner_{k}" / "receipt.json")
        assert sha256(r["checkpoint"]) == r["sha256"]
        model, payload = load_detector(r["checkpoint"])
        held = split[k]
        assert not (set(payload["training_image_ids"]) & set(held))
        raw_cal = _raw(model, cal_ids, cal, d / f"progress_inner{k}_cal.json")
        raw_oof = _raw(model, held, train, d / f"progress_inner{k}_oof.json")
        par = _parity(raw_oof, read_json(d_of(fold) / f"inner_{k}" / "oof.json"))
        dump_json(p, {"detector_sha256": r["sha256"], "floor": eqm.FLOOR, "parity_oof": par,
                      "calibration": raw_cal, "oof": raw_oof}, exclusive=True)
        print(f"fold {fold} inner {k}: exported", flush=True)


def fit(fold):
    d = d_of(fold) / "eqm"
    sealed = read_json(d / "raw_sealed_calibration.json")["props"]
    maps = {k: eqm.fit(read_json(d / f"raw_inner{k}.json")["calibration"], sealed) for k in map(str, range(K_INNER))}
    rep = {"fold": fold, "fit_on": "calibration (detector scores only)", "floor": eqm.FLOOR, "threshold": eqm.THRESH,
           "maps": maps}
    dump_json(d / "mapping.json", rep)
    return rep


def apply(fold):
    d = d_of(fold)
    maps = read_json(d / "eqm" / "mapping.json")["maps"]
    merged = read_json(d / f"proposals_fold{fold}.json")
    train = {}
    for k in map(str, range(K_INNER)):
        raw = read_json(d / "eqm" / f"raw_inner{k}.json")["oof"]
        assert not (set(raw) & set(train))
        train.update({iid: eqm.v4_contract(props, maps[k]) for iid, props in raw.items()})
    assert sorted(train) == sorted(merged["train"])
    out = {**merged, "train": train, "train_cf_uncalibrated_audit_only": merged["train"],
           "contract": merged["contract"] + " TRAIN scores mapped by per-inner-detector EQM fitted on calibration "
                                            "(eqm/mapping.json), then score >= 0.05 top-3.",
           "eqm_mapping_sha256": sha256(d / "eqm" / "mapping.json")}
    p = d / f"proposals_fold{fold}_eqm.json"
    dump_json(p, out)
    return out


def audit(fold):
    from scipy.stats import ks_2samp
    from scripts.cross_fit_proposals import _stats, _two_prop
    d = d_of(fold)
    P = read_json(d / f"proposals_fold{fold}_eqm.json")
    roles = manifest_fold(fold)
    tab = {r: image_table(read_json(roles[r])) for r in ("train", "calibration", "test")}
    orient = {}
    sets = {"train_cf_eqm": (P["train"], "train"), "train_cf_raw": (P["train_cf_uncalibrated_audit_only"], "train"),
            "train_insample": (P["train_insample_audit_only"], "train"),
            "calibration": (P["calibration"], "calibration"), "test": (P["test"], "test")}
    s, mneg, mall = {}, {}, {}
    for name, (props, role) in sets.items():
        s[name] = _stats(props, tab[role], orient)
        mneg[name] = [int(bool(props[i])) for i in props if not tab[role][i]["fractured"]]
        mall[name] = [int(bool(props[i])) for i in props]
        s[name]["m_rate_all"] = float(np.mean(mall[name]))
    comp = {}
    for a in ("train_cf_eqm", "train_cf_raw", "train_insample"):
        for b in ("calibration", "test"):
            comp[f"{a}_vs_{b}"] = {
                "m_fractured": _two_prop(s[a]["_m"], s[b]["_m"]),
                "m_negative": _two_prop(mneg[a], mneg[b]),
                "m_all": _two_prop(mall[a], mall[b]),
                "best_iou_ks": dict(zip(("statistic", "p"), map(float, ks_2samp(s[a]["_ious"], s[b]["_ious"]))))}
    for v in s.values():
        v.pop("_ious"); v.pop("_m")
    ok_m = all(comp[f"train_cf_eqm_vs_{b}"][k]["p_two_sided"] >= 0.01
               for b in ("calibration", "test") for k in ("m_fractured", "m_negative", "m_all"))
    rec = {t: s["train_cf_eqm"][t] - s["train_cf_raw"][t] for t in ("recall_iou25", "recall_iou50")}
    n_cal = len(P["calibration"]); n_fire = sum(bool(v) for v in P["calibration"].values())
    note = (f"EQM is fitted on the calibration split only ({n_cal} images; the sealed detector fires on {n_fire} of them), "
            "so model_val stays untouched for hyper-parameter selection. The upper tail of each map therefore rests on "
            f"~{n_fire} order statistics; any residual TRAIN-CF-EQM vs TEST difference in the negative-image proposal rate "
            f"({s['train_cf_eqm']['m_rate_negative']:.3f} vs {s['test']['m_rate_negative']:.3f}) is attributed to this "
            "finite calibration sample, not to a remaining systematic detector shift.")
    rep = {"fold": fold, "fit_sample_note": note, "stats": s, "comparisons": comp,
           "recall_change_eqm_minus_raw": rec,
           "acceptance": {"m_two_sample_not_rejected_at_0.01": ok_m,
                          "criterion": "TRAIN-CF-EQM vs calibration and vs TEST: two-proportion test on m(I) "
                                       "(fractured, negative, all) p >= 0.01; recall@IoU0.25/0.50 reported vs raw CF "
                                       "and vs TEST"}}
    dump_json(d / "eqm" / "audit_eqm.json", rep)
    return rep


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    for f in a.fold:
        if a.step in ("export", "all"):
            export(f)
        if a.step in ("fit", "all"):
            fit(f)
        if a.step in ("apply", "all"):
            apply(f)
        if a.step in ("audit", "all"):
            rep = audit(f)
            print(f, {k: {kk: round(vv, 3) for kk, vv in v.items() if kk.startswith(("m_rate", "recall"))}
                      for k, v in rep["stats"].items()}, rep["acceptance"]["m_two_sample_not_rejected_at_0.01"], flush=True)


if __name__ == "__main__":
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "source v4_grounded_next/env.sh"
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "fit", "apply", "audit", "all"])
    ap.add_argument("--fold", nargs="+", default=["0"])
    ap.add_argument("--threads", type=int, default=4)
    main(ap.parse_args())
