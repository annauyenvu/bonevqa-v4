"""Per-inner-detector empirical quantile mapping (EQM) of detector scores.

Inner cross-fit detectors D_{f,k} and the sealed outer detector D_{f,sealed}
share the V4 recipe but differ in confidence (fold 1: sealed fires on 3.5 % of
TEST negatives, the inner detectors on ~25 % of TRAIN negatives at the same
0.05 threshold).  EQM puts every inner score on the sealed detector's scale,
fitted ONLY on the outer fold's calibration split:

    a_k[i], a_s[i]   sorted image-level s_max on calibration (raw floor 0.001;
                     an image with no box has s_max = 0) for D_k and D_sealed
    s~ = F_s^{-1}(F_k(s))   implemented as the monotone piecewise-linear map
                     through the knots (a_k[i], a_s[i]) (paired order statistics
                     of the same n calibration images), extended linearly to
                     (0, 0) below and (1, 1) above.

so tau_k = F_k^{-1}(1 - r_sealed) maps to 0.05 and quantiles above it are
interpolated into [0.05, 1].  The map is non-decreasing, so the within-image
box order is preserved.  The V4 contract (keep top-3 with score >= 0.05) is
then applied to the mapped scores.  No model_val or test data is used.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

THRESH, TOP_K, FLOOR = 0.05, 3, 0.001


def smax(props: List[list]) -> float:
    return max((p[4] for p in props), default=0.0)


def fit(inner_cal: Dict[str, list], sealed_cal: Dict[str, list]) -> dict:
    ids = sorted(sealed_cal)
    assert sorted(inner_cal) == ids, "both detectors must be scored on the same calibration images"
    a_k = np.sort([smax(inner_cal[i]) for i in ids])
    a_s = np.sort([smax(sealed_cal[i]) for i in ids])
    # Ties in a_k (e.g. several images with no box) -> one knot at the mean target.
    xs, inv = np.unique(a_k, return_inverse=True)
    ys = np.array([a_s[inv == j].mean() for j in range(len(xs))])
    ys = np.maximum.accumulate(ys)
    if xs[0] > 0:
        xs, ys = np.r_[0.0, xs], np.r_[0.0, ys]
    if xs[-1] < 1:
        xs, ys = np.r_[xs, 1.0], np.r_[ys, max(1.0, ys[-1])]
    r_sealed = float((a_s >= THRESH).mean())
    tau_k = float(np.quantile(a_k, 1 - r_sealed)) if 0 < r_sealed < 1 else None
    return {"knots_x": xs.tolist(), "knots_y": ys.tolist(), "n_calibration": len(ids),
            "r_sealed@0.05": r_sealed, "r_inner_raw@0.05": float((a_k >= THRESH).mean()), "tau_k": tau_k}


def apply(m: dict, s):
    return np.interp(s, m["knots_x"], m["knots_y"])


def v4_contract(props: List[list], m: dict = None) -> List[list]:
    """Map scores (identity when m is None), keep score >= 0.05, top-3 in score order."""
    out = []
    for p in props:  # predict_raw order = model score order; the map is monotone
        s = float(apply(m, p[4])) if m is not None else p[4]
        if s >= THRESH and len(out) < TOP_K:
            out.append([*p[:4], s])
    return out
