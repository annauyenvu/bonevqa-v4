"""Apply the pre-registered lambda_2 rule (runs/lambda/sweep_plan_v2.json) to the
fold-0 model_val results of the sweep and write runs/lambda/selection.json."""
from __future__ import annotations

import json

from vg.paths import RUNS, dump_json, read_json


def main():
    plan = read_json(RUNS / "lambda/sweep_plan_v2.json")
    res = {}
    for r in plan["runs"]:
        c = read_json(RUNS / "arms" / r["tag"] / "fold_0/seed_42/completed.json")
        res[r["tag"]] = {"arm": r["arm"], "lambda_contrast": r.get("lambda_contrast"), **c["model_val"]}
    ref = res["fold0/V4-G"]["auprc"]
    cands = [(t, v) for t, v in res.items() if v["arm"] == "V4-GC"]
    ok = [(t, v) for t, v in cands if v["auprc"] >= ref - 0.01]
    if ok:
        best = max(v["energy_in_gt"] for _, v in ok)
        tied = sorted([(v["lambda_contrast"], t) for t, v in ok if v["energy_in_gt"] >= best - 0.005])
        lam, tag, why = tied[0][0], tied[0][1], "rule"
    else:
        lam, tag = sorted((v["lambda_contrast"], t) for t, v in cands)[0]; why = "no candidate non-inferior; smallest"
    out = {"selected_lambda_contrast": lam, "selected_tag": tag, "reason": why, "v4g_auprc": ref, "results": res}
    dump_json(RUNS / "lambda/selection.json", out)
    print(lam)


if __name__ == "__main__":
    main()
