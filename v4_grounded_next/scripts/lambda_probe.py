"""Gradient-scale probe for lambda_2 (L_contrast) and lambda_1 (L_ground).

On the first ``--updates`` optimizer updates of the real fold-0 V4 epoch-1
schedule (4 microbatches each, cross-fitted proposals, V4-GC model at PREtask
init), measure for every loss term t in {task, ground, contrast} the
per-update gradient g_t = mean over the 4 microbatches of d t / d theta,
restricted to each parameter group.  Reports L2 norms, the ratio
||g_contrast|| / ||g_task|| and ||g_ground|| / ||g_task|| on the shared
encoder groups, the cosine between g_task and each auxiliary gradient, and the
pre-clip total norm of the task gradient (to see whether clip=1.0 binds).

The lambda grid is then chosen so lambda * ||g_aux|| / ||g_task|| on the visual
LoRA spans {0.1, 0.3, 1.0} (median over updates).  No model_val information is
used here.

    source v4_grounded_next/env.sh && $PY -m scripts.lambda_probe --updates 12 --device mps
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vg.paths import RUNS, dump_json
from vg.trainer import FoldData, build_model, param_groups, v4_schedule

SHARED = ("visual_lora", "local_tokens", "fusion", "llm_lora", "vqa_heads")


def main(a):
    torch.set_num_threads(a.threads)
    t0 = time.time()
    dev = a.device
    model, v4cfg = build_model("V4-GC", a.fold, device=dev,
                               overrides={"vg_lambda_ground": 1.0, "vg_lambda_contrast": 1.0})
    data = FoldData(a.fold, "crossfit")
    groups = param_groups(model)
    named = dict(model.named_parameters())
    sched = v4_schedule(a.fold)[0]["ordered_indices"]
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    from src.train import seed_training_epoch
    seed_training_epoch(42, 0)
    model.train()

    terms = ("task", "ground", "contrast")
    per_update, diag = [], []
    mb = 0
    for u in range(a.updates):
        acc = {t: {g: None for g in groups} for t in terms}
        present = {t: 0 for t in terms}
        for _ in range(4):
            idxs = sched[mb]; mb += 1
            b = data.train_batch(idxs, model.cfg)
            with private_training_loss(model, b, 3, "token_mean", data.table):
                out = model(b, stage=3)
            T = out["terms"]
            n_pos = sum(1 for c in b["contrast_crops"] if c)
            diag.append({"mb": mb - 1, "task": float(T["task"]), "ground": float(T.get("ground", torch.nan)),
                         "contrast": float(T.get("contrast", torch.nan)), "rows_with_crops": n_pos})
            live = [t for t in terms if t in T and T[t].requires_grad]
            order = [n for names in groups.values() for n in names]
            for k, t in enumerate(live):
                gr = dict(zip(order, torch.autograd.grad(T[t], [named[n] for n in order],
                                                          retain_graph=k < len(live) - 1, allow_unused=True)))
                for g, names in groups.items():
                    flat = torch.cat([(gr[n] if gr[n] is not None else torch.zeros_like(named[n])).detach()
                                      .reshape(-1).cpu() for n in names]) / 4.0
                    acc[t][g] = flat if acc[t][g] is None else acc[t][g] + flat
                present[t] += 1
                del gr
            del out, T
        rec = {"update": u, "microbatches_with_term": present}
        for t in terms:
            rec[t] = {g: (float(v.norm()) if v is not None else 0.0) for g, v in acc[t].items()}
        tot = torch.cat([acc["task"][g] for g in groups if acc["task"][g] is not None])
        rec["task_total_norm"] = float(tot.norm())
        rec["cos"] = {}
        for aux in ("ground", "contrast"):
            rec["cos"][aux] = {}
            for g in SHARED:
                if g in groups and acc[aux][g] is not None and acc["task"][g] is not None:
                    x, y = acc["task"][g], acc[aux][g]
                    den = float(x.norm() * y.norm())
                    rec["cos"][aux][g] = float(x @ y) / den if den > 0 else None
        per_update.append(rec)
        print(json.dumps({"u": u, "task_total": round(rec["task_total_norm"], 4),
                          **{f"{t}_vlora": round(rec[t].get("visual_lora", 0.0), 5) for t in terms},
                          "sec": round(time.time() - t0)}), flush=True)

    summary = {}
    for g in groups:
        s = {}
        for t in terms:
            v = np.array([r[t][g] for r in per_update])
            s[t] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
        for aux in ("ground", "contrast"):
            ratio = np.array([r[aux][g] / r["task"][g] for r in per_update if r["task"][g] > 0 and r[aux][g] > 0])
            s[f"{aux}_over_task"] = ({"median": float(np.median(ratio)), "min": float(ratio.min()),
                                      "max": float(ratio.max())} if len(ratio) else None)
        summary[g] = s
    tot = np.array([r["task_total_norm"] for r in per_update])
    rec = {}
    for aux in ("contrast", "ground"):
        r = (summary.get("visual_lora", {}).get(f"{aux}_over_task") or {}).get("median")
        rec[aux] = {"median_ratio_visual_lora": r,
                    "lambda_for_target_ratio": {str(q): (q / r if r else None) for q in (0.1, 0.3, 1.0)}}
    report = {"fold": a.fold, "arm": "V4-GC", "device": dev, "updates": a.updates, "microbatches": mb,
              "proposals_sha256": data.proposals_sha256, "groups": {g: len(v) for g, v in groups.items()},
              "task_total_norm_pre_clip": {"median": float(np.median(tot)), "frac_above_clip_1": float((tot > 1).mean())},
              "summary": summary, "lambda_recommendation": rec, "per_update": per_update, "microbatch_losses": diag,
              "seconds": time.time() - t0}
    out = RUNS / "lambda" / f"probe_fold{a.fold}_{dev}.json"
    dump_json(out, report)
    print(json.dumps({"task_total_norm_pre_clip": report["task_total_norm_pre_clip"],
                      "lambda_recommendation": rec,
                      "ratios": {g: {k: v for k, v in s.items() if k.endswith("over_task")} for g, s in summary.items()}},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--updates", type=int, default=12)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    main(ap.parse_args())
