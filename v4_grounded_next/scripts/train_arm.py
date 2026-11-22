"""Train one ablation arm with the exact V4 schedule and budget.

Same as sealed ``local_noaux_refinement_v1.train``: PREtask init, the sealed
per-epoch batch order (``schedule.json``), AdamW(lr 1e-4, betas .9/.98, wd .01),
5 % warm-up + cosine over the FULL V4 update count, gradient accumulation 4,
global clip 1.0, private token-mean sequence loss, ``seed_training_epoch``.

``--max-updates N`` stops after the first N updates of that same schedule (the
LR curve is not compressed), so a prefix run is bit-for-bit the beginning of
the full run.  It is used only for hyper-parameter screening on model_val.

    source v4_grounded_next/env.sh
    $PY -m scripts.train_arm --arm V4-GC --fold 0 --device mps --max-updates 360 \
        --overrides '{"vg_lambda_contrast": 0.05}' --tag lambda_sweep/lc0.05 --no-save
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from vg.paths import RUNS, dump_json, guard_write
from vg.registry import ARMS
from vg.trainer import FoldData, build_model, evaluate_canonical, lr_lambda, make_optimizer, v4_schedule


def dump_nonfinite(folder, model, epoch, j, updates, idxs, rng, terms, bad, pos_in_update):
    """First microbatch whose backward leaves a non-finite gradient: save what is
    needed to replay it exactly (weights before the optimizer step that would
    follow, the microbatch indices, the RNG state before its forward) and stop.
    Gradients of earlier microbatches in the same update were finite (checked
    after each backward), so this microbatch alone produced the non-finite value."""
    d = folder / "nonfinite_dump"
    d.mkdir(exist_ok=False)
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "rng": rng}, d / "replay.pt")
    dump_json(d / "info.json", {"epoch": epoch, "microbatch": j, "updates_done": updates,
                                "microbatch_in_update": pos_in_update, "indices": list(idxs), "terms": terms,
                                "nonfinite_grad_params": bad}, exclusive=True)
    raise FloatingPointError(f"non-finite gradient at epoch {epoch} microbatch {j} (update {updates + 1}); "
                             f"{len(bad)} params, first: {bad[:5]}; replay state in {d}")


def main(a):
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    from src.train import seed_training_epoch
    torch.set_num_threads(a.threads)
    overrides = json.loads(a.overrides) if a.overrides else {}
    folder = guard_write(RUNS / "arms" / (a.tag or a.arm) / f"fold_{a.fold}" / f"seed_{a.seed}")
    folder.mkdir(parents=True, exist_ok=False)
    model, v4cfg = build_model(a.arm, a.fold, device=a.device, overrides=overrides)
    src = a.proposals or ARMS[a.arm]["proposals"]
    assert a.proposals is None or ARMS[a.arm]["proposals"] != "insample_v4", "V4 keeps its in-sample proposals"
    data = FoldData(a.fold, src, a.proposals_file)
    schedules = v4_schedule(a.fold)
    total = sum(e["updates"] for e in schedules)
    stop = min(total, a.max_updates or total)
    params, opt = make_optimizer(model, v4cfg["train"]["lr"], a.readout_lr_mult, a.readout_scope)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total))
    dump_json(folder / "started.json", {"pid": os.getpid(), "arm": a.arm, "fold": a.fold, "seed": a.seed,
                                        "device": a.device, "overrides": overrides, "model_cfg": model.cfg,
                                        "total_updates_schedule": total, "stop_after": stop,
                                        "proposals_sha256": data.proposals_sha256, "train_source": data.source["train"],
                                        "train_proposals": src, "proposals_file": a.proposals_file, "readout_lr_mult": a.readout_lr_mult, "readout_scope": a.readout_scope, "threads": a.threads},
              exclusive=True)
    updates, history, t0 = 0, [], time.time()
    keys = ("task", "ground", "contrast", "recovery")
    for epoch, s in enumerate(schedules):
        if updates >= stop:
            break
        seed_training_epoch(a.seed, epoch)
        model.train(); opt.zero_grad(set_to_none=True)
        batches = s["ordered_indices"]
        run = {k: [] for k in keys}
        for j, idxs in enumerate(batches):
            b = data.train_batch(idxs, model.cfg)
            rng = {"cpu": torch.get_rng_state(), "mps": torch.mps.get_rng_state() if a.device == "mps" else None}
            with private_training_loss(model, b, 3, "token_mean", data.table):
                out = model(b, stage=3)
            assert torch.isfinite(out["loss"]), (epoch, j)
            terms = {k: float(v.detach()) for k, v in out["terms"].items()}
            for k in keys:
                if k in terms:
                    run[k].append(terms[k])
            (out["loss"] / 4).backward()
            del out
            if a.nan_dump:
                bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:
                    dump_nonfinite(folder, model, epoch, j, updates, idxs, rng, terms, bad, j % 4)
            if (j + 1) % 4 == 0 or j + 1 == len(batches):
                if a.nan_dump:  # float64 CPU norm before clipping: tells an MPS norm artefact from real overflow
                    sq = {n: float(p.grad.detach().cpu().double().pow(2).sum()) for n, p in model.named_parameters()
                          if p.grad is not None}
                    mx = {n: float(p.grad.detach().abs().max().cpu()) for n, p in model.named_parameters()
                          if p.grad is not None}
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if a.nan_dump and not torch.isfinite(gn):
                    top = sorted(mx.items(), key=lambda kv: -kv[1] if np.isfinite(kv[1]) else -np.inf)[:10]
                    d = folder / "nonfinite_norm_dump"; d.mkdir(exist_ok=False)
                    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, d / "state.pt")
                    dump_json(d / "info.json", {"epoch": epoch, "microbatch_end": j, "updates_done": updates,
                                                "microbatches": [list(x) for x in batches[j - (j % 4): j + 1]],
                                                "mps_clip_norm": float(gn), "cpu_float64_norm": float(np.sqrt(sum(sq.values()))),
                                                "nonfinite_sq_params": [n for n, v in sq.items() if not np.isfinite(v)],
                                                "top_abs_grad": top, "recent_terms": {k: v[-4:] for k, v in run.items()}})
                assert torch.isfinite(gn)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); updates += 1
                if updates % 10 == 0 or updates == stop:
                    dump_json(folder / "progress.json", {"epoch": epoch + 1, "updates": updates, "stop": stop,
                                                         "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                                                         **{f"mean50_{k}": float(np.mean(v[-50:])) for k, v in run.items() if v},
                                                         "seconds": time.time() - t0})
                if updates >= stop:
                    break
        history.append({"epoch": epoch + 1, "updates": updates,
                        **{f"mean_{k}": float(np.mean(v)) for k, v in run.items() if v},
                        **{f"n_{k}": len(v) for k, v in run.items()}})
    metrics, rows = evaluate_canonical(model, data, "model_val", limit=a.eval_limit)
    res = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "overrides": overrides, "updates": updates,
           "history": history, "model_val": metrics, "seconds": time.time() - t0}
    dump_json(folder / "model_val_canonical.json", {"metrics": metrics, "rows": rows}, exclusive=True)
    if not a.no_save:
        model.save(str(folder / "LAST.pt"), {"updates": updates, "arm": a.arm, "seed": a.seed, "overrides": overrides})
    dump_json(folder / "completed.json", res, exclusive=True)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--fold", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--readout-lr-mult", type=float, default=1.0,
                    help="lr multiplier for the zero-init M_hat read-out (h_les, t_les, emphasis)")
    ap.add_argument("--readout-scope", default="all", choices=["all", "h_les"],
                    help="which read-out parameters get --readout-lr-mult (all = h_les, t_les, emphasis)")
    ap.add_argument("--proposals", default=None, choices=["crossfit", "crossfit_eqm"],
                    help="override the arm's TRAIN proposal source (default: registry)")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--proposals-file", default=None,
                    help="vg-proposals-v1 file of another detector (e.g. YOLOv8); the arm's registry entry still "
                         "decides whether TRAIN uses the cross-fit or the in-sample key")
    ap.add_argument("--nan-dump", action="store_true",
                    help="check gradients after every microbatch; on the first non-finite one save a replay dump and stop")
    main(ap.parse_args())
