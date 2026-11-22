"""Train one arm on FracAtlas + BTXRD (vg/joint.py).  CPU only, like every V4 run.

Everything of scripts/train_arm.py is kept: PREtask init, AdamW(lr 1e-4, betas .9/.98, wd .01),
5 % warm-up + cosine over the full update count, gradient accumulation 4, global clip 1.0,
private token-mean sequence loss, ``seed_training_epoch``.  The only change is the batch
stream: each sealed FracAtlas microbatch is followed by one BTXRD microbatch, so an epoch has
about twice the updates of a V4 epoch.  This script is separate from train_arm.py on purpose:
train_arm.py stays exactly what the fold-0 ablation ran.

    source v4_grounded_next/env.sh
    $PY -m scripts.train_joint --arm V4 --tag joint_fold0/V4_joint_insample --device cpu --threads 6 \
        --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0_insample.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from vg.joint import JointFoldData, evaluate_btxrd
from vg.paths import RUNS, dump_json, guard_write
from vg.registry import ARMS
from vg.trainer import build_model, evaluate_canonical, lr_lambda, make_optimizer, v4_schedule


def main(a):
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    from src.train import seed_training_epoch
    assert a.device == "cpu", "V4 trains on CPU only (MPS gradients are wrong for this model)"
    torch.set_num_threads(a.threads)
    folder = guard_write(RUNS / "arms" / a.tag / f"fold_{a.fold}" / f"seed_{a.seed}")
    folder.mkdir(parents=True, exist_ok=False)
    model, v4cfg = build_model(a.arm, a.fold, device=a.device)
    data = JointFoldData(a.fold, ARMS[a.arm]["proposals"], a.proposals_file, a.btxrd_vqa, a.btxrd_proposals, a.btxrd_train_key)
    schedules = data.joint_schedule(v4_schedule(a.fold), a.seed)
    total = sum(e["updates"] for e in schedules)
    stop = min(total, a.max_updates or total)
    params, opt = make_optimizer(model, v4cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total))
    dump_json(folder / "started.json", {
        "pid": os.getpid(), "arm": a.arm, "fold": a.fold, "seed": a.seed, "device": a.device, "overrides": {},
        "model_cfg": model.cfg, "joint": True, "total_updates_schedule": total, "stop_after": stop,
        "schedule": [{k: v for k, v in s.items() if k != "ordered_indices"} for s in schedules],
        "train_rows": {"fracatlas": data.n_fa, "btxrd": data.n_bt},
        "proposals_file": a.proposals_file, "proposals_sha256": data.proposals_sha256, "train_source": data.source["train"],
        "btxrd_vqa": a.btxrd_vqa, "btxrd_vqa_sha256": data.btxrd_vqa_sha256, "btxrd_proposals": a.btxrd_proposals,
        "btxrd_proposals_sha256": data.btxrd_proposals_sha256, "btxrd_train_key": a.btxrd_train_key,
        "btxrd_train_source": data.bt_source["train"], "threads": a.threads}, exclusive=True)
    updates, history, t0 = 0, [], time.time()
    for epoch, s in enumerate(schedules):
        if updates >= stop:
            break
        seed_training_epoch(a.seed, epoch)
        model.train(); opt.zero_grad(set_to_none=True)
        batches = s["ordered_indices"]
        run = {"fracatlas": [], "btxrd": []}
        for j, idxs in enumerate(batches):
            b = data.train_batch(idxs, model.cfg)
            with private_training_loss(model, b, 3, "token_mean", data.table):
                out = model(b, stage=3)
            assert torch.isfinite(out["loss"]), (epoch, j)
            run["fracatlas" if idxs[0] < data.n_fa else "btxrd"].append(float(out["loss"].detach()))
            (out["loss"] / 4).backward()
            del out
            if (j + 1) % 4 == 0 or j + 1 == len(batches):
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(gn), (epoch, j)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); updates += 1
                if updates % 10 == 0 or updates == stop:
                    dump_json(folder / "progress.json", {
                        "epoch": epoch + 1, "updates": updates, "stop": stop, "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                        **{f"mean50_loss_{k}": float(np.mean(v[-50:])) for k, v in run.items() if v}, "seconds": time.time() - t0})
                if updates >= stop:
                    break
        history.append({"epoch": epoch + 1, "updates": updates,
                        **{f"mean_loss_{k}": float(np.mean(v)) for k, v in run.items() if v},
                        **{f"n_{k}": len(v) for k, v in run.items()}})
    fa_metrics, fa_rows = evaluate_canonical(model, data, "model_val", limit=a.eval_limit)
    bt_metrics, bt_rows = evaluate_btxrd(model, data, "model_val", limit=a.eval_limit)
    res = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "overrides": {}, "joint": True, "updates": updates, "history": history,
           "model_val": fa_metrics, "btxrd_model_val": bt_metrics, "seconds": time.time() - t0}
    dump_json(folder / "model_val_canonical.json", {"metrics": fa_metrics, "rows": fa_rows}, exclusive=True)
    dump_json(folder / "btxrd_model_val.json", {"metrics": bt_metrics, "rows": bt_rows}, exclusive=True)
    if not a.no_save:
        model.save(str(folder / "LAST.pt"), {"updates": updates, "arm": a.arm, "seed": a.seed, "overrides": {}})
    dump_json(folder / "completed.json", res, exclusive=True)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["V4", "V4-CF"],
                    help="V4: FracAtlas TRAIN proposals in-sample; V4-CF: cross-fitted (same meaning as in train_arm)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--proposals-file", required=True, help="FracAtlas proposals (vg-proposals-v1)")
    ap.add_argument("--btxrd-vqa", required=True, help="folder with vqa_{train,val,test,external}.json")
    ap.add_argument("--btxrd-proposals", required=True, help="BTXRD proposals (vg-proposals-v1, dataset btxrd)")
    ap.add_argument("--btxrd-train-key", default="train_insample_audit_only", choices=["train_insample_audit_only", "train"])
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    main(ap.parse_args())
