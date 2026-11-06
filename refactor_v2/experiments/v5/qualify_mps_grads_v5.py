"""MPS training qualification: per-parameter gradients on MPS vs CPU for real v5 arms.

Same initial weights (study_v5.construct), same TRAIN micro-batches, dropout disabled, one backward
of the private training loss.  Reports every trainable parameter whose MPS gradient deviates from CPU.
Run twice: with the layer_norm fix (default) and with BONEVQA_MPS_LN_FIX=0 to show the torch 2.6 bug.

  python refactor_v2/experiments/v5/qualify_mps_grads_v5.py B P V4
"""
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from experiments.v5 import study_v5 as S
from src.utils import normalize_answer
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

N_MICRO = 4  # one optimizer update's worth of micro-batches (accum 4)

# Qualification-only variants: same model config as the V / L cells, with region sources that
# already exist (detector proposals instead of AVPC, MedSAM-only lens masks) so every code path
# (HAG, circle prompts, Lens views + pixel shuffle, LaPA, T) is checked before the masks are done.
_V, _L = S.FACTORS["V"], S.FACTORS["L"]
QUAL_ARMS = {
    "QV_det": dict(model=dict(_V["model"]), proposals="detector"),
    "QL_medsam": dict(model=dict(_L["model"]), proposals="none", lens={"sources": ["medsam"], "lesion": False}),
    "QVLPT": dict(model={**_V["model"], **_L["model"], **S.FACTORS["P"]["model"], **S.FACTORS["T"]["model"]},
                  proposals="detector", lens={"sources": ["medsam"], "lesion": False}, kg="real"),
}
S.ARMS.update(QUAL_ARMS)


def grads(arm, device, fold="0", seed=42, suite="factorial_fold0"):
    spec = S.arm_spec(arm)
    cfg = S.build_cfg(fold, arm, seed, suite)
    model = S.construct(cfg, device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0
    ds, collate = S.batcher(fold, spec)
    table = {(r["question"], normalize_answer(r["answer"])): 1.0 for r in ds["train"].records if r["answer_type"] == "OPEN"}
    order = S.schedule_for(fold, int(seed))[0]["ordered_indices"][:N_MICRO]
    model.train()
    losses = []
    for b in DataLoader(ds["train"], batch_sampler=order, collate_fn=collate, num_workers=0):
        with private_training_loss(model, b, 3, "token_mean", table):
            out = model(b, stage=3)
        (out["loss"] / N_MICRO).backward()
        losses.append(float(out["loss"].detach()))
    g = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    del model
    return g, losses, missing


TOL, ZERO = 1e-2, 1e-6   # ||g_mps - g_cpu|| / ||g_cpu|| per parameter; skip analytically-zero grads


def compare(gc_, gm, mc, mm):
    """Per-parameter MPS-vs-CPU check used both here and as a hard gate before every MPS job."""
    rows, problems = [], []
    if sorted(mc) != sorted(mm) or sorted(gc_) != sorted(gm):
        problems.append({"param_set_differs": sorted(set(gc_) ^ set(gm)) + sorted(set(mc) ^ set(mm))})
    for n, a in gc_.items():
        b = gm.get(n)
        if b is None:
            continue
        if not bool(torch.isfinite(b).all()):
            problems.append({"param": n, "issue": "non-finite on mps"})
            continue
        na, nb = float(a.norm()), float(b.norm())
        rel = float((a - b).norm()) / (na + 1e-12)
        rows.append((n, rel, na))
        if max(na, nb) >= ZERO and rel > TOL:
            problems.append({"param": n, "rel": rel, "norm_cpu": na, "norm_mps": nb})
    rows.sort(key=lambda r: -r[1])
    return {"passed": not problems, "problems": problems[:20], "params": len(rows), "tol": TOL, "zero_floor": ZERO,
            "worst_checked": [r for r in rows if r[2] >= ZERO][:5]}


def qualify(arm, fold, seed, suite):
    gc_, lc, mc = grads(arm, "cpu", fold, seed, suite)
    gm, lm, mm = grads(arm, "mps", fold, seed, suite)
    return {**compare(gc_, gm, mc, mm), "loss_cpu": lc, "loss_mps": lm}


def main(arms):
    torch.set_num_threads(8)
    fix = os.environ.get("BONEVQA_MPS_LN_FIX", "1") != "0"
    report = {}
    for arm in arms:
        t = time.time()
        gc_, lc, mc = grads(arm, "cpu")
        gm, lm, mm = grads(arm, "mps")
        rows = []
        for n, a in gc_.items():
            b = gm.get(n)
            if b is None:
                rows.append((n, None, "missing on mps"))
                continue
            finite = bool(torch.isfinite(b).all())
            na = float(a.norm())
            rel = float((a - b).norm()) / (na + 1e-12) if finite else float("inf")
            cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)) if finite else None
            rows.append((n, rel, f"norm_cpu={na:.3g} cos={cos}"))
        rows.sort(key=lambda r: -(r[1] if r[1] is not None else 1e9))
        bad = [r for r in rows if r[1] is None or r[1] > 1e-2]  # ||g_mps - g_cpu|| / ||g_cpu||
        report[arm] = {"ln_fix": fix, "params": len(rows), "loss_cpu": lc, "loss_mps": lm,
                       "max_rel_err": rows[0][1] if rows else None, "n_rel_err_gt_1e-2": len(bad),
                       "worst": rows[:8], "missing_cpu": mc, "missing_mps": mm, "seconds": time.time() - t}
        print(json.dumps({arm: report[arm]}, indent=1), flush=True)
    out = S.OUT / f"mps_grad_qualification_{'fix' if fix else 'nofix'}.json"
    S.dump(out, report)
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["B", "P", "V4"])
