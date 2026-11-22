"""Exact CPU-vs-MPS gradient parity on real training microbatches (dropout off).

Loads a saved state (nonfinite_norm_dump/state.pt of --tag), runs each listed
microbatch on both devices and reports, per parameter, |g_cpu - g_mps| relative to
the global CPU gradient norm.  Passes when the worst parameter is below --tol.

    $PY -m scripts.grad_device_parity --tag fold0/V4-G
"""
from __future__ import annotations

import argparse
import json

import torch

from vg.paths import RUNS, dump_json, read_json
from vg.registry import ARMS
from vg.trainer import FoldData, build_model


def grads(model, data, idxs, device):
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    model.to(device).train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    b = data.train_batch(idxs, model.cfg)
    with private_training_loss(model, b, 3, "token_mean", data.table):
        o = model(b, stage=3)
    o["loss"].backward()
    return float(o["loss"]), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def main(a):
    torch.set_num_threads(4)
    d = RUNS / "arms" / a.tag / f"fold_{a.fold}" / "seed_42"
    info = read_json(d / "nonfinite_norm_dump" / "info.json")
    started = read_json(d / "started.json")
    model, _ = build_model(started["arm"], a.fold, device="cpu", overrides={**started["overrides"], "lora_dropout": 0.0})
    model.load_state_dict(torch.load(d / "nonfinite_norm_dump" / "state.pt", map_location="cpu", weights_only=False)["state"])
    data = FoldData(a.fold, ARMS[started["arm"]]["proposals"])
    out, ok = [], True
    for k, idxs in enumerate(info["microbatches"]):
        lc, gc = grads(model, data, idxs, "cpu")
        lm, gm = grads(model, data, idxs, "mps")
        tot = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gc.values())))
        err = {n: float((gc[n] - gm[n]).norm()) / tot for n in gc}
        worst = sorted(err.items(), key=lambda kv: -kv[1])[:5]
        rec = {"microbatch": k, "indices": idxs, "loss_cpu": lc, "loss_mps": lm, "cpu_grad_norm": tot,
               "worst_rel_err": worst, "pass": worst[0][1] < a.tol and abs(lc - lm) < 1e-3}
        ok &= rec["pass"]; out.append(rec); print(json.dumps(rec), flush=True)
    dump_json(RUNS / "gates" / "grad_device_parity.json", {"tag": a.tag, "tol": a.tol, "passed": ok, "microbatches": out})
    print("PASSED" if ok else "FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--tol", type=float, default=2e-2)
    main(ap.parse_args())
