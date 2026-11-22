"""Replay the update saved by train_arm --nan-dump (nonfinite_norm_dump/): same weights,
same 4 microbatches, per-microbatch gradient of the suspicious parameters on a device.

    $PY -m scripts.replay_nonfinite --tag fold0/V4-G --device cpu
"""
from __future__ import annotations

import argparse
import json

import torch

from vg.paths import RUNS, dump_json, read_json
from vg.registry import ARMS
from vg.trainer import FoldData, build_model


def main(a):
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    torch.set_num_threads(4)
    d = RUNS / "arms" / a.tag / f"fold_{a.fold}" / "seed_42"
    info = read_json(d / "nonfinite_norm_dump" / "info.json")
    started = read_json(d / "started.json")
    model, _ = build_model(started["arm"], a.fold, device="cpu", overrides=started["overrides"])
    state = torch.load(d / "nonfinite_norm_dump" / "state.pt", map_location="cpu", weights_only=False)["state"]
    model.load_state_dict(state, strict=True)
    model.to(a.device).train()
    data = FoldData(a.fold, ARMS[started["arm"]]["proposals"])
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    named = dict(model.named_parameters())
    out = []
    for rep in range(a.repeats):
        torch.manual_seed(rep)
        for k, idxs in enumerate(info["microbatches"]):
            model.zero_grad(set_to_none=True)
            b = data.train_batch(idxs, model.cfg)
            with private_training_loss(model, b, 3, "token_mean", data.table):
                o = model(b, stage=3)
            (o["loss"] / 4).backward()
            mx = {n: float(named[n].grad.detach().abs().max().cpu()) for n in names if named[n].grad is not None}
            top = sorted(mx.items(), key=lambda kv: -kv[1])[:3]
            g = named["local_tokens.norm.bias"].grad.detach().cpu()
            rec = {"rep": rep, "microbatch": k, "indices": idxs, "loss": float(o["loss"]),
                   "n_proposals": [len(p) for p in b["proposals"]], "top3": top,
                   "ln_bias_grad_max": float(g.abs().max()), "ln_bias_argmax": int(g.abs().argmax()),
                   "ln_bias_grad_median": float(g.abs().median())}
            out.append(rec); print(json.dumps(rec), flush=True)
    dump_json(d / "nonfinite_norm_dump" / f"replay_{a.device}.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=1)
    main(ap.parse_args())
