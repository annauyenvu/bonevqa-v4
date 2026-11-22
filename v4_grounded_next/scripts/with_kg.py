"""Run an existing script with the knowledge-graph latent prompt (vg/kg.py), leaving that script as it is.

Training: the model is built by the unchanged V4 protocol, then the graph module is attached.

    source v4_grounded_next/env.sh
    $PY -m scripts.with_kg train_joint --kg-bank kg -- --arm V4-CF --tag joint_fold0/V4-CF_yolojoint_btxrd_kg \
        --device cpu --threads 6 --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json

Evaluation: the graph settings are read from the run's started.json (and the card bank must hash to
the one used in training), so nothing about the graph is passed; runs without a graph load as before.

    $PY -m scripts.with_kg eval_joint -- --name K_kg --tag joint_fold0/V4-CF_yolojoint_btxrd_kg --role test ...
    $PY -m scripts.with_kg eval_arm   -- --name K_kg --tag joint_fold0/V4-CF_yolojoint_btxrd_kg --role test ...

Everything before ``--`` belongs to this wrapper, everything after it to the wrapped script.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys

SCRIPTS = ("train_joint", "eval_joint", "eval_arm")


def main():
    argv = sys.argv[1:]
    assert "--" in argv, "usage: with_kg <script> [--kg-bank ...] -- <arguments of the script>"
    own, rest = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    import vg.kg as K
    ap = argparse.ArgumentParser()
    ap.add_argument("script", choices=SCRIPTS)
    ap.add_argument("--kg-graph", default=str(K.KG_GRAPH))
    ap.add_argument("--kg-bank", choices=K.BANK_MODES, default=None, help="training only")
    ap.add_argument("--kg-latents", type=int, default=16)
    ap.add_argument("--kg-gamma-lr-mult", type=float, default=1.0,
                    help="training only: learning-rate multiplier of the zero-initialised gate kg.gamma "
                         "(1 = exactly the V4 optimiser; same schedule, betas, weight decay and clip)")
    a = ap.parse_args(own)

    import scripts.mechanism_tests as MT
    import vg.trainer as T
    MT.load_arm = K.load_kg_arm
    if a.script == "train_joint":
        assert a.kg_bank, "--kg-bank is required for training"
        base = T.build_model

        base_opt = T.make_optimizer

        def build(*args, **kwargs):
            model, cfg = base(*args, **kwargs)
            meta = K.attach_kg(model, a.kg_graph, a.kg_bank, a.kg_latents)
            meta["gamma_lr_mult"] = a.kg_gamma_lr_mult      # meta is model.cfg["kg"], so it is saved with the run
            print(json.dumps({"kg": meta}), flush=True)
            return model, cfg

        def make_optimizer(model, lr):
            if a.kg_gamma_lr_mult == 1.0:
                return base_opt(model, lr)
            import torch
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            gate = [p for n, p in named if n == "kg.gamma"]
            assert len(gate) == 1
            groups = [{"params": [p for n, p in named if n != "kg.gamma"], "lr": lr},
                      {"params": gate, "lr": lr * a.kg_gamma_lr_mult}]
            return [p for _, p in named], torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

        T.build_model, T.make_optimizer = build, make_optimizer
    else:
        assert a.kg_bank is None and a.kg_gamma_lr_mult == 1.0, "evaluation reads the graph settings from started.json"
    sys.argv = [f"scripts/{a.script}.py", *rest]
    runpy.run_module(f"scripts.{a.script}", run_name="__main__")


if __name__ == "__main__":
    main()
