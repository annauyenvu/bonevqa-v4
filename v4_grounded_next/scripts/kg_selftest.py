"""Checks of the knowledge-graph latent prompt (vg/kg.py).  CPU, a few batches, writes nothing.

    source v4_grounded_next/env.sh
    $PY -m scripts.kg_selftest --cards-only          # card texts and lengths only (no model weights changed)
    $PY -m scripts.kg_selftest --threads 3 \
        --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json

1. cards: no card over the token limit; the shuffled control is a derangement; the placebo keeps norms
2. parity: with gamma = 0 the attached model returns the outputs of the model before attachment
3. effect: with gamma != 0 the outputs move, and a change of the bank changes them
4. gradients: at gamma = 0 only gamma receives gradient from the new module; after one step all do
"""
from __future__ import annotations

import argparse
import json

import torch

from vg import kg as K
from vg.trainer import build_model


def cards_report(tokenizer):
    out = {}
    for mode in ("kg", "shuffled"):
        cards, info = K.build_cards(mode=mode)
        n = [len(x) for x in tokenizer([c["text"] for c in cards], padding=False, truncation=False)["input_ids"]]
        out[mode] = {"cards": len(cards), "diagnosis_cards": info["diagnosis_cards"], "tokens_min": min(n), "tokens_max": max(n),
                     "tokens_mean": round(sum(n) / len(n), 1), "longest": cards[n.index(max(n))]["node"]}
        assert max(n) <= K.CARD_MAX_TOKENS, out[mode]
        if mode == "shuffled":
            dx = [c for c in cards if c["type"] == "diagnosis"]
            assert all(c["node"] != c["facts_from"] for c in dx) and sorted(c["facts_from"] for c in dx) == sorted(c["node"] for c in dx)
    cards, info = K.build_cards()
    out["graph"] = {k: info[k] for k in ("graph_version", "graph_sha256", "cards")}
    out["example"] = next(c["text"] for c in cards if c["node"] == "simple_bone_cyst")
    return out, cards


def main(a):
    torch.set_num_threads(a.threads)
    model, _ = build_model("V4-CF", a.fold, device="cpu")
    rep, cards = cards_report(model.text.tokenizer)
    print(json.dumps(rep, indent=1), flush=True)
    if a.cards_only:
        return
    from vg.joint import JointFoldData
    data = JointFoldData(a.fold, "crossfit", a.proposals_file, a.btxrd_vqa, a.btxrd_proposals)
    _, bt = data.bt_eval_batch("model_val", data.bt_indices("model_val", "diagnosis")[:4])
    fa = data.eval_batch("model_val", data.canonical_indices("model_val")[:4])
    fa = fa[-1] if isinstance(fa, tuple) else fa

    def run(batch):
        r = model.infer(batch)
        return r["logits_closed"].clone(), r["p_final"].clone() if r["p_final"] is not None else r["p_local"].clone()

    model.eval()
    before = [run(bt), run(fa)]
    meta = K.attach_kg(model, bank_mode="kg", n_latent=a.kg_latents)
    model.eval()
    new = sum(p.numel() for p in model.kg.parameters())
    bank = model.kg.bank.clone()
    sim = torch.nn.functional.normalize(bank, dim=-1) @ torch.nn.functional.normalize(bank, dim=-1).t()
    ids = [c["node"] for c in cards]
    i = ids.index("osteochondroma")
    near = [ids[j] for j in sim[i].argsort(descending=True)[1:4].tolist()]
    after = [run(bt), run(fa)]
    parity = max(float((x - y).abs().max()) for b0, b1 in zip(before, after) for x, y in zip(b0, b1))
    assert parity == 0.0, parity

    with torch.no_grad():
        model.kg.gamma.fill_(0.5)
    moved = [run(bt), run(fa)]
    effect = max(float((x - y).abs().max()) for b0, b1 in zip(before, moved) for x, y in zip(b0, b1))
    w = model._kg_last["weights"]
    model.kg.bank = K.placebo_bank(bank)
    assert abs(float(model.kg.bank.norm(dim=-1).sub(bank.norm(dim=-1)).abs().max())) < 1e-4
    placebo = [run(bt), run(fa)]
    bank_effect = max(float((x - y).abs().max()) for b0, b1 in zip(moved, placebo) for x, y in zip(b0, b1))
    model.kg.bank = bank
    assert effect > 0 and bank_effect > 0, (effect, bank_effect)

    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

    def grads(gamma):
        with torch.no_grad():
            model.kg.gamma.fill_(gamma)
        model.train()
        model.zero_grad(set_to_none=True)
        b = data.train_batch([data.n_fa, data.n_fa + 1, data.n_fa + 2, data.n_fa + 3], model.cfg)
        with private_training_loss(model, b, 3, "token_mean", data.table):
            out = model(b, stage=3)
        out["loss"].backward()
        g = {n: (0.0 if p.grad is None else float(p.grad.abs().sum())) for n, p in model.kg.named_parameters()}
        model.zero_grad(set_to_none=True)
        return float(out["loss"]), g

    loss0, g0 = grads(0.0)
    loss1, g1 = grads(0.5)
    assert g0["gamma"] > 0 and all(v == 0 for n, v in g0.items() if n != "gamma"), g0
    dead = [n for n, v in g1.items() if v == 0]
    assert not dead, dead
    with torch.no_grad():
        model.kg.gamma.zero_()
    print(json.dumps({"kg": meta, "new_parameters": new, "nearest_cards_to_osteochondroma": near,
                      "parity_max_abs_diff_gamma0": parity, "max_abs_change_gamma0.5": effect,
                      "max_abs_change_kg_vs_placebo_bank": bank_effect, "retrieval_weights_shape": list(w.shape),
                      "retrieval_weights_row_sum": float(w.sum(-1).mean()),
                      "loss_gamma0": loss0, "loss_gamma0.5": loss1, "grad_gamma_at_0": g0["gamma"],
                      "kg_parameters_with_gradient_at_0.5": len(g1) - len(dead)}, indent=1), flush=True)
    print("KG self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--kg-latents", type=int, default=16)
    ap.add_argument("--cards-only", action="store_true")
    ap.add_argument("--proposals-file")
    ap.add_argument("--btxrd-vqa")
    ap.add_argument("--btxrd-proposals")
    main(ap.parse_args())
