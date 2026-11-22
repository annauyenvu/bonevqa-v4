"""After a KG run: how much the model uses the graph, what it retrieves, and its open / reasoning answers.

    source v4_grounded_next/env.sh
    $PY -m scripts.kg_report --tag joint_fold0/V4-CF_yolojoint_btxrd_kg --reference-tag joint_fold0/V4-CF_yolojoint_btxrd \
        --out kg_report_K --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json

Inference only, CPU.  Writes runs/joint/fold_<f>/<out>.json:
  gate        kg.gamma, norm of the KG residual relative to the pooled representation
  retrieval   on BTXRD test diagnosis rows: weight on the card of the true diagnosis (chance = 1 / cards),
              how often it is the top card, the most retrieved cards per true class
  answers     the same sampled OPEN and reasoning questions answered by the run and by the reference run
              (BTXRD test and FracAtlas test), with the ground truth; rows where the two differ are flagged
Labels are read only after inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter, defaultdict

import torch

from vg import kg as K
from vg.joint import JointFoldData
from vg.paths import RUNS, dump_json

SEED = 20261116


def pick_rows(data, n_bt, n_fa):
    rng = random.Random(SEED)
    bt = data.bt["test"].records
    ctx = [i for i, r in enumerate(bt) if r.get("clinical_context")]
    reason = [i for i, r in enumerate(bt) if r["question"].startswith(("Is the diagnosis", "Is the bone lesion benign?"))]
    opens = [i for i, r in enumerate(bt) if r["answer_type"] == "OPEN" and not r.get("clinical_context")
             and r["question"] != "What type of imaging is this?"]
    fa = [i for i, r in enumerate(data.ds["test"].records) if r["answer_type"] == "OPEN" and r["question"] != "What type of imaging is this?"]
    return {"btxrd": sorted(set(ctx + rng.sample(reason, min(60, len(reason))) + rng.sample(opens, min(n_bt, len(opens))))),
            "fracatlas": sorted(rng.sample(fa, min(n_fa, len(fa))))}


@torch.no_grad()
def answer(model, data, picks, batch_size=8):
    from src.utils import normalize_answer
    out = {}
    for ds, idx in picks.items():
        rows = []
        for off in range(0, len(idx), batch_size):
            chunk = idx[off:off + batch_size]
            if ds == "btxrd":
                records, b = data.bt_eval_batch("test", chunk)
            else:
                records, _, b = data.eval_batch("test", chunk)
            r = model.infer(b)
            closed = torch.softmax(r["logits_closed"], dim=-1)
            gen = model.generate_open(r["core"], r["core"]["q_texts"])
            for k, rec in enumerate(records):
                if rec["answer_type"] == "OPEN":
                    pred, conf = gen[k]["text"], gen[k]["confidence"]
                else:
                    j = int(closed[k].argmax())
                    pred, conf = model.answer_vocab[j], float(closed[k, j])
                rows.append({"dataset": ds, "image_id": rec["image_id"], "question": rec["question"], "type": rec["answer_type"],
                             "truth": rec["answer"], "pred": pred, "confidence": round(conf, 4),
                             "correct": normalize_answer(pred) == normalize_answer(rec["answer"]),
                             "diagnosis": rec.get("diagnosis"), "has_proposal": bool(len(b["proposals"][k]))})
        out[ds] = rows
    return out


@torch.no_grad()
def retrieval(model, data, limit=None, batch_size=8):
    cards = [c["node"] for c in model.kg_cards]
    idx = data.bt_indices("test", "diagnosis")[: limit or None]
    per = defaultdict(lambda: {"n": 0, "w_true": 0.0, "top1": 0, "top": Counter()})
    ratio, n = 0.0, 0
    for off in range(0, len(idx), batch_size):
        records, b = data.bt_eval_batch("test", idx[off:off + batch_size])
        core = model.infer(b)["core"]
        w, delta = core["kg_weights"], core["kg_delta"]
        ratio += float((delta.norm(dim=-1) / core["fus"]["x_f"].norm(dim=-1)).sum())
        n += len(records)
        if w is None:
            continue
        for k, rec in enumerate(records):
            node = model.kg_label_map["btxrd"].get(rec["subtype"]) if rec.get("subtype") else None
            node = node or model.kg_label_map["btxrd"].get(rec["diagnosis"])
            s = per[rec["answer"]]
            s["n"] += 1
            top = int(w[k].argmax())
            s["top"][cards[top]] += 1
            if node in cards:
                s["w_true"] += float(w[k, cards.index(node)])
                s["top1"] += int(cards[top] == node)
    classes = {c: {"n": s["n"], "mean_weight_on_true_card": s["w_true"] / s["n"], "true_card_is_top": s["top1"] / s["n"],
                   "most_retrieved": s["top"].most_common(3)} for c, s in sorted(per.items())}
    tot = sum(s["n"] for s in per.values())
    return {"rows": n, "cards": len(cards), "chance_weight": 1 / len(cards) if cards else None,
            "mean_weight_on_true_card": sum(s["w_true"] for s in per.values()) / tot if tot else None,
            "true_card_is_top": sum(s["top1"] for s in per.values()) / tot if tot else None,
            "residual_norm_over_xf_norm": ratio / max(n, 1), "by_class": classes}


def main(a):
    torch.set_num_threads(a.threads)
    out = RUNS / "joint" / f"fold_{a.fold}" / f"{a.out}.json"
    assert not out.exists(), out
    data = JointFoldData(a.fold, "insample_v4", a.proposals_file, a.btxrd_vqa, a.btxrd_proposals)
    picks = pick_rows(data, a.n_btxrd, a.n_fracatlas)
    if a.limit:
        picks = {k: v[: a.limit] for k, v in picks.items()}
    model, started, sha = K.load_kg_arm(a.tag, a.fold, "cpu")
    rep = {"tag": a.tag, "checkpoint_sha256": sha, "kg": started["model_cfg"].get("kg")}
    if rep["kg"]:
        rep["gate"] = {"gamma": float(model.kg.gamma), "gamma_lr_mult": rep["kg"].get("gamma_lr_mult")}
        rep["retrieval"] = retrieval(model, data, a.limit)
    mine = answer(model, data, picks)
    del model
    gc.collect()
    ref = None
    if a.reference_tag:
        ref_model, _, ref_sha = K.load_kg_arm(a.reference_tag, a.fold, "cpu")
        ref = answer(ref_model, data, picks)
        rep["reference"] = {"tag": a.reference_tag, "checkpoint_sha256": ref_sha}
        del ref_model
    rows, summary = [], {}
    for ds in mine:
        for k, r in enumerate(mine[ds]):
            row = dict(r)
            if ref:
                row.update(ref_pred=ref[ds][k]["pred"], ref_correct=ref[ds][k]["correct"],
                           differs=ref[ds][k]["pred"].strip().lower() != r["pred"].strip().lower())
            rows.append(row)
    by_q = defaultdict(list)
    for r in rows:
        q = "diagnosis with age and sex" if r["question"].startswith("The patient is") else \
            ("Is the diagnosis X?" if r["question"].startswith("Is the diagnosis") else r["question"])
        by_q[(r["dataset"], q)].append(r)
    for (ds, q), rs in sorted(by_q.items()):
        summary[f"{ds} | {q}"] = {"n": len(rs), "accuracy": sum(x["correct"] for x in rs) / len(rs),
                                  **({"reference_accuracy": sum(x["ref_correct"] for x in rs) / len(rs),
                                      "answers_differ": sum(x["differs"] for x in rs)} if ref else {})}
    rep["answers_summary"] = summary
    rep["notable"] = [r for r in rows if r.get("differs") and r["correct"] != r.get("ref_correct")][:80]
    dump_json(out, {**rep, "rows": rows}, exclusive=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "notable"}, ensure_ascii=False, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reference-tag", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--proposals-file", required=True)
    ap.add_argument("--btxrd-vqa", required=True)
    ap.add_argument("--btxrd-proposals", required=True)
    ap.add_argument("--n-btxrd", type=int, default=200)
    ap.add_argument("--n-fracatlas", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args())
