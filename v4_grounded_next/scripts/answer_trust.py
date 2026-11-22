"""Answer questions with a joint checkpoint and attach the trust layer's verdict.  Inference only, CPU.

    source v4_grounded_next/env.sh; cd v4_grounded_next
    # every canonical question of some images
    $PY -m scripts.answer_trust --name J_v4cf_yolojoint_btxrd --dataset btxrd --role test --image-id BT_IMG000002 BT_IMG000140
    # the first 40 rows of one endpoint, checked against the stored evaluation scores
    $PY -m scripts.answer_trust --name J_v4cf_yolojoint_btxrd --dataset fracatlas --role test --endpoint fracture --limit 40 --check
    # a free question about one image; the diagnosis shortlist costs one Qwen pass per candidate
    $PY -m scripts.answer_trust --name J_v4cf_yolojoint_btxrd --dataset btxrd --role external --image-id BT_IMG001346 \
        --question "What is the most likely diagnosis?" --shortlist

The model, its answers and its probabilities are exactly those of scripts.eval_arm / scripts.eval_joint; this script only
adds vg.trust.TrustLayer on top.  Images must be dataset images, because proposals are read from the run's proposal files
(running the detector on a new image is not wired in here).  Labels are never given to the model; with --check the stored
evaluation rows are compared after inference.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import torch

from scripts.reference_score import candidate_loglik
from vg import kg as K
from vg import trust as T
from vg.data import public_batch
from vg.joint import JointFoldData
from vg.paths import RUNS, dump_json, read_json

YES_NO = re.compile(r"^(is|are|does|do|has|have|can|was|were)\b", re.I)


def is_closed(question: str) -> bool:
    return bool(YES_NO.match(question.strip())) and " or " not in question.lower()


def pick_rows(data, a):
    """[(record, dataset index)] for the request; a free --question reuses the image's first record with new text."""
    ds = data.bt[a.role] if a.dataset == "btxrd" else data.ds[a.role]
    wanted = set(T.ENDPOINT_OF) | {T.DIAGNOSIS_Q}
    rows = []
    if a.image_id:
        for im in a.image_id:
            idx = [i for i, r in enumerate(ds.records) if r["image_id"] == im]
            assert idx, f"{im} is not in {a.dataset} {a.role}"
            if a.question:
                rows.append((dict(ds.records[idx[0]], question=a.question, answer=None, answer_type=None), idx[0]))
            else:
                rows += [(ds.records[i], i) for i in idx if ds.records[i]["question"] in wanted]
    else:
        q = next(q for q, e in {**T.ENDPOINT_OF, T.DIAGNOSIS_Q: "diagnosis"}.items() if e == a.endpoint)
        rows = [(r, i) for i, r in enumerate(ds.records) if r["question"] == q][: a.limit]
    return ds, rows


@torch.no_grad()
def run(model, data, layer, ds, rows, a):
    props = data.bt_props[a.role] if a.dataset == "btxrd" else data.props[a.role]
    source = data.bt_source[a.role] if a.dataset == "btxrd" else data.source[a.role]
    yes, no = model.answer_to_idx["yes"], model.answer_to_idx["no"]
    kind = lambda q: ("endpoint" if q in T.ENDPOINT_OF else "diagnosis" if q == T.DIAGNOSIS_Q and a.shortlist
                      else "closed" if is_closed(q) else "open")
    out = []
    for k in ("endpoint", "diagnosis", "closed", "open"):
        group = [(r, i) for r, i in rows if kind(r["question"]) == k]
        for off in range(0, len(group), a.batch_size):
            chunk = group[off:off + a.batch_size]
            records = [r for r, _ in chunk]
            b = public_batch(records, [ds[i]["image"] for _, i in chunk], props, source, data.lookup)
            res = model.infer(b)
            pr = torch.softmax(res["logits_closed"], dim=-1)
            if k == "diagnosis":
                scores = candidate_loglik(model, res["core"], res["core"]["q_texts"], layer.dx["candidates"])
                gen = model.generate_open(res["core"], res["core"]["q_texts"])
            elif k == "open":
                gen = model.generate_open(res["core"], res["core"]["q_texts"])
            for j, rec in enumerate(records):
                q = rec["question"]
                if k == "endpoint":
                    ep = T.ENDPOINT_OF[q]
                    p = float(res["p_final"][j]) if ep == "fracture" else float(pr[j, yes] / (pr[j, yes] + pr[j, no]).clamp_min(1e-9))
                    v = layer.closed(ep, p, T.top_box_score(b["proposals"][j]), a.coverage, a.signal)
                elif k == "diagnosis":
                    v = layer.diagnosis(scores[j].numpy(), a.coverage, generated=gen[j]["text"])
                elif k == "closed":
                    c = int(pr[j].argmax())
                    v = layer.uncalibrated(model.answer_vocab[c], float(pr[j, c]))
                else:
                    v = layer.uncalibrated(gen[j]["text"], float(gen[j]["confidence"]))
                out.append({"image_id": rec["image_id"], "question": q, **v, "truth": rec.get("answer")})
    return out


def check(out, a):
    """Probabilities and verdicts against the stored evaluation rows of the same run (read after inference)."""
    stored = {}
    if a.dataset == "fracatlas":
        f = RUNS / "yolo_ablation" / f"fold_{a.fold}" / f"{a.name}_{'calibration' if a.role == 'calibration' else a.role}.json"
        stored["fracture"] = {r["image_id"]: r["p_final"] for r in read_json(f)["rows"]}
    else:
        R = read_json(RUNS / "joint" / f"fold_{a.fold}" / f"ref_{a.name}_{a.role}.json")
        for ep in ("tumor", "malignancy"):
            stored[ep] = {r["image_id"]: r["plain"] for r in R["endpoints"][ep]}
    d = [abs(o["p_yes"] - stored[o["endpoint"]][o["image_id"]]) for o in out if o.get("endpoint") in stored and o["image_id"] in stored[o["endpoint"]]]
    rows = [o for o in out if o.get("endpoint") in stored]
    ok = [o["answer"] == o["truth"] for o in rows]
    kept = [o["decision"] == "answer" for o in rows]
    return {"rows_compared": len(d), "max_abs_dp_vs_stored": max(d) if d else None, "n": len(rows),
            "accuracy_all": sum(ok) / len(ok) if ok else None, "answered": sum(kept),
            "accuracy_answered": (sum(o and k for o, k in zip(ok, kept)) / sum(kept)) if sum(kept) else None,
            "errors": len(ok) - sum(ok), "errors_deferred": sum((not o) and (not k) for o, k in zip(ok, kept))}


def main(a):
    torch.set_num_threads(a.threads)
    fold_dir = RUNS / "joint" / f"fold_{a.fold}"
    layer = T.TrustLayer.load(a.trust or fold_dir / f"trust_{a.name}.json")
    tag = a.tag or layer.spec["tag"]
    cf = RUNS / "crossfit_yolo_joint"
    data = JointFoldData(a.fold, "insample_v4", a.proposals_file or cf / f"fold_{a.fold}" / f"proposals_fold{a.fold}.json",
                         a.btxrd_vqa, a.btxrd_proposals or cf / ("btxrd" if str(a.fold) == "0" else f"btxrd_fold{a.fold}") / "proposals_btxrd_insample.json")
    ds, rows = pick_rows(data, a)
    model, _, sha = K.load_kg_arm(tag, a.fold, "cpu")
    assert sha == layer.spec["checkpoint_sha256"], "the trust file was fitted for another checkpoint"
    t0 = time.time()
    out = run(model, data, layer, ds, rows, a)
    rep = {"name": a.name, "tag": tag, "checkpoint_sha256": sha, "dataset": a.dataset, "role": a.role, "coverage": a.coverage,
           "signal": a.signal, "seconds": time.time() - t0, "rows": out}
    if a.check:
        rep["check"] = check(out, a)
    if a.out:
        dump_json(fold_dir / f"{a.out}.json", rep, exclusive=True)
    for o in out[: a.show]:
        print(f"{o['image_id']} | {o['question']}\n   {o['message_vi']}" + (f"   [đáp án: {o['truth']}]" if o.get("truth") else ""), flush=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None, help="default: the tag stored in the trust file")
    ap.add_argument("--trust", default=None, help="default: runs/joint/fold_<f>/trust_<name>.json")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--dataset", required=True, choices=["fracatlas", "btxrd"])
    ap.add_argument("--role", required=True, choices=["model_val", "calibration", "test", "external"])
    ap.add_argument("--image-id", nargs="*", default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--endpoint", default=None, choices=["fracture", "tumor", "malignancy", "diagnosis"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--coverage", type=float, default=0.8, choices=list(T.COVERAGES))
    ap.add_argument("--signal", default="head_detector", choices=list(T.SIGNALS))
    ap.add_argument("--shortlist", action="store_true", help="top-3 diagnosis list (one Qwen pass per candidate)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--proposals-file", default=None)
    ap.add_argument("--btxrd-vqa", default="/Users/uyen/Downloads/BONEVQA/yolo_pilot/datasets/btxrd_vqa")
    ap.add_argument("--btxrd-proposals", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    assert bool(a.image_id) != bool(a.endpoint), "give --image-id ... or --endpoint (with --limit)"
    main(a)
