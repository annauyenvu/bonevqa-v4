"""Scores needed to test "reference information" (age, sex, site) on the BTXRD endpoints.  Inference only.

    source v4_grounded_next/env.sh
    $PY -m scripts.reference_score --name J_v4cf_yolojoint_btxrd --tag joint_fold0/V4-CF_yolojoint_btxrd --role test \
        --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0.json \
        --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json

For every image of the role and each wording of the question:
  plain          the question as trained
  age_sex        "The patient is a 14-year-old female. <question>"  (the wording of the 1 % clinical-context TRAIN rows)
  age_sex_site   "... female with a lesion in the tibia. <question>"  (never seen in training; tumour images only)
it stores
  tumor, malignancy   P(yes) of the closed head
  diagnosis           the log-likelihood Qwen gives to each of the candidate diagnoses (the TRAIN answer set),
                      so that a prior over diagnoses can be combined with the image evidence afterwards
together with the image's metadata.  Nothing is fitted here; scripts/reference_analyze.py does the analysis.
Writes runs/joint/fold_<f>/ref_<name>_<role>.json.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from vg import kg as K
from vg.data import public_batch
from vg.joint import JointFoldData
from vg.paths import RUNS, dump_json

META = ("image_id", "center", "group", "age", "gender", "region", "views", "bones", "joints", "tumor", "malignant")


def reword(rec, variant):
    q = rec["question"]
    if variant == "plain":
        return q
    age = int(rec["age"])
    head = f"The patient is {'an' if age in (8, 11, 18) or 80 <= age < 90 else 'a'} {age}-year-old {rec['gender']}"
    if variant == "age_sex":
        return f"{head}. {q}"
    sites = list(rec["bones"]) + [j.replace("-", " ") for j in rec["joints"]]
    assert sites, rec["image_id"]
    where = sites[0] if len(sites) == 1 else ", ".join(sites[:-1]) + " and " + sites[-1]
    return f"{head} with a lesion in the {where}. {q}"


@torch.no_grad()
def candidate_loglik(model, core, q_texts, candidates):
    """(B, C) sum of token log-probabilities of each candidate answer, teacher-forced."""
    prefix = model._llm_prefix(core)
    b = prefix.shape[0]
    out = torch.zeros(b, len(candidates))
    for j, cand in enumerate(candidates):
        emb, att, labels = model._build_llm_inputs(prefix, q_texts, [cand] * b)
        logits = model.llm(inputs_embeds=emb, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        tok = logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        out[:, j] = (tok * mask).sum(1).cpu()
    return out


@torch.no_grad()
def main(a):
    torch.set_num_threads(a.threads)
    out = RUNS / "joint" / f"fold_{a.fold}" / f"ref_{a.name}_{a.role}.json"
    assert not out.exists(), out
    data = JointFoldData(a.fold, "insample_v4", a.proposals_file, a.btxrd_vqa, a.btxrd_proposals)
    candidates = sorted({r["answer"] for r in data.bt["train"].records if r.get("canonical") == "diagnosis"})
    model, started, sha = K.load_kg_arm(a.tag, a.fold, "cpu")
    yes, no = model.answer_to_idx["yes"], model.answer_to_idx["no"]
    ds = data.bt[a.role]
    t0 = time.time()
    res = {"name": a.name, "tag": a.tag, "role": a.role, "checkpoint_sha256": sha, "kg": started["model_cfg"].get("kg"),
           "candidates": candidates, "endpoints": {}}
    for ep, variants in (("tumor", ("plain", "age_sex")), ("malignancy", ("plain", "age_sex", "age_sex_site")),
                         ("diagnosis", ("plain", "age_sex", "age_sex_site"))):
        idx = data.bt_indices(a.role, ep)[: a.limit or None]
        rows = [{**{k: ds.records[i][k] for k in META}, "truth": ds.records[i]["answer"]} for i in idx]
        for variant in variants:
            for off in range(0, len(idx), a.batch_size):
                chunk = idx[off:off + a.batch_size]
                records = [dict(ds.records[i], question=reword(ds.records[i], variant)) for i in chunk]
                images = [ds[i]["image"] for i in chunk]
                b = public_batch(records, images, data.bt_props[a.role], data.bt_source[a.role], data.lookup)
                r = model.infer(b)
                if ep == "diagnosis":
                    ll = candidate_loglik(model, r["core"], r["core"]["q_texts"], candidates)
                    for k in range(len(chunk)):
                        rows[off + k][variant] = [round(float(x), 4) for x in ll[k]]
                else:
                    pr = torch.softmax(r["logits_closed"], dim=-1)
                    p = pr[:, yes] / (pr[:, yes] + pr[:, no]).clamp_min(1e-9)
                    for k in range(len(chunk)):
                        rows[off + k][variant] = float(p[k])
                for k in range(len(chunk)):
                    rows[off + k]["has_proposal"] = bool(len(b["proposals"][k]))
        res["endpoints"][ep] = rows
    res["seconds"] = time.time() - t0
    dump_json(out, res, exclusive=True)
    print(json.dumps({"written": str(out), "seconds": res["seconds"], "n": {k: len(v) for k, v in res["endpoints"].items()}}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--role", required=True, choices=["model_val", "test", "external"])
    ap.add_argument("--fold", default="0")
    ap.add_argument("--proposals-file", required=True)
    ap.add_argument("--btxrd-vqa", required=True)
    ap.add_argument("--btxrd-proposals", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args())
