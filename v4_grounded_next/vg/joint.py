"""FracAtlas + BTXRD in one training stream, without touching the V4 model.

The FracAtlas side is :class:`vg.trainer.FoldData`, unchanged: sealed role records, the
sealed per-epoch batch order, the same proposal files.  BTXRD adds its own QA records
(yolo_pilot/code/make_btxrd_vqa.py) and its own proposal file (lesion-class boxes).

What the model does with a BTXRD row, with no new parameters:
  * CLOSED rows are yes / no, both in the V4 answer vocabulary -> ClosedHead cross-entropy;
  * OPEN rows -> Qwen sequence loss, as for FracAtlas OPEN rows;
  * the fracture head never sees them: BTXRD has no fracture label and none of its
    questions is the canonical fracture question.
Proposals are question-routed: BTXRD rows carry lesion-class boxes, FracAtlas rows carry
fracture-class boxes.

TRAIN index space: 0 .. n_fa-1 are FracAtlas rows (the sealed indices), n_fa .. are BTXRD
rows.  A microbatch never mixes the two sources, so every FracAtlas microbatch is the sealed
one; BTXRD microbatches are interleaved one for one.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .data import public_batch
from .paths import ROOT, read_json, sha256
from .trainer import FoldData

BTXRD_ROOT = ROOT / "yolo_pilot/data_raw/btxrd/BTXRD"
BT_ROLE = {"train": "train", "model_val": "val", "test": "test", "external": "external"}
ENDPOINTS = ("tumor", "malignancy", "diagnosis")


class JointFoldData(FoldData):
    def __init__(self, fold, train_proposals, proposals_path, btxrd_vqa, btxrd_proposals,
                 btxrd_train_key: str = "train_insample_audit_only"):
        super().__init__(fold, train_proposals, proposals_path)
        from src.data.fracatlas_vqa import FracAtlasVQADataset
        from src.utils import normalize_answer
        P = read_json(btxrd_proposals)
        assert P["schema"] == "vg-proposals-v1" and P.get("dataset") == "btxrd" and P.get("detector"), btxrd_proposals
        assert btxrd_train_key in P, f"{btxrd_train_key} missing in {btxrd_proposals}"
        self.btxrd_proposals_sha256 = sha256(btxrd_proposals)
        self.bt_source = {"train": "detector_full" if btxrd_train_key == "train_insample_audit_only" else "detector_cf",
                          "model_val": "detector_full", "test": "detector_full", "external": "detector_full"}
        self.bt, self.bt_props = {}, {}
        for role, short in BT_ROLE.items():
            path = Path(btxrd_vqa) / f"vqa_{short}.json"
            # The FracAtlas dataset class is a plain record reader once records_path and root are given.
            self.bt[role] = FracAtlasVQADataset(records_path=str(path), root=BTXRD_ROOT, with_masks=False, with_boxes=False)
            self.bt_props[role] = P[btxrd_train_key if role == "train" else role]
            recs = self.bt[role].records
            assert all(r["dataset"] == "btxrd" and r["role"] == short and r["fractured"] is None for r in recs), path
            missing = {r["image_id"] for r in recs} - set(self.bt_props[role])
            assert not missing, (role, sorted(missing)[:3])
        self.btxrd_vqa_sha256 = {role: sha256(Path(btxrd_vqa) / f"vqa_{short}.json") for role, short in BT_ROLE.items()}
        fa_ids = {r["image_id"] for ds in self.ds.values() for r in ds.records}
        bt_ids = {r["image_id"] for ds in self.bt.values() for r in ds.records}
        assert not fa_ids & bt_ids, "image id collision between FracAtlas and BTXRD"
        # Pseudo-patient groups as component ids, numbered after the FracAtlas components.
        groups = sorted({(r["center"], r["group"]) for ds in self.bt.values() for r in ds.records})
        gid = {g: 10_000_000 + k for k, g in enumerate(groups)}
        for ds in self.bt.values():
            for r in ds.records:
                self.lookup[r["image_id"]] = gid[(r["center"], r["group"])]
        for r in self.bt["train"].records:
            if r["answer_type"] == "OPEN":
                self.table[(r["question"], normalize_answer(r["answer"]))] = 1.0
        self.n_fa = len(self.ds["train"].records)
        self.n_bt = len(self.bt["train"].records)

    # ------------------------------------------------------------------ training
    def joint_schedule(self, fa_schedule: List[dict], seed: int, batch_size: int = 4, grad_accum: int = 4) -> List[dict]:
        """Sealed FracAtlas microbatches, in their sealed order, with one BTXRD microbatch after each.
        BTXRD rows are shuffled per epoch with their own generator, so the FracAtlas order never changes."""
        out = []
        for e, s in enumerate(fa_schedule):
            fa = [list(b) for b in s["ordered_indices"]]
            order = list(range(self.n_fa, self.n_fa + self.n_bt))
            random.Random(f"btxrd:{seed}:{e}").shuffle(order)
            bt = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            batches = []
            for k in range(max(len(fa), len(bt))):
                if k < len(fa):
                    batches.append(fa[k])
                if k < len(bt):
                    batches.append(bt[k])
            out.append({"epoch": e, "ordered_indices": batches, "batches": len(batches),
                        "updates": -(-len(batches) // grad_accum), "fracatlas_batches": len(fa), "btxrd_batches": len(bt)})
        return out

    def train_batch(self, idxs, cfg, supervision=True):
        if all(i < self.n_fa for i in idxs):
            return super().train_batch(idxs, cfg, supervision)
        assert all(i >= self.n_fa for i in idxs), "a microbatch must not mix FracAtlas and BTXRD rows"
        ds = self.bt["train"]
        local = [i - self.n_fa for i in idxs]
        records, images = [ds.records[i] for i in local], [ds[i]["image"] for i in local]
        b = public_batch(records, images, self.bt_props["train"], self.bt_source["train"], self.lookup)
        b.update(answers=[r["answer"] for r in records], answer_types=[r["answer_type"] for r in records])
        if supervision:  # no fracture mask and no contrast crops exist for BTXRD
            res = cfg["vg_loss_res"]
            b.update(gt_masks=[np.zeros((res, res), dtype=np.float32) for _ in records], contrast_crops=[None] * len(records))
        return b

    # ---------------------------------------------------------------- evaluation
    def bt_indices(self, role, endpoint):
        return [i for i, r in enumerate(self.bt[role].records) if r["canonical"] == endpoint]

    def bt_eval_batch(self, role, idxs):
        ds = self.bt[role]
        records, images = [ds.records[i] for i in idxs], [ds[i]["image"] for i in idxs]
        return records, public_batch(records, images, self.bt_props[role], self.bt_source[role], self.lookup)


def _binary_metrics(y, p, m) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p, m = np.asarray(y), np.asarray(p), np.asarray(m, dtype=bool)
    out = {"n": int(len(y)), "prevalence": float(y.mean())}
    if 0 < y.sum() < len(y):
        pos, neg = y == 1, y == 0
        out.update(auprc=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)),
                   **{"sens@0.5": float((p[pos] >= .5).mean()), "spec@0.5": float((p[neg] < .5).mean()),
                      "accuracy@0.5": float(((p >= .5) == (y == 1)).mean())},
                   m_rate_pos=float(m[pos].mean()), m_rate_neg=float(m[neg].mean()),
                   auroc_of_proposal_flag=float(roc_auc_score(y, m.astype(float))))
    return out


@torch.no_grad()
def evaluate_btxrd(model, data: JointFoldData, role="model_val", batch_size=8, limit=None, endpoints=ENDPOINTS):
    """The three BTXRD endpoints.  tumor / malignancy: P(yes) = yes / (yes + no) of the closed head.
    diagnosis: greedy Qwen generation, exact match after answer normalisation.  Labels are read here,
    after inference; the model only receives images, questions and detector proposals."""
    from src.utils import normalize_answer
    model.eval()
    yes, no = model.answer_to_idx["yes"], model.answer_to_idx["no"]
    metrics, rows_out = {}, {}
    for ep in endpoints:
        idx = data.bt_indices(role, ep)[: limit or None]
        rows = []
        for off in range(0, len(idx), batch_size):
            records, b = data.bt_eval_batch(role, idx[off:off + batch_size])
            r = model.infer(b)
            if ep == "diagnosis":
                gen = model.generate_open(r["core"], r["core"]["q_texts"])
                for rec, props, g in zip(records, b["proposals"], gen):
                    rows.append({"image_id": rec["image_id"], "gt": rec["answer"], "pred": g["text"], "m": bool(len(props)),
                                 "correct": normalize_answer(g["text"]) == normalize_answer(rec["answer"])})
            else:
                pr = torch.softmax(r["logits_closed"], dim=-1)
                p_yes = (pr[:, yes] / (pr[:, yes] + pr[:, no]).clamp_min(1e-9)).tolist()
                for rec, props, p in zip(records, b["proposals"], p_yes):
                    rows.append({"image_id": rec["image_id"], "y": int(rec["answer"] == "yes"), "p_yes": float(p),
                                 "m": bool(len(props))})
        rows_out[ep] = rows
        if not rows:
            metrics[ep] = {"n": 0}
        elif ep == "diagnosis":
            classes = sorted({x["gt"] for x in rows})
            recall = {c: float(np.mean([x["correct"] for x in rows if x["gt"] == c])) for c in classes}
            f1 = []
            for c in classes:
                tp = sum(x["correct"] for x in rows if x["gt"] == c)
                pred_c = sum(normalize_answer(x["pred"]) == normalize_answer(c) for x in rows)
                gt_c = sum(x["gt"] == c for x in rows)
                f1.append(2 * tp / (pred_c + gt_c) if pred_c + gt_c else 0.0)
            metrics[ep] = {"n": len(rows), "accuracy": float(np.mean([x["correct"] for x in rows])),
                           "balanced_accuracy": float(np.mean(list(recall.values()))), "macro_f1": float(np.mean(f1)),
                           "recall_by_class": recall, "n_by_class": {c: sum(x["gt"] == c for x in rows) for c in classes},
                           "answers_outside_label_set": int(sum(normalize_answer(x["pred"]) not in
                                                                {normalize_answer(c) for c in classes} for x in rows))}
        else:
            metrics[ep] = _binary_metrics([x["y"] for x in rows], [x["p_yes"] for x in rows], [x["m"] for x in rows])
    return metrics, rows_out
