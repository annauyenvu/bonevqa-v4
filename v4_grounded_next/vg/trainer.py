"""Shared training/evaluation plumbing for every ablation arm.

Everything that is not the arm's own model keys is taken from the sealed V4
run (read-only): model config, init checkpoint (PREtask), per-epoch batch
schedule (``schedule.json``), optimiser, warm-up + cosine LR, gradient
accumulation 4, clip 1.0, private token-mean sequence loss.  So an arm differs
from V4 only by its registry keys and its TRAIN proposal source.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from . import geometry
from .data import gt_geometry, public_batch, training_supervision
from .model import VNextGroundedModel
from .paths import CANONICAL_Q, RUNS, V4_REPORT, component_lookup, manifest_fold, read_json, sha256
from .registry import ARMS, arm_model_cfg


def v4_config(fold) -> dict:
    return read_json(V4_REPORT / f"fold_{fold}/local_real_noaux.json")


def v4_schedule(fold) -> list:
    return read_json(V4_REPORT / f"fold_{fold}/schedule.json")


def build_model(arm: str, fold, device="cpu", overrides: Optional[dict] = None, use_llm=True):
    """V4 construction protocol: set_seed(42), PREtask init, strict key accounting."""
    from src.utils import set_seed
    cfg = v4_config(fold)
    base = copy.deepcopy(cfg["model"])
    base["use_llm"] = bool(use_llm)
    mcfg = arm_model_cfg(base, arm)
    mcfg.update(copy.deepcopy(overrides or {}))
    payload = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
    set_seed(42)
    m = VNextGroundedModel(mcfg, payload["answer_vocab"], device="cpu")
    own = m.state_dict()
    state = {k: v for k, v in payload["state"].items() if k in own}
    if use_llm:
        assert set(state) == set(payload["state"]), sorted(set(payload["state"]) - set(state))[:5]
    res = m.load_state_dict(state, strict=False)
    assert not res.unexpected_keys
    m.sync_global_head()
    return m.to(device), cfg


def param_groups(model) -> Dict[str, List[str]]:
    groups = {"visual_lora": [], "llm_lora": [], "local_tokens": [], "fusion": [], "vqa_heads": [],
              "ground_new": [], "contrast_new": [], "recovery_new": [], "other": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("visual") and "lora" in n:
            g = "visual_lora"
        elif n.startswith("llm") and "lora" in n:
            g = "llm_lora"
        elif n.startswith("local_tokens."):
            g = "local_tokens"
        elif n.startswith("fusion."):
            g = "fusion"
        elif n.split(".")[0] in ("closed_head", "fracture_head", "type_head"):
            g = "vqa_heads"
        elif n.split(".")[0] in ("ground_head", "lesion_pool", "w_map"):
            g = "ground_new"
        elif n.split(".")[0] in ("contrast_img", "contrast_query"):
            g = "contrast_new"
        elif n.split(".")[0] in ("global_fracture_head", "recovery_gate"):
            g = "recovery_new"
        else:
            g = "other"
        groups[g].append(n)
    return {k: v for k, v in groups.items() if v}


class FoldData:
    """Sealed role records + cross-fitted (or in-sample) proposals for one outer fold."""

    def __init__(self, fold, train_proposals: str = "crossfit", proposals_path=None):
        from src.data.fracatlas_vqa import FracAtlasVQADataset
        from src.utils import normalize_answer
        # crossfit_eqm: cross-fit TRAIN proposals after per-inner-detector EQM score mapping
        # (scripts/eqm_calibrate.py); held-out roles are the same sealed-detector proposals.
        assert train_proposals in ("crossfit", "crossfit_eqm", "insample_v4")
        self.fold = str(fold)
        self.roles = manifest_fold(fold)
        self.ds = {r: FracAtlasVQADataset(records_path=str(p), with_masks=False, with_boxes=False)
                   for r, p in self.roles.items()}
        if proposals_path is not None:
            # Proposals of another detector (e.g. YOLOv8), same schema and same keys.  There is no
            # sealed boxes.json to check them against, so the file must name its detector instead.
            ppath = proposals_path
            P = read_json(ppath)
            assert P["schema"] == "vg-proposals-v1" and str(P["outer_fold"]) == self.fold and P.get("detector"), ppath
        else:
            ppath = RUNS / "crossfit" / f"fold_{fold}" / (f"proposals_fold{fold}_eqm.json" if train_proposals == "crossfit_eqm"
                                                         else f"proposals_fold{fold}.json")
            P = read_json(ppath)
            assert P["parity"]["max_abs_diff"] == 0.0 and P["parity"]["count_mismatch"] == 0
        self.proposals_sha256 = sha256(ppath)
        self.props = {"train": P["train_insample_audit_only" if train_proposals == "insample_v4" else "train"]}
        self.source = {"train": "detector_full" if train_proposals == "insample_v4" else "detector_cf"}
        for r in ("model_val", "calibration", "test"):
            self.props[r] = P[r]
            self.source[r] = "detector_full"
        for r, ds in self.ds.items():
            missing = {x["image_id"] for x in ds.records} - set(self.props[r])
            assert not missing, (r, sorted(missing)[:3])
        self.lookup = component_lookup()
        self.table = {(x["question"], normalize_answer(x["answer"])): 1.0
                      for x in self.ds["train"].records if x["answer_type"] == "OPEN"}
        self._sup: Dict[str, tuple] = {}

    def _samples(self, role, idxs):
        ds = self.ds[role]
        samples = [ds[i] for i in idxs]
        return [ds.records[i] for i in idxs], [s["image"] for s in samples]

    def _supervision(self, records, images, cfg, seed=0):
        """Per-image cache: gt mask grid and hard-negative crops are deterministic."""
        gm, cc = [], []
        for r, im in zip(records, images):
            key = r["image_id"]
            if key not in self._sup:
                s = training_supervision([r], [im], cfg, seed=seed)
                self._sup[key] = (s["gt_masks"][0], s["contrast_crops"][0])
            gm.append(self._sup[key][0]); cc.append(self._sup[key][1])
        return {"gt_masks": gm, "contrast_crops": cc}

    def train_batch(self, idxs, cfg, supervision=True):
        records, images = self._samples("train", idxs)
        b = public_batch(records, images, self.props["train"], self.source["train"], self.lookup)
        b.update(answers=[r["answer"] for r in records], answer_types=[r["answer_type"] for r in records])
        if supervision:
            b.update(self._supervision(records, images, cfg))
        return b

    def eval_batch(self, role, idxs):
        records, images = self._samples(role, idxs)
        return records, images, public_batch(records, images, self.props[role], self.source[role], self.lookup)

    def canonical_indices(self, role):
        return [i for i, r in enumerate(self.ds[role].records) if r["question"] == CANONICAL_Q]


def lr_lambda(total: int):
    warmup = int(0.05 * total)

    def rate(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return rate


READOUT_PREFIXES = ("les_head.", "lesion_pool.token.", "lesion_pool.emphasis.")
READOUT_SCOPES = {"all": READOUT_PREFIXES, "h_les": ("les_head.",)}


def make_optimizer(model, lr, readout_lr_mult: float = 1.0, readout_scope: str = "all"):
    """V4 AdamW.  ``readout_lr_mult`` > 1 gives the zero-initialised M_hat read-out
    (h_les, t_les, emphasis) its own group at lr * mult; schedule, betas, weight decay
    and the global clip are unchanged.  mult == 1 is exactly the single-group V4 optimiser."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if readout_lr_mult == 1.0:
        return params, torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    pref = READOUT_SCOPES[readout_scope]  # 'h_les': only h_les gets lr * mult; t_les / emphasis stay at lr
    ro = [p for n, p in named if n.startswith(pref)]
    rest = [p for n, p in named if not n.startswith(pref)]
    assert ro, "no read-out parameters (arm without grounding?)"
    return params, torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": ro, "lr": lr * readout_lr_mult}],
                                     lr=lr, betas=(0.9, 0.98), weight_decay=0.01)


@torch.no_grad()
def evaluate_canonical(model, data: FoldData, role="model_val", batch_size=8, limit=None):
    """Fracture endpoint + grounding on canonical rows.  GT is read here, by the
    metric, after inference -- never passed to the model."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    idx = data.canonical_indices(role)[: limit or None]
    rows = []
    for off in range(0, len(idx), batch_size):
        records, images, b = data.eval_batch(role, idx[off:off + batch_size])
        r = model.infer(b)
        for j, (rec, im) in enumerate(zip(records, images)):
            row = {"image_id": rec["image_id"], "y": int(bool(rec["fractured"])),
                   "m": bool(len(b["proposals"][j])), "p_final": float(r["p_final"][j]),
                   "p_local": float(r["p_local"][j]),
                   "p_global": None if r["p_global"] is None else float(r["p_global"][j]),
                   "alpha": None if r["alpha"] is None else float(r["alpha"][j])}
            if rec["fractured"] and r["m_hat"] is not None:
                boxes, polys, size = gt_geometry(rec)
                mask = geometry.rasterize(polys, size, boxes)
                g = model.cfg["vg_grid"]
                gt = geometry.mask_to_grid(mask, size, model.cfg["image_size"], g).reshape(-1)
                ell = r["core"]["ell"][j, 0].detach().float().cpu().reshape(-1)
                w = torch.softmax(ell, 0).numpy()
                row.update(point_hit=bool(gt[int(ell.argmax())] > 0), energy_in_gt=float((w * (gt > 0)).sum()),
                           gt_area_frac=float((gt > 0).mean()))
            rows.append(row)
    y = np.array([r["y"] for r in rows]); p = np.array([r["p_final"] for r in rows])
    out = {"n": len(rows), "prevalence": float(y.mean())}
    if 0 < y.sum() < len(y):
        out.update(auprc=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)))
        for name, sel in (("m1", np.array([r["m"] for r in rows])), ("m0", ~np.array([r["m"] for r in rows]))):
            pos = (y == 1) & sel
            out[f"sens@0.5_{name}"] = float((p[pos] >= 0.5).mean()) if pos.any() else None
            out[f"n_pos_{name}"] = int(pos.sum())
        out["spec@0.5"] = float((p[y == 0] < 0.5).mean())
    g = [r for r in rows if "point_hit" in r]
    if g:
        out["pointing_game"] = float(np.mean([r["point_hit"] for r in g]))
        out["energy_in_gt"] = float(np.mean([r["energy_in_gt"] for r in g]))
        out["energy_chance"] = float(np.mean([r["gt_area_frac"] for r in g]))
        gm = [r for r in g if r["m"]]
        out["pointing_game_m1"] = float(np.mean([r["point_hit"] for r in gm])) if gm else None
    return out, rows
