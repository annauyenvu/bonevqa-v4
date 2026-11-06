"""BoneVQA-Prompt v5 study runner: screening and confirmation with a job queue.

Usage (from /Users/uyen/Downloads/BONEVQA, env as in accel_v1/run_all_v3.sh):
  python refactor_v2/experiments/v5/study_v5.py plan <suite>
  python refactor_v2/experiments/v5/study_v5.py worker <suite> --device mps   # and/or --device cpu
  python refactor_v2/experiments/v5/study_v5.py analyze <suite>

Design rules carried over from the sealed v4 study (local_noaux_refinement_v1):
* same per-fold PRE initialisation, TRAIN vocabulary, component batch schedule,
  token-mean sequence loss adapter, AdamW 1e-4, batch 4 x accum 4, 2 epochs,
  fixed LAST checkpoint, no test-based selection;
* same fold detector proposals (score 0.05, top 3) re-used by sha256;
* every arm of a suite runs on ONE device (comparisons never mix devices);
* each job: 2-update gradient gate -> train -> evaluate -> fresh-process replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
R = ROOT / "refactor_v2/reports"
V4 = R / "local_noaux_refinement_v1"
OUT = R / "v5_studies"
MASKS = ROOT / ".evaluation_runtime/v5_masks"
CAN = "Is there a fracture in this X-ray?"
# Device policy (accel_v1, torch 2.6 qualification 2026-10-30): MPS training matches CPU
# loss/gradients (|dloss| 3e-6) and fracture probabilities (2.4e-7) but greedy OPEN
# generation on MPS is corrupted.  Therefore training may run on MPS, while ALL
# evaluation, generation and replay run on CPU.
EVAL_DEVICE = "cpu"

# --------------------------------------------------------------------- arms
# v5 is a factorial ablation of prompt TYPES, not of each paper's internals.  Each factor
# uses the best configuration its source paper reports, adapted only where the bone domain
# requires it (bone knowledge graph instead of SLAKE; region names for Lens alignment).
#
#   V  FAVP (AAAI 2025): AVPC SAM-Med2D, 50 keypoints, NMS 0.80 (Fig. 4), author filters
#      0.68/0.70; circle prompt (Table 4, best on VQA-RAD radiographs); HAG = local+global
#      ViT tokens -> Q-Former -> LLM; ViT LoRA r4, LLM LoRA r8 (Fig. 3, Table 3).
#   L  Localization Lens (MICCAI 2025), Table 3 best row: SAM + MedSAM segmentation,
#      unicolor/multicolor/masked views (2 random + original), DCL phase-1 alignment to the
#      organ/region name, InfoNCE QA phase, pixel shuffle r = 4, original-image inference.
#   P  LaPA (CVPRW 2024): GM + LF (L->I->MM) + PF, latent 32, eta 0.1, alpha 1, theta = beta
#      = 0.1 (Tables 2-5, Fig. 3); PF graph = bone graph built from TRAIN (SLAKE is off-domain).
#   T  textual prompt: fixed clinical instruction, no region/label words (no answer leakage).
#
# Region/mask sources (paper settings by default; see SOURCE_ALTERNATIVES in the report):
V_REGIONS = "avpc"                       # paper: SAM-Med2D AVPC.  Alternative: "detector" (v4)
# Manuscript deadline 2026-11-01: SAM-H stopped at 753/4013 images, so MedSAM only (deviation from the paper).
L_MASKS = ["medsam"]                      # paper: ["sam_huge", "medsam"]
AVPC_SETTING = {"points_per_side": 50, "nms": 0.80}

FACTORS = {
    "V": dict(model={"v5_local": True, "v5_hag": "replace", "v5_local_view": "circle"}),
    "L": dict(model={"v5_lens": True, "v5_lens_shuffle_r": 4, "v5_lens_weight": 0.1, "v5_lens_phase1_loss": "dcl"},
              lens={"sources": L_MASKS, "lesion": False}),
    "P": dict(model={"v5_latent": True, "v5_latent_mode": "lapa", "v5_eta": 0.1, "v5_n_latent": 32, "v5_pf": "graph"},
              kg="real"),
    "T": dict(model={"use_textual_prompt": True, "textual_prompt_style": "clinical", "use_region_in_prompt": False}),
}


def _combo(letters):
    model, spec = {}, {"proposals": V_REGIONS if "V" in letters else "none"}
    for x in letters:
        model.update(FACTORS[x]["model"])
        for k in ("lens", "kg"):
            if k in FACTORS[x]:
                spec[k] = FACTORS[x][k]
    spec["model"] = model
    return spec


FACTORIAL = ["B"] + ["".join(c) for n in range(1, 5) for c in __import__("itertools").combinations("VLPT", n)]
ARMS = {name: _combo("" if name == "B" else name) for name in FACTORIAL}
# Reference: the sealed v4 local route (detector crops, no HAG) re-run on the suite device.
ARMS["V4"] = dict(model={"v5_local": True}, proposals="detector")
# Retrained controls for the IEEE Access study (reports/access_v1/PROTOCOL.md).
ARMS["V4_flag"] = dict(model={"v5_local": True, "v5_local_content": "slot_constant"}, proposals="detector")
ARMS["V4_grid"] = dict(model={"v5_local": True}, proposals="grid")
ARMS["V4_placebo"] = dict(model={"v5_local": True}, proposals="placebo")

SUITES = {
    # 2^4 factorial screen on fold 0 (MPS training, CPU evaluation), decisions on model_val
    "factorial_fold0": dict(folds=["0"], seeds=[42], device="mps",
                            arms=["B", "T", "P", "PT", "V4", "V", "VT", "VP", "VPT",
                                  "L", "LT", "LP", "LPT", "VL", "VLT", "VLP", "VLPT"]),
    # confirmation: edit arms after the screen (FINAL = best passing arm)
    "confirm": dict(folds=["0", "1", "2", "3", "4"], seeds=[42, 43, 44], device="mps", arms=["B", "V4", "FINAL"]),
    "access_v1": dict(folds=["0", "1", "2", "3", "4"], seeds=[43], device="mps", arms=["V4", "V4_flag", "B", "V4_grid", "V4_placebo"]),
    # Part E: fold 0 of these arms already exists in factorial_fold0 with the same recipe and seed.
    "access_v1_methods": dict(folds=["1", "2", "3", "4"], seeds=[42], device="mps", arms=["P", "L", "V"]),
}


# ------------------------------------------------------------------ helpers
def read(p):
    return json.loads(Path(p).read_text())


def dump(p, v, exclusive=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with p.open("x") as f:
            json.dump(v, f, indent=2)
    else:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(v, indent=2))
        tmp.replace(p)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def guards(start=None, cap=12 * 3600):
    import shutil

    assert shutil.disk_usage(ROOT).free >= 10 * 1024 ** 3, "disk reserve 10 GiB"
    if start is not None:
        assert time.time() - start < cap, "wall budget"


def arm_spec(name):
    if name in ARMS:
        return ARMS[name]
    final = OUT / "FINAL_ARM.json"
    if name == "FINAL" and final.exists():
        return ARMS[read(final)["arm"]]
    raise KeyError(name)


def fold_meta(fold):
    return read(V4 / "metadata.json")[fold]


# ------------------------------------------------------------------- config
def build_cfg(fold, arm, seed, suite):
    from experiments.v5.bone_kg_v5 import build_bone_kg, degree_preserving_rewire

    spec = arm_spec(arm)
    parent = read(V4 / f"fold_{fold}/B.json")
    cfg = copy.deepcopy(parent)
    m = {k: v for k, v in cfg["model"].items() if not k.startswith("vnext_")}
    m.update(spec["model"])
    if spec.get("kg"):
        train = read(fold_meta(fold)["roles"]["train"]["path"])
        kg = build_bone_kg(train)
        m["v5_kg"] = kg if spec["kg"] == "real" else degree_preserving_rewire(kg, seed=20261030 + int(fold))
    cfg["model"] = m
    cfg["seed"] = seed
    cfg["run_name"] = f"v5_{suite}_fold{fold}_{arm}_seed{seed}"
    cfg["v5_arm"] = arm
    cfg["v5_spec"] = {k: v for k, v in spec.items() if k != "model"}
    return cfg


def proposals_for(fold, kind):
    """Return (boxes dict, provenance) for a proposal kind; all sha-checked."""
    folder = V4 / f"fold_{fold}"
    if kind == "none":
        return None, {"kind": "none"}
    if kind in ("detector", "placebo"):
        receipt = read(folder / "proposal_receipt.json")
        name = "boxes.json" if kind == "detector" else "placebo.json"
        key = "boxes_sha256" if kind == "detector" else "placebo_sha256"
        assert sha(folder / name) == receipt[key], f"{name} hash mismatch"
        return read(folder / name), {"kind": kind, "path": str(folder / name), "sha256": receipt[key]}
    if kind == "grid":
        # Detector-free control: fixed 2x2 quadrant crops for EVERY image (same for
        # positives/negatives).  Tests whether gains come from extra resolution alone.
        info = read(folder / "proposal_receipt.json")["images"]
        boxes = {}
        for iid, v in info.items():
            w, h = v["size"]
            boxes[iid] = [[0, 0, w / 2, h / 2], [w / 2, 0, w, h / 2], [0, h / 2, w / 2, h]]
        return boxes, {"kind": "grid", "rule": "3 fixed quadrants (top-left, top-right, bottom-left)"}
    if kind == "avpc":
        d = MASKS / f"avpc_p{AVPC_SETTING['points_per_side']}_nms{AVPC_SETTING['nms']}"
        receipt = read(d / "receipt.json")
        assert sha(d / "boxes.json") == receipt["boxes_sha256"]
        assert receipt["images"] >= 4013, "AVPC proposals incomplete"
        return read(d / "boxes.json"), {"kind": "avpc", "path": str(d), "sha256": receipt["boxes_sha256"], **AVPC_SETTING}
    raise ValueError(kind)


# ------------------------------------------------------------------- model
def construct(cfg, device):
    import torch
    from src.utils import set_seed
    from experiments.v5.model_v5 import BoneVQAv5

    payload = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
    set_seed(int(cfg["seed"]))
    m = BoneVQAv5(cfg["model"], payload["answer_vocab"], device="cpu")
    result = m.load_state_dict(payload["state"], strict=False)
    assert not result.unexpected_keys
    current = m.trainable_state_dict()
    for k, t in payload["state"].items():
        assert torch.equal(t, current[k]), k
    new = {k for k in current if k not in payload["state"]}
    allowed = ("local_tokens.", "gamma_hag", "v5_")
    assert all(k.startswith(allowed) for k in new), sorted(k for k in new if not k.startswith(allowed))[:5]
    return m.to(device)


def datasets(fold):
    from src.data.fracatlas_vqa import FracAtlasVQADataset

    meta = fold_meta(fold)
    return {r: FracAtlasVQADataset(records_path=b["path"], with_masks=False, with_boxes=False) for r, b in meta["roles"].items()}


def _box_masks(boxes, size):
    from src.data.fracatlas_vqa import FracAtlasVQADataset

    return FracAtlasVQADataset._boxes_to_masks(boxes, size)


def lens_mask_loader(fold, spec):
    """TRAIN-only lens masks: anatomy (all images) + optional lesion views."""
    from experiments.v5.masks_v5 import load_masks

    lens = spec.get("lens")
    if not lens:
        return None
    train = read(fold_meta(fold)["roles"]["train"]["path"])
    gt = {}
    for r in train:
        if r.get("boxes"):
            gt.setdefault(r["image_id"], ([b[:4] for b in r["boxes"]], (r["width"], r["height"])))
    donors = sorted(gt)

    def load(iid, image):
        parts = []
        for src in lens["sources"]:
            p = MASKS / f"anatomy_{src}" / f"{iid}.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing anatomy mask {p}; run masks_v5.py anatomy --source {src}")
            m = load_masks(p)
            if len(m):
                from src.data.prompts import _fit_masks
                parts.append(_fit_masks(m, image.size))
        if lens.get("lesion"):
            if iid in gt:
                parts.append(_box_masks(gt[iid][0], image.size))
            else:  # label-blind placebo: same geometry distribution as positives
                b, size = gt[donors[int(hashlib.md5(iid.encode()).hexdigest(), 16) % len(donors)]]
                sx, sy = image.size[0] / size[0], image.size[1] / size[1]
                parts.append(_box_masks([[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in b], image.size))
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts, 0)[:8] if parts else None

    return load


def batcher(fold, spec, training_roles=("train",)):
    from src.context_controls import load_component_lookup
    from src.data import collate_fn

    ds = datasets(fold)
    parent = read(V4 / f"fold_{fold}/B.json")
    lookup, _ = load_component_lookup(parent["train"]["component_batching"]["manifest"],
                                      parent["train"]["component_batching"]["seal"], ROOT)
    boxes, _ = proposals_for(fold, spec["proposals"])
    lens = lens_mask_loader(fold, spec)

    def collate(samples, role="train"):
        b = collate_fn(samples)
        b["component_ids"] = [lookup[i] for i in b["image_ids"]]
        b["boxes"] = [None] * len(samples)
        b["masks"] = [None] * len(samples)
        if boxes is not None:
            b["boxes"] = [boxes.get(i) or None for i in b["image_ids"]]
            b["masks"] = [_box_masks(boxes.get(i) or [], im.size) for i, im in zip(b["image_ids"], b["images"])]
        if lens is not None and role in training_roles:
            b["lens_masks"] = [lens(i, im) for i, im in zip(b["image_ids"], b["images"])]
        return b

    return ds, collate


def public(b, spec):
    keys = ("images", "questions") if spec["proposals"] == "none" else ("images", "questions", "boxes", "masks")
    return {k: b[k] for k in keys}


def predict(model, ds, collate, spec, role, canonical=False, limit=None, progress=None):
    import torch

    expected = ds[role].records
    idx = [i for i, e in enumerate(expected) if not canonical or e["question"] == CAN]
    if limit:
        idx = idx[:limit]
    rows, start = [], time.time()
    model.eval()
    with torch.no_grad():
        for off in range(0, len(idx), 4):
            guards()
            ids = idx[off: off + 4]
            b = collate([ds[role][i] for i in ids], role=role)
            for i, p in zip(ids, model.predict_batch(public(b, spec))):
                e = expected[i]
                row = {"image_id": e["image_id"], "question": e["question"], "gt": e["answer"],
                       "answer_type": e["answer_type"], "pred": p["answer"]}
                if "p_yes" in p:
                    row["p_yes"] = float(p["p_yes"])
                    assert math.isfinite(row["p_yes"]) and 0 <= row["p_yes"] <= 1
                if e["question"] == CAN:
                    assert "p_yes" in row
                rows.append(row)
            if progress and off % 80 == 0:
                dump(progress, {"role": role, "done": len(rows), "total": len(idx), "seconds": time.time() - start})
    return rows


def schedule_for(fold, seed):
    s = read(V4 / f"fold_{fold}/schedule.json")
    if seed == 42:
        return s
    out = []
    for e in s:  # seed-specific order of the same component-aware batches
        rng = random.Random(seed * 1000 + int(e.get("epoch", 0)))
        batches = list(e["ordered_indices"])
        rng.shuffle(batches)
        out.append({**e, "ordered_indices": batches})
    return out


# -------------------------------------------------------------------- job
def job_dir(suite, fold, arm, seed):
    return OUT / suite / f"fold_{fold}" / f"{arm}_seed{seed}"


def run_job(suite, fold, arm, seed, device):
    import torch
    from torch.utils.data import DataLoader
    from src.train import seed_training_epoch
    from src.utils import compute_metrics, normalize_answer
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

    torch.set_num_threads(8 if device == "cpu" else MPS_JOB_THREADS)
    if device == "mps":
        assert torch.backends.mps.is_available() and torch.__version__.startswith("2.6"), "MPS qualified only for torch 2.6"
    folder = job_dir(suite, fold, arm, int(seed))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completed.json").exists():
        return
    seed = int(seed)
    spec = arm_spec(arm)
    cfg = build_cfg(fold, arm, seed, suite)
    _, prov = proposals_for(fold, spec["proposals"])
    dump(folder / "config.json", {"cfg": cfg, "proposals": prov, "device": device, "torch": torch.__version__})
    if not (folder / "gate.json").exists():   # before building the training model (memory)
        gate(suite, fold, arm, seed, device, folder)
        import gc
        gc.collect()
    model = construct(cfg, device)
    ds, collate = batcher(fold, spec)
    table = {(r["question"], normalize_answer(r["answer"])): 1.0 for r in ds["train"].records if r["answer_type"] == "OPEN"}
    params = [p for p in model.parameters() if p.requires_grad]
    start = time.time()

    # ---- optional Localization Lens phase 1 (region-text DCL), one TRAIN pass
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    if spec.get("lens"):
        seed_training_epoch(seed, 100)
        model.train()
        seen_img, batch_rows = set(), []
        for i, r in enumerate(ds["train"].records):
            if r["image_id"] not in seen_img:
                seen_img.add(r["image_id"])
                batch_rows.append(i)
        random.Random(seed).shuffle(batch_rows)
        p1 = []
        for j in range(0, len(batch_rows), 8):
            guards(start)
            b = collate([ds["train"][i] for i in batch_rows[j:j + 8]])
            loss = model.forward_lens_align(b)["loss"]
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            p1.append(float(loss.detach()))
            if len(p1) % 20 == 0:
                dump(folder / "progress.json", {"phase": "lens_region_alignment", "steps": len(p1), "loss": p1[-1]})
        dump(folder / "lens_phase1.json", {"steps": len(p1), "mean_loss": float(np.mean(p1)) if p1 else None})
        opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], betas=(0.9, 0.98), weight_decay=0.01)

    # ---- QA training: identical recipe to v4
    schedules = schedule_for(fold, seed)
    total = sum(e["updates"] for e in schedules)
    warmup = int(0.05 * total)

    def rate(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    updates, history = 0, []
    for epoch, s in enumerate(schedules):
        seed_training_epoch(seed, epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        losses, t0 = [], time.time()
        loader = DataLoader(ds["train"], batch_sampler=s["ordered_indices"], collate_fn=collate, num_workers=0)
        for j, b in enumerate(loader):
            guards(start)
            with private_training_loss(model, b, 3, "token_mean", table):
                out = model(b, stage=3)
            assert torch.isfinite(out["loss"])
            losses.append(float(out["loss"].detach()))
            (out["loss"] / 4).backward()
            if (j + 1) % 4 == 0 or j + 1 == len(loader):
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1))
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                updates += 1
                if updates % 10 == 0:
                    dump(folder / "progress.json", {"phase": "qa", "epoch": epoch + 1, "updates": updates, "total": total,
                                                    "loss": losses[-1], "wall_seconds": time.time() - start})
        to_device(model, EVAL_DEVICE)
        rows = predict(model, ds, collate, spec, "model_val", progress=folder / "eval_progress.json")
        to_device(model, device)
        history.append({"epoch": epoch + 1, "updates": updates, "mean_loss": float(np.mean(losses)),
                        "wall_seconds": time.time() - t0, "metrics": compute_metrics(rows, fracture_threshold=0.5)})
        dump(folder / "history.json", history)
    assert updates == total
    model.save(str(folder / "LAST.pt"), {"updates": updates, "epoch": 2, "model_class": "BoneVQAv5",
                                         "suite": suite, "arm": arm, "seed": seed, "device": device})
    # ---- evaluation on all roles with the fixed LAST checkpoint (CPU, see EVAL_DEVICE)
    to_device(model, EVAL_DEVICE)
    assess(model, ds, collate, spec, folder)
    dump(folder / "trained.json", {"checkpoint_sha256": sha(folder / "LAST.pt"), "wall_seconds": time.time() - start,
                                   "assessment_threads": torch.get_num_threads()})
    del model
    # ---- replay in a fresh process (same device)
    subprocess.run([sys.executable, "-u", __file__, "replay", suite, fold, arm, str(seed), EVAL_DEVICE], check=True, cwd=ROOT)
    dump(folder / "completed.json", {"wall_seconds": time.time() - start, "device": device})



MPS_JOB_THREADS = 4  # MPS jobs run training and CPU assessment with this many threads


def to_device(model, device):
    """Move the model and mark v5 buffers (answer bank, KG node features) for recomputation.

    They come from the frozen text encoder; computed on MPS and copied to CPU they differ from a
    CPU recomputation by ~1e-7, so CPU evaluation must recompute them exactly like a fresh-process
    load does (fresh-process replay of P failed by 3.6e-7 on 2026-10-30 before this fix).
    Training is unchanged: back on MPS they are recomputed on MPS, as at the start of training.
    """
    model.to(device)
    if hasattr(model, "v5_ready"):
        model.v5_ready = False
    return model


def assess(model, ds, collate, spec, folder):
    from src.utils import compute_metrics

    for role in ("model_val", "calibration", "test"):
        rows = predict(model, ds, collate, spec, role, canonical=(role == "calibration"),
                       progress=folder / "assessment_progress.json")
        dump(folder / f"{role}_predictions.json", {"records": rows, "metrics": compute_metrics(rows, fracture_threshold=0.5)})


def evaluate(suite, fold, arm, seed):
    """Re-run the final assessment + replay from a job's saved LAST checkpoint (no retraining).

    For jobs trained before the to_device() fix: training is identical, only the CPU assessment
    used MPS-computed v5 buffers.  Superseded outputs are kept in stale_assessment/.
    """
    import shutil
    import torch
    from experiments.v5.model_v5 import BoneVQAv5

    folder = job_dir(suite, fold, arm, int(seed))
    trained = read(folder / "trained.json")
    torch.set_num_threads(trained.get("assessment_threads", MPS_JOB_THREADS))
    assert sha(folder / "LAST.pt") == trained["checkpoint_sha256"], "LAST.pt hash mismatch"
    start = time.time()
    stale = folder / "stale_assessment"
    stale.mkdir(exist_ok=True)
    for name in ("model_val_predictions.json", "calibration_predictions.json", "test_predictions.json",
                 "replay.json", "FAILED.json", "assessment_progress.json"):
        if (folder / name).exists():
            shutil.move(str(folder / name), str(stale / name))
    spec = arm_spec(arm)
    model = BoneVQAv5.load(str(folder / "LAST.pt"), device=EVAL_DEVICE).eval()
    ds, collate = batcher(fold, spec)
    assess(model, ds, collate, spec, folder)
    del model
    subprocess.run([sys.executable, "-u", __file__, "replay", suite, fold, arm, str(seed), EVAL_DEVICE], check=True, cwd=ROOT)
    dump(folder / "reeval.json", {"reason": "CPU assessment recomputes v5 buffers (to_device fix)",
                                  "checkpoint_sha256": trained["checkpoint_sha256"], "seconds": time.time() - start,
                                  "note": "history.json epoch metrics still used MPS-computed buffers (monitoring only)"})
    dump(folder / "completed.json", {"wall_seconds": trained["wall_seconds"] + time.time() - start,
                                     "device": read(folder / "config.json")["device"], "reevaluated": True})


def gate(suite, fold, arm, seed, device, folder=None):
    """Two optimizer updates on 4 TRAIN rows; every enabled module must get gradient."""
    import torch
    from src.utils import normalize_answer
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

    seed = int(seed)
    folder = Path(folder) if folder else OUT / "gates" / f"fold_{fold}" / f"{arm}_seed{seed}_{device}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = arm_spec(arm)
    cfg = build_cfg(fold, arm, seed, suite)
    ds, collate = batcher(fold, spec)
    table = {(r["question"], normalize_answer(r["answer"])): 1.0 for r in ds["train"].records if r["answer_type"] == "OPEN"}
    start = time.time()
    if device == "mps":
        # Hard MPS qualification (added 2026-10-30 after two torch 2.6 MPS bugs passed every loss-level
        # check): every trainable parameter's MPS gradient must match CPU, same weights and batches.
        import gc
        from experiments.v5.qualify_mps_grads_v5 import qualify
        q = qualify(arm, fold, seed, suite)
        dump(folder / "mps_qualification.json", {**q, "arm": arm, "seconds": time.time() - start})
        assert q["passed"], f"MPS gradient qualification failed: {q['problems'][:3]}"
        gc.collect()
    gate_model = construct(cfg, device)
    rows = [i for i, r in enumerate(ds["train"].records) if r["question"] == CAN or r["answer_type"] == "OPEN"][:64]
    gate_boxes, _ = proposals_for(fold, spec["proposals"])
    sel = [i for i in rows if gate_boxes is None or gate_boxes.get(ds["train"].records[i]["image_id"])][:4] or rows[:4]
    b = collate([ds["train"][i] for i in sel])
    opt = torch.optim.AdamW([p for p in gate_model.parameters() if p.requires_grad], lr=1e-4)
    seen = {}
    gate_model.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        if spec.get("lens"):
            gate_model.forward_lens_align(b)["loss"].backward()
        with private_training_loss(gate_model, b, 3, "token_mean", table):
            out = gate_model(b, stage=3)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        for k, p in gate_model.named_parameters():
            if p.grad is not None and torch.count_nonzero(p.grad):
                seen[k] = True
        opt.step()
    need = ["qformer.query", "prefix_proj.0.weight", "fracture_head.net.0.weight"]
    mc = cfg["model"]
    if mc.get("v5_local"):
        need.append("local_tokens.attention.in_proj_weight")
    if mc.get("v5_lens"):
        need.append("v5_lens_fusion.compress.weight")
    if mc.get("v5_latent"):
        need += ["v5_gm.latent", "v5_lf.ca_l.attn.in_proj_weight"]
    if mc.get("v5_pf", "none") != "none":
        need.append("v5_prior.gnn.0.w.weight")
    missing = [k for k in need if k not in seen]
    dump(folder / "gate.json", {"arm": arm, "device": device, "required": need, "missing": missing,
                                "passed": not missing, "loss": float(out["loss"].detach()), "seconds": time.time() - start})
    assert not missing, f"gradient gate failed: {missing}"
    return folder

def replay(suite, fold, arm, seed, device):
    import torch
    from experiments.v5.model_v5 import BoneVQAv5

    folder = job_dir(suite, fold, arm, int(seed))
    trained = read(folder / "trained.json")
    assert sha(folder / "LAST.pt") == trained["checkpoint_sha256"]
    # CPU results depend on the thread count (PT, 2026-10-31: 8 vs 4 threads differ by 6e-8 on 17/400 rows).
    torch.set_num_threads(trained.get("assessment_threads", MPS_JOB_THREADS))
    spec = arm_spec(arm)
    model = BoneVQAv5.load(str(folder / "LAST.pt"), device=device).eval()
    ds, collate = batcher(fold, spec)
    rows = predict(model, ds, collate, spec, "test", limit=400)
    ref = read(folder / "test_predictions.json")["records"][:400]
    exact = rows == ref
    answers = all(a["pred"] == b["pred"] for a, b in zip(rows, ref))
    pmax = max([abs(a.get("p_yes", 0) - b.get("p_yes", 0)) for a, b in zip(rows, ref)] or [0])
    ok = exact or (device != "cpu" and answers and pmax <= 1e-4)
    dump(folder / "replay.json", {"records": len(rows), "exact": exact, "answers_equal": answers,
                                  "p_yes_max_abs_diff": pmax, "passed": ok, "device": device})
    assert ok, "fresh-process replay failed"


# ------------------------------------------------------------------ queue
def plan(suite):
    s = SUITES[suite]
    # Fold by fold, so a partial run still gives complete paired comparisons.
    jobs = [{"fold": f, "arm": a, "seed": sd, "device": s["device"]} for sd in s["seeds"] for f in s["folds"] for a in s["arms"]]
    dump(OUT / suite / "plan.json", {"suite": suite, "jobs": jobs, "created": time.time()})
    print(f"{len(jobs)} jobs planned for {suite}")


def prerequisites_ready(fold, arm):
    """V needs complete AVPC proposals; L needs anatomy masks for every TRAIN image of the fold."""
    spec = arm_spec(arm)
    if spec.get("proposals") == "avpc":
        d = MASKS / f"avpc_p{AVPC_SETTING['points_per_side']}_nms{AVPC_SETTING['nms']}"
        if not (d / "receipt.json").exists() or read(d / "receipt.json").get("images", 0) < 4013:
            return False
    if spec.get("lens"):
        train = {r["image_id"] for r in read(fold_meta(fold)["roles"]["train"]["path"])}
        for src in spec["lens"]["sources"]:
            folder = MASKS / f"anatomy_{src}"
            if not all((folder / f"{i}.npz").exists() for i in train):
                return False
    return True


def worker(suite, device, poll_seconds=600):
    """Run this device's jobs; jobs whose masks are not ready yet are retried later."""
    while True:
        p = read(OUT / suite / "plan.json")
        pending, ran = 0, False
        for job in p["jobs"]:
            if job["device"] != device:
                continue
            folder = job_dir(suite, job["fold"], job["arm"], job["seed"])
            if (folder / "completed.json").exists() or (folder / "FAILED.json").exists():
                continue
            if (folder / "CLAIMED").exists():
                continue
            if not prerequisites_ready(job["fold"], job["arm"]):
                pending += 1
                continue
            folder.mkdir(parents=True, exist_ok=True)
            try:
                os.close(os.open(folder / "CLAIMED", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            except FileExistsError:
                continue
            ran = True
            log = folder / "job.log"
            with log.open("a") as f:
                code = subprocess.call([sys.executable, "-u", __file__, "job", suite, job["fold"], job["arm"], str(job["seed"]), device],
                                       stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            if code != 0:
                dump(folder / "FAILED.json", {"exit": code, "log": str(log), "wall": time.time()})
                print(f"FAILED {job} -> see {log}", flush=True)
            else:
                print(f"done {job}", flush=True)
            break  # re-read the plan and re-check prerequisites after every job
        if not ran:
            if pending == 0:
                print("worker: no runnable or pending jobs left", flush=True)
                return
            dump(OUT / suite / f"worker_{device}_waiting.json", {"pending_jobs_waiting_for_masks": pending, "wall": time.time()})
            time.sleep(poll_seconds)


if __name__ == "__main__":
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "worker", "job", "replay", "evaluate", "analyze", "gate"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.args[0])
    elif a.cmd == "worker":
        worker(a.args[0], a.device)
    elif a.cmd == "job":
        run_job(*a.args)
    elif a.cmd == "replay":
        replay(*a.args)
    elif a.cmd == "evaluate":  # evaluate <suite> <fold> <arm> <seed>
        evaluate(*a.args)
    elif a.cmd == "gate":  # gate <fold> <arm> [seed]
        gate("gatecheck", a.args[0], a.args[1], a.args[2] if len(a.args) > 2 else 42, a.device)
    else:
        from experiments.v5.analyze_v5 import analyze
        analyze(a.args[0])
