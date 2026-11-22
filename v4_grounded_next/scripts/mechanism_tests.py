"""Three mechanism tests on fold-f model_val (canonical rows) for every trained arm.

T1  M_hat vs detector box map.  Does M_hat copy the proposal boxes?
    Spearman(M_hat, box_map) on m=1 images; M_hat mass inside GT vs inside the
    boxes; the decisive split is fractured images whose boxes MISS the GT
    (box grid and GT grid disjoint): a box-copier puts its mass in the box, a
    lesion-grounder in the GT.  m=0 fractured images show M_hat with no box at all.
T2  Distribution-preserving spatial shift of M_hat.  ell is replaced by a
    random permutation of its own 196 cells (3 seeds) and by a (7,7) circular
    roll: max / mean / LSE and the whole value histogram are unchanged, only
    the location moves.  |dp|, decision flips and dAUPRC measure how much the
    fracture decision depends on WHERE M_hat points.
T3  Lesion Occlusion Test.  Each fractured image: occlude the GT lesion
    (padded GT boxes) or a size-matched, same-radiograph background region
    (IoU 0 with every GT, mean grey > 40, bone foreground), with three fills:
    mean grey, Gaussian blur sigma 5, cv2.inpaint (Telea, r 5).  d = p(orig) -
    p(occluded).  Statistic D = d_lesion - d_background per image; 95 % CI by
    component bootstrap (2000), two-sided sign-flip permutation p (10000).
    Two proposal modes: 're-detect' (the sealed full-TRAIN detector runs on the
    occluded image: the true end-to-end input contract) and 'fixed' (original
    proposals: isolates the VQA model from the detector).

GT is used to build interventions and metrics only; the model never receives it.

    $PY -m scripts.mechanism_tests --fold 0 [--limit 40] [--arms fold0/V4-G ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from vg import geometry, hard_negatives
from vg.data import gt_geometry, public_batch
from vg.detector import load_detector, predict_scored
from vg.paths import RUNS, V4_REPORT, dump_json, read_json, sha256
from vg.trainer import FoldData, build_model

B_RESAMPLES, PERM_RESAMPLES = 2000, 10000


# ------------------------------------------------------------------ statistics
def comp_bootstrap(values, comps, seed=0):
    values, comps = np.asarray(values, float), np.asarray(comps)
    uniq = np.unique(comps)
    idx = {c: np.where(comps == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B_RESAMPLES):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        stats.append(values[sel].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def signflip_p(values, comps, seed=1):
    """Two-sided sign-flip permutation test of mean(values)=0, flipping whole components."""
    values, comps = np.asarray(values, float), np.asarray(comps)
    uniq = np.unique(comps)
    sums = np.array([values[comps == c].sum() for c in uniq])
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(PERM_RESAMPLES, len(uniq)))
    null = np.abs(flips @ sums)
    return float((1 + (null >= obs - 1e-12).sum()) / (PERM_RESAMPLES + 1))


def summarise(values, comps):
    if len(values) == 0:
        return None
    m, lo, hi = comp_bootstrap(values, comps)
    return {"mean": m, "ci95": [lo, hi], "p_signflip": signflip_p(values, comps), "n": len(values)}


# ------------------------------------------------------------------ model loading
def load_arm(tag, fold, device):
    from vg.paths import seed_dir
    d = seed_dir(tag, fold)
    started, completed = read_json(d / "started.json"), read_json(d / "completed.json")
    ck = d / "LAST.pt"
    assert ck.exists(), f"{ck} missing (run trained with --no-save?)"
    model, _ = build_model(started["arm"], fold, device="cpu", overrides=started["overrides"])
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    own = model.state_dict()
    res = model.load_state_dict({k: v for k, v in payload["state"].items() if k in own}, strict=False)
    assert not res.unexpected_keys and not (set(payload["state"]) - set(own))
    assert payload["extra"]["updates"] == completed["updates"]
    return model.to(device).eval(), started, sha256(ck)


def load_sealed_v4(fold, device):
    """Sealed V4 (refactor_v2, local_real_noaux/LAST.pt, full 1438-update run), read
    only, into the V4 arm.  Reference for T3; it has no M_hat, so no T1/T2."""
    from vg.paths import V4_REPORT
    folder = V4_REPORT / f"fold_{fold}/local_real_noaux"
    ck = folder / "LAST.pt"
    h = sha256(ck)
    assert h == read_json(folder / "completed.json")["checkpoint_sha256"]
    model, _ = build_model("V4", fold, device="cpu")
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    own = model.state_dict()
    res = model.load_state_dict({k: v for k, v in payload["state"].items() if k in own}, strict=False)
    assert not res.unexpected_keys and not (set(payload["state"]) - set(own))
    return model.to(device).eval(), {"arm": "V4", "overrides": {}, "note": "sealed V4, 1438 updates (2 epochs)"}, h


def gt_grid(rec, cfg):
    boxes, polys, size = gt_geometry(rec)
    mask = geometry.rasterize(polys, size, boxes)
    return geometry.mask_to_grid(mask, size, cfg["image_size"], cfg["vg_grid"]).reshape(-1) > 0, boxes


# ------------------------------------------------------------------ T1 / T2
@torch.no_grad()
def t1_t2(model, data, role, limit, bs=8):
    cfg = model.cfg
    idx = data.canonical_indices(role)[: limit or None]
    rows, base_p, perm_p, y, comps = [], [], {}, [], []
    variants = [("perm_s0", 0), ("perm_s1", 1), ("perm_s2", 2), ("roll_7_7", None)]
    for off in range(0, len(idx), bs):
        records, images, b = data.eval_batch(role, idx[off:off + bs])
        r = model.infer(b)
        ell = r["core"]["ell"].detach()
        base_p.extend(r["p_final"].tolist())
        y.extend(int(bool(x["fractured"])) for x in records)
        comps.extend(data.lookup[x["image_id"]] for x in records)
        for name, seed in variants:
            flat = ell.reshape(len(records), -1)
            if seed is None:
                new = torch.roll(ell, shifts=(7, 7), dims=(2, 3))
            else:
                g = torch.Generator().manual_seed(1000 + seed)
                perm = torch.stack([torch.randperm(flat.shape[1], generator=g) for _ in range(len(records))]).to(flat.device)
                new = torch.gather(flat, 1, perm).reshape_as(ell)
            assert torch.allclose(new.flatten(1).sort(1).values, flat.sort(1).values)  # same distribution
            with model.intervene(ell=new):
                rp = model.infer(b)
            perm_p.setdefault(name, []).extend(rp["p_final"].tolist())
        mh = r["m_hat"].reshape(len(records), -1).numpy()
        bm = r["box_map"].reshape(len(records), -1).numpy()
        for j, rec in enumerate(records):
            w = mh[j] / mh[j].sum()
            row = {"image_id": rec["image_id"], "fractured": bool(rec["fractured"]), "m": bool(len(b["proposals"][j])),
                   "comp": data.lookup[rec["image_id"]]}
            box = bm[j] > 0
            if row["m"]:
                row["spearman_mhat_box"] = float(spearmanr(mh[j], bm[j]).correlation) if bm[j].std() > 0 else None
                row["mass_in_box"] = float(w[box].sum()); row["box_area"] = float(box.mean())
            if rec["fractured"]:
                gt, _ = gt_grid(rec, cfg)
                row["mass_in_gt"] = float(w[gt].sum()); row["gt_area"] = float(gt.mean())
                row["point_in_gt"] = bool(gt[int(mh[j].argmax())])
                if row["m"]:
                    row["box_hits_gt"] = bool((box & gt).any())
                    row["point_in_box"] = bool(box[int(mh[j].argmax())])
            rows.append(row)
    y = np.array(y); base = np.array(base_p); comps = np.array(comps)

    def agg(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    t1 = {}
    m1 = [r for r in rows if r["m"]]
    t1["all_m1"] = {"n": len(m1), "spearman_mhat_box": agg(m1, "spearman_mhat_box"),
                    "mass_in_box": agg(m1, "mass_in_box"), "box_area_chance": agg(m1, "box_area")}
    for name, sel in (("fractured_box_hits_gt", [r for r in rows if r["fractured"] and r["m"] and r["box_hits_gt"]]),
                      ("fractured_box_misses_gt", [r for r in rows if r["fractured"] and r["m"] and not r["box_hits_gt"]]),
                      ("fractured_m0", [r for r in rows if r["fractured"] and not r["m"]])):
        rec = {"n": len(sel), "mass_in_gt": agg(sel, "mass_in_gt"), "gt_area_chance": agg(sel, "gt_area"),
               "pointing_game_gt": agg(sel, "point_in_gt")}
        if name != "fractured_m0":
            rec.update(mass_in_box=agg(sel, "mass_in_box"), box_area_chance=agg(sel, "box_area"),
                       pointing_game_box=agg(sel, "point_in_box"), spearman_mhat_box=agg(sel, "spearman_mhat_box"))
        if sel:  # paired (mass_in_gt - gt_area): above-chance grounding with component CI
            t1_key = [r["mass_in_gt"] - r["gt_area"] for r in sel]
            rec["mass_in_gt_minus_chance"] = summarise(t1_key, [r["comp"] for r in sel])
            if name != "fractured_m0":
                rec["mass_in_gt_minus_mass_in_box"] = summarise([r["mass_in_gt"] - r["mass_in_box"] for r in sel],
                                                                 [r["comp"] for r in sel])
        t1[name] = rec

    t2 = {"base_auprc": float(average_precision_score(y, base))}
    for name, vals in perm_p.items():
        v = np.array(vals); dp = np.abs(v - base)
        t2[name] = {"mean_abs_dp_fractured": float(dp[y == 1].mean()), "mean_abs_dp_negative": float(dp[y == 0].mean()),
                    "max_abs_dp": float(dp.max()), "decision_flips@0.5": int(((v >= .5) != (base >= .5)).sum()),
                    "auprc": float(average_precision_score(y, v)),
                    "d_auprc": float(average_precision_score(y, v) - t2["base_auprc"]),
                    "dp_fractured_signed": summarise(list((base - v)[y == 1]), list(comps[y == 1]))}
    return t1, t2, rows


# ------------------------------------------------------------------ T3
def fill(arr, boxes, mode):
    out = arr.copy()
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 1
    m = mask.astype(bool)
    if mode == "mean":
        out[m] = arr.reshape(-1, arr.shape[2]).mean(0).astype(arr.dtype)
    elif mode == "blur5":
        bl = cv2.GaussianBlur(arr, (0, 0), sigmaX=5, sigmaY=5)
        out[m] = bl[m]
    elif mode == "inpaint":
        out = cv2.inpaint(arr, mask * 255, 5, cv2.INPAINT_TELEA)
    else:
        raise ValueError(mode)
    return out


def build_occlusions(data, role, limit, fold):
    """Detector-independent of the arm: build once, cache occluded images + re-detected proposals."""
    receipt = read_json(V4_REPORT / f"fold_{fold}/detector/receipt.json")
    assert sha256(receipt["checkpoint"]) == receipt["sha256"]
    det, _ = load_detector(receipt["checkpoint"])
    idx = [i for i in data.canonical_indices(role) if data.ds[role].records[i]["fractured"]][: limit or None]
    items = []
    for i in idx:
        rec = data.ds[role].records[i]
        im = data.ds[role][i]["image"]
        gtb, _, size = gt_geometry(rec)
        assert size == im.size
        gray = np.asarray(im.convert("L"))
        groups = hard_negatives.sample(gray, gtb, rec["image_id"], n_neg=8, seed=0)
        lesion = [g["positive"] for g in groups]
        bg = []
        for g in groups:
            ok = [n for n in g["negatives"] if all(geometry.box_iou(n, p) == 0 for p in lesion) and
                  all(geometry.box_iou(n, x) == 0 for x in bg)]
            if not ok:
                bg = None; break
            bg.append(ok[0])
        if not bg:
            continue
        arr = np.asarray(im.convert("RGB"))
        it = {"image_id": rec["image_id"], "comp": data.lookup[rec["image_id"]], "orig": im,
              "orig_props": data.props[role][rec["image_id"]], "lesion": lesion, "background": bg,
              "relax": [g["relax"][0] if g["relax"] else None for g in groups], "variants": {}}
        for region, boxes in (("lesion", lesion), ("background", bg)):
            for mode in ("mean", "blur5", "inpaint"):
                occ = Image.fromarray(fill(arr, boxes, mode))
                props, _ = predict_scored(det, occ)
                it["variants"][(region, mode)] = {"image": occ, "props": props}
        items.append(it)
    return items


@torch.no_grad()
def t3(model, items, question):
    def p_of(images, props):
        recs = [{"question": question, "image_id": f"occ{k}"} for k in range(len(images))]
        b = public_batch(recs, images, {f"occ{k}": p for k, p in enumerate(props)}, "detector_full")
        return model.infer(b)["p_final"].tolist()
    res, per = {}, []
    for it in items:
        row = {"image_id": it["image_id"], "comp": it["comp"], "m_orig": bool(it["orig_props"])}
        row["p_orig"] = p_of([it["orig"]], [it["orig_props"]])[0]
        for (region, mode), v in it["variants"].items():
            pr, pf = p_of([v["image"], v["image"]], [v["props"], it["orig_props"]])
            row[f"{region}|{mode}|redetect"] = row["p_orig"] - pr
            row[f"{region}|{mode}|fixed"] = row["p_orig"] - pf
            row[f"m_after|{region}|{mode}"] = bool(v["props"])
        per.append(row)
    comps = [r["comp"] for r in per]
    for mode in ("mean", "blur5", "inpaint"):
        for pm in ("redetect", "fixed"):
            dl = [r[f"lesion|{mode}|{pm}"] for r in per]
            db = [r[f"background|{mode}|{pm}"] for r in per]
            res[f"{mode}|{pm}"] = {"d_lesion": float(np.mean(dl)), "d_background": float(np.mean(db)),
                                   "D=d_lesion-d_background": summarise([a - b for a, b in zip(dl, db)], comps)}
        res[f"{mode}|proposal_lost_rate"] = {
            "lesion": float(np.mean([r["m_orig"] and not r[f"m_after|lesion|{mode}"] for r in per])),
            "background": float(np.mean([r["m_orig"] and not r[f"m_after|background|{mode}"] for r in per]))}
    return res, per


# ------------------------------------------------------------------ driver
def main(a):
    torch.set_num_threads(4)
    t0 = time.time()
    data = FoldData(a.fold, "crossfit")
    tags = a.arms or sorted(str(p.parents[2].relative_to(RUNS / "arms")) for p in
                            (RUNS / "arms" / f"fold{a.fold}").glob(f"*/fold_{a.fold}/seed_42/completed.json"))
    print("arms:", tags, flush=True)
    items = build_occlusions(data, a.role, a.limit, a.fold)
    print(f"occlusion set: {len(items)} fractured images ({time.time() - t0:.0f}s)", flush=True)
    from vg.paths import CANONICAL_Q
    report = {"fold": a.fold, "role": a.role, "limit": a.limit, "n_occlusion_images": len(items),
              "occlusion_relax_levels": sorted({x for it in items for x in it["relax"] if x}), "arms": {}}
    if a.sealed_v4:
        tags = list(tags) + ["sealed/V4"]
    for tag in tags:
        if tag == "sealed/V4":
            model, started, ck = load_sealed_v4(a.fold, a.device)
        else:
            model, started, ck = load_arm(tag, a.fold, a.device)
        rec = {"arm": started["arm"], "overrides": started["overrides"], "checkpoint_sha256": ck,
               **({"note": started["note"]} if "note" in started else {})}
        if model.cfg["vg_ground"]:
            rec["T1_mhat_vs_box"], rec["T2_spatial_shift"], rows = t1_t2(model, data, a.role, a.limit)
        rec["T3_occlusion"], per = t3(model, items, CANONICAL_Q)
        report["arms"][tag] = rec
        dump_json(RUNS / "mechanism" / f"fold{a.fold}" / (tag.replace("/", "_") + ("_smoke.json" if a.limit else ".json")),
                  {**rec, "t3_rows": per, "t1_rows": rows if model.cfg["vg_ground"] else None})
        print(json.dumps({"arm": tag, "sec": round(time.time() - t0)}), flush=True)
        del model
        if a.device == "mps":
            torch.mps.empty_cache()
    report["seconds"] = time.time() - t0
    out = RUNS / "mechanism" / f"fold{a.fold}" / (a.out or ("summary_smoke.json" if a.limit else "summary.json"))
    dump_json(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="model_val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--sealed-v4", action="store_true", help="add the sealed V4 checkpoint as a T3 reference (read only)")
    ap.add_argument("--out", default=None, help="summary file name (default summary.json)")
    main(ap.parse_args())
