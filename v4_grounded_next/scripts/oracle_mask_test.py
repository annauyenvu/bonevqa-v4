"""Oracle GT-mask injection: is the small T2 effect a localisation problem or a
read-out (head-weight) problem?

On fractured model_val canonical rows, replace ell (M_hat = sigmoid(ell)) via
``model.intervene`` with:

  native        the model's own ell
  uniform       constant ell (= its mean): z_les = mean patch, no localisation
  gt_binary     +6 on GT-mask cells, -6 elsewhere          (M_hat ~ 0.998 / 0.002)
  shift_binary  the same cell pattern translated onto bone, no overlap with GT
  gt_rank       the model's OWN ell values, the |GT| largest put on GT cells
  shift_rank    the same values, the |GT| largest put on the shifted cells

The shifted mask is the GT grid mask translated by the integer offset that keeps
it inside the 14x14 grid, has zero overlap with the (1-cell dilated) GT mask and
maximises the fraction of its cells on bone (gray > 40); so cell count and shape
are identical and only location differs.  *_rank pairs keep the native value
distribution exactly (sorted values asserted equal).

Reading:  gt_binary vs shift_binary large but native T2 tiny -> the read-out can
use location, M_hat is too flat / mislocalised.  gt_binary vs shift_binary tiny
too -> the M_hat -> logit read-out (t_les, emphasis, h_les) is too weak after
training.  Each logit change is split into the h_les(z_les) route and the
fusion route (t_les + emphasis -> x_f -> FractureHead).

    $PY -m scripts.oracle_mask_test --fold 0 --arms fold0/V4-GC_lc0.33
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vg import geometry
from vg.paths import RUNS, dump_json
from vg.trainer import FoldData
from scripts.mechanism_tests import gt_grid, load_arm, summarise

L = 6.0
CONDS = ("native", "uniform", "gt_binary", "shift_binary", "gt_rank", "shift_rank")


def bone_grid(image, cfg):
    gray = np.asarray(image.convert("L"))
    return geometry.mask_to_grid((gray > 40).astype(np.uint8), image.size, cfg["image_size"], cfg["vg_grid"]) > 0.5


def shifted_mask(gt, bone):
    """gt, bone: (G,G) bool.  Best in-grid translation off the dilated GT, on bone."""
    G = gt.shape[0]
    if not gt.any():
        return None, None, None
    ys, xs = np.nonzero(gt)
    dil = gt.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= np.roll(np.roll(gt, dy, 0), dx, 1)
    best = None
    for dy in range(-ys.min(), G - ys.max()):
        for dx in range(-xs.min(), G - xs.max()):
            if dy == 0 and dx == 0:
                continue
            m = np.zeros_like(gt)
            m[ys + dy, xs + dx] = True
            if (m & dil).any():
                continue
            key = (bone[m].mean(), -(dy * dy + dx * dx))  # most on bone, then nearest
            if best is None or key > best[0]:
                best = (key, m, (dy, dx))
    return (None, None, None) if best is None else (best[1], best[2], float(best[0][0]))


def rank_assign(ell_flat, target, gen):
    """Native values; the |target| largest go (randomly) onto target cells, rest randomly elsewhere."""
    v = ell_flat.sort(descending=True).values
    n = int(target.sum())
    out = torch.empty_like(ell_flat)
    tgt = torch.nonzero(target).squeeze(1)
    rest = torch.nonzero(~target).squeeze(1)
    out[tgt[torch.randperm(len(tgt), generator=gen)]] = v[:n]
    out[rest[torch.randperm(len(rest), generator=gen)]] = v[n:]
    assert torch.equal(out.sort().values, ell_flat.sort().values)
    return out


@torch.no_grad()
def run_arm(model, data, role, limit):
    cfg = model.cfg
    G = cfg["vg_grid"]
    idx = [i for i in data.canonical_indices(role) if data.ds[role].records[i]["fractured"]][: limit or None]
    gen = torch.Generator().manual_seed(7)
    rows = []
    for off in range(0, len(idx), 8):
        records, images, b = data.eval_batch(role, idx[off:off + 8])
        base = model.infer(b)
        ell = base["core"]["ell"].detach()
        n = len(records)
        masks, keep = {c: ell.clone() for c in CONDS}, []
        for j, rec in enumerate(records):
            gt, _ = gt_grid(rec, cfg)
            gt = gt.reshape(G, G)
            sh, offset, on_bone = shifted_mask(gt, bone_grid(images[j], cfg))
            keep.append((gt, sh, offset, on_bone))
            if sh is None:
                continue
            e = ell[j, 0].flatten()
            gtt, sht = torch.from_numpy(gt.flatten()).to(e.device), torch.from_numpy(sh.flatten()).to(e.device)
            masks["uniform"][j, 0] = e.mean()
            masks["gt_binary"][j, 0] = torch.where(gtt, L, -L).reshape(G, G).to(e)
            masks["shift_binary"][j, 0] = torch.where(sht, L, -L).reshape(G, G).to(e)
            masks["gt_rank"][j, 0] = rank_assign(e.cpu(), gtt.cpu(), gen).to(e).reshape(G, G)
            masks["shift_rank"][j, 0] = rank_assign(e.cpu(), sht.cpu(), gen).to(e).reshape(G, G)
        out = {}
        for c in CONDS:
            if c == "native":
                r = base
            else:
                with model.intervene(ell=masks[c]):
                    r = model.infer(b)
            core = r["core"]
            h = model.les_head(core["z_les"]).squeeze(-1).float().cpu()
            out[c] = {"p": r["p_final"], "logit": core["logit_local"].detach().float().cpu(), "h": h,
                      "z": core["z_les"].detach().float().cpu()}
        for j, rec in enumerate(records):
            gt, sh, offset, on_bone = keep[j]
            row = {"image_id": rec["image_id"], "comp": data.lookup[rec["image_id"]], "m": bool(b["proposals"][j]),
                   "gt_cells": int(gt.sum()), "shift_ok": sh is not None}
            if sh is not None:
                row.update(shift=list(map(int, offset)), shift_on_bone=on_bone,
                           gt_on_bone=float(bone_grid(images[j], cfg)[gt].mean()))
                zu = out["uniform"]["z"][j]
                for c in CONDS:
                    row[f"p|{c}"] = float(out[c]["p"][j]); row[f"logit|{c}"] = float(out[c]["logit"][j])
                    row[f"h|{c}"] = float(out[c]["h"][j]); row[f"zdist|{c}"] = float((out[c]["z"][j] - zu).norm())
            rows.append(row)
    return rows


def summary(rows):
    ok = [r for r in rows if r["shift_ok"]]
    comps = [r["comp"] for r in ok]
    res = {"n": len(ok), "n_no_valid_shift": len(rows) - len(ok),
           "mean_gt_cells": float(np.mean([r["gt_cells"] for r in ok])),
           "shift_on_bone": float(np.mean([r["shift_on_bone"] for r in ok])),
           "gt_on_bone": float(np.mean([r["gt_on_bone"] for r in ok])),
           "mean_p": {c: float(np.mean([r[f"p|{c}"] for r in ok])) for c in CONDS},
           "zdist_from_uniform": {c: float(np.mean([r[f"zdist|{c}"] for r in ok])) for c in CONDS}}
    for a, b in (("gt_binary", "shift_binary"), ("gt_rank", "shift_rank"), ("native", "uniform"),
                 ("gt_binary", "native"), ("native", "shift_rank")):
        dp = [r[f"p|{a}"] - r[f"p|{b}"] for r in ok]
        dl = [r[f"logit|{a}"] - r[f"logit|{b}"] for r in ok]
        dh = [r[f"h|{a}"] - r[f"h|{b}"] for r in ok]
        res[f"{a}_minus_{b}"] = {"dp": summarise(dp, comps), "mean_abs_dp": float(np.mean(np.abs(dp))),
                                 "dlogit": float(np.mean(dl)), "dlogit_h_les_route": float(np.mean(dh)),
                                 "dlogit_fusion_route": float(np.mean(np.array(dl) - np.array(dh)))}
    return res


def head_norms(model):
    lp = model.lesion_pool
    return {"les_head.linear_w": float(model.les_head[1].weight.norm()), "les_head.linear_b": float(model.les_head[1].bias.abs()),
            "t_les.token_w": float(lp.token.weight.norm()), "emphasis_w": float(lp.emphasis.weight.norm()),
            "fracture_head_params": float(torch.cat([p.flatten() for p in model.fracture_head.parameters()]).norm())}


def main(a):
    torch.set_num_threads(4)
    t0 = time.time()
    data = FoldData(a.fold, "crossfit")
    report = {"fold": a.fold, "role": a.role, "L": L, "arms": {}}
    for tag in a.arms:
        model, started, ck = load_arm(tag, a.fold, a.device)
        assert model.cfg["vg_ground"]
        rows = run_arm(model, data, a.role, a.limit)
        rec = {"arm": started["arm"], "overrides": started["overrides"], "checkpoint_sha256": ck,
               "head_norms": head_norms(model), **summary(rows)}
        report["arms"][tag] = rec
        sfx = "_smoke" if a.limit else ""
        dump_json(RUNS / "mechanism" / f"fold{a.fold}" / f"oracle_{tag.replace('/', '_')}{sfx}.json", {**rec, "rows": rows})
        print(json.dumps({"arm": tag, "sec": round(time.time() - t0),
                          "gt_bin-shift_bin": rec["gt_binary_minus_shift_binary"]["dp"]["mean"],
                          "gt_rank-shift_rank": rec["gt_rank_minus_shift_rank"]["dp"]["mean"]}), flush=True)
        del model
        if a.device == "mps":
            torch.mps.empty_cache()
    dump_json(RUNS / "mechanism" / f"fold{a.fold}" / (a.out or ("oracle_summary_smoke.json" if a.limit else "oracle_summary.json")), report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="model_val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
