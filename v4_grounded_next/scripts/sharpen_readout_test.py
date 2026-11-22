"""Read-time sharpening of the z_les spatial weights (no retraining).

Trained read:  w = M_hat / sum M_hat,  z_les = sum_p w_p G1_p.
Diagnostic reads (applied at evaluation only, inside LesionPooling; the
emphasis term m_hat * emphasis(patches) is left unchanged, so only z_les -- and
through it h_les(z_les) and t_les -- changes):

  native     the model's own read (vg_read_gamma; = gamma1 for models trained with gamma 1)
  gamma1     w = M_hat / sum M_hat
  gamma2     w = M_hat^2 / sum M_hat^2
  gamma4     w = M_hat^4 / sum M_hat^4
  top16      w = M_hat * 1[top-16 cells] / sum           (16 of 196 cells)

For each read, on model_val canonical rows:
  base       AUPRC, AUROC (the head was trained on the native read, so a drop is expected)
  T2         the same spatial permutations / roll(7,7) of ell as mechanism_tests;
             |dp|, dAUPRC relative to the *same* read without permutation
  oracle     fractured rows: gt_rank / shift_rank (native ell values re-placed) and
             gt_binary / shift_binary (+-6), as in oracle_mask_test; dp, logit split
             (h_les route vs fusion route), and ||z_les - mean patch||
  weights    effective number of cells 1/sum w^2, weight mass on GT (native ell)

    $PY -m scripts.sharpen_readout_test --fold 0 --arms fold0/V4-GC-dp30_e2 fold0/V4-GC_e2
"""
from __future__ import annotations

import argparse
import json
import time
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from vg.paths import RUNS, dump_json
from vg.trainer import FoldData
from scripts.mechanism_tests import gt_grid, load_arm, summarise
from scripts.oracle_mask_test import L, bone_grid, head_norms, rank_assign, shifted_mask

READS = ("native", "gamma1", "gamma2", "gamma4", "top16")
T2_VARIANTS = (("perm_s0", 0), ("perm_s1", 1), ("perm_s2", 2), ("roll_7_7", None))
ORACLE = ("gt_rank", "shift_rank", "gt_binary", "shift_binary")


def weights(m_hat, read, ell_flat=None):
    if read.startswith("gamma"):
        # log-space: M_hat^g / sum M_hat^g = softmax(g * log sigmoid(ell)).  The direct power
        # underflows (sum < 1e-12) on rows whose M_hat is uniformly small, leaving w ~ 0.
        return torch.softmax(float(read[5:]) * F.logsigmoid(ell_flat), dim=1)
    if read == "native":
        w = m_hat
    elif read.startswith("top"):
        k = int(read[3:])
        idx = m_hat.topk(k, dim=1).indices
        w = torch.zeros_like(m_hat).scatter(1, idx, m_hat.gather(1, idx))
    else:
        raise ValueError(read)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)


@contextmanager
def read_mode(model, read):
    """Swap LesionPooling.forward for one whose z_les uses the sharpened weights."""
    lp = model.lesion_pool
    assert lp.read == "feature" and lp.init == "identity"
    if read == "native":  # the model's own trained read (gamma / detach from its config), unpatched
        yield
        return
    orig = lp.forward

    def fwd(self, g1, ell):
        b = g1.shape[0]
        ell_flat = ell.reshape(b, -1)
        m_hat = torch.sigmoid(ell_flat)
        patches = g1[:, 1:]
        w = weights(m_hat, read, ell_flat)
        z_les = torch.einsum("bp,bpd->bd", w, patches)
        evidence = torch.stack([self.lse(ell_flat) - self.ell0, m_hat.max(1).values, m_hat.mean(1)], dim=1)
        t_les = self.token(z_les)
        emphasized = patches + m_hat.unsqueeze(-1) * self.emphasis(patches)
        g2 = torch.cat([(g1[:, 0] + t_les).unsqueeze(1), emphasized], dim=1)
        self.last_w = w.detach()
        return g2, z_les, evidence

    lp.forward = types.MethodType(fwd, lp)
    try:
        yield
    finally:
        lp.forward = orig


def permuted(ell, name, seed):
    n = ell.shape[0]
    flat = ell.reshape(n, -1)
    if seed is None:
        return torch.roll(ell, shifts=(7, 7), dims=(2, 3))
    g = torch.Generator().manual_seed(1000 + seed)  # same permutations as mechanism_tests.t1_t2
    perm = torch.stack([torch.randperm(flat.shape[1], generator=g) for _ in range(n)]).to(flat.device)
    return torch.gather(flat, 1, perm).reshape_as(ell)


@torch.no_grad()
def run_arm(model, data, role, limit, bs=8):
    cfg = model.cfg
    G = cfg["vg_grid"]
    idx = data.canonical_indices(role)[: limit or None]
    gen = torch.Generator().manual_seed(7)
    y, comps = [], []
    P = {r: {"base": [], **{v: [] for v, _ in T2_VARIANTS}} for r in READS}
    orows = {r: [] for r in READS}
    wstats = {r: {"n_eff": [], "w_in_gt": [], "zdist_native": []} for r in READS}
    for off in range(0, len(idx), bs):
        records, images, b = data.eval_batch(role, idx[off:off + bs])
        r_ref = model.infer(b)  # unpatched model
        ell = r_ref["core"]["ell"].detach()  # ell does not depend on the read mode
        with model.intervene(ell=torch.zeros_like(ell)):  # uniform M_hat -> z_les = mean patch
            zu = model.infer(b)["core"]["z_les"].detach().float().cpu()
        n = len(records)
        y.extend(int(bool(r["fractured"])) for r in records)
        comps.extend(data.lookup[r["image_id"]] for r in records)
        # oracle masks for fractured rows with a valid shift
        omask = {c: ell.clone() for c in ORACLE}
        ometa = {}
        for j, rec in enumerate(records):
            if not rec["fractured"]:
                continue
            gt, _ = gt_grid(rec, cfg)
            gtg = gt.reshape(G, G)
            sh, offset, on_bone = shifted_mask(gtg, bone_grid(images[j], cfg))
            ometa[j] = (gt, sh)
            if sh is None:
                continue
            e = ell[j, 0].flatten()
            gtt, sht = torch.from_numpy(gtg.flatten()), torch.from_numpy(sh.flatten())
            omask["gt_rank"][j, 0] = rank_assign(e.cpu(), gtt, gen).to(e).reshape(G, G)
            omask["shift_rank"][j, 0] = rank_assign(e.cpu(), sht, gen).to(e).reshape(G, G)
            omask["gt_binary"][j, 0] = torch.where(gtt.to(e.device), L, -L).reshape(G, G).to(e)
            omask["shift_binary"][j, 0] = torch.where(sht.to(e.device), L, -L).reshape(G, G).to(e)
        for read in READS:
            with read_mode(model, read):
                r0 = model.infer(b)
                w = model.lesion_pool.last_w.detach().float().cpu()
                c0 = r0["core"]
                P[read]["base"].extend(r0["p_final"].tolist())
                for v, seed in T2_VARIANTS:
                    with model.intervene(ell=permuted(ell, v, seed)):
                        P[read][v].extend(model.infer(b)["p_final"].tolist())
                # z distance from the mean patch (= uniform-M_hat read)
                outs = {"native_ell": (r0, w)}
                for c in ORACLE:
                    with model.intervene(ell=omask[c]):
                        rc = model.infer(b)
                    outs[c] = (rc, model.lesion_pool.last_w.detach().float().cpu())
            if read == "native":  # the patched native read must be the trained model exactly
                d = float((r0["p_final"] - r_ref["p_final"]).abs().max())
                assert d < 1e-5, f"native read parity failed {d}"
            z0 = c0["z_les"].detach().float().cpu()
            for j, rec in enumerate(records):
                wstats[read]["n_eff"].append(float(1.0 / (w[j] ** 2).sum()))
                wstats[read]["zdist_native"].append(float((z0[j] - zu[j]).norm()))
                if j in ometa:
                    gt, sh = ometa[j]
                    wstats[read]["w_in_gt"].append(float(w[j][torch.from_numpy(gt)].sum()))
                    if sh is None:
                        continue
                    row = {"image_id": rec["image_id"], "comp": data.lookup[rec["image_id"]]}
                    for c in ("native_ell",) + ORACLE:
                        rc, wc = outs[c]
                        core = rc["core"]
                        row[f"p|{c}"] = float(rc["p_final"][j])
                        row[f"logit|{c}"] = float(core["logit_local"][j])
                        row[f"h|{c}"] = float(model.les_head(core["z_les"][j:j + 1]).squeeze())
                        row[f"zdist|{c}"] = float((core["z_les"][j].float().cpu() - zu[j]).norm())
                        row[f"w_in_gt|{c}"] = float(wc[j][torch.from_numpy(gt)].sum())
                    orows[read].append(row)
        print(f"  {off + n}/{len(idx)}", flush=True)
    y = np.array(y)
    res = {}
    for read in READS:
        base = np.array(P[read]["base"])
        rec = {"auprc": float(average_precision_score(y, base)), "auroc": float(roc_auc_score(y, base)),
               "n_eff_cells": float(np.mean(wstats[read]["n_eff"])),
               "w_mass_in_gt_fractured": float(np.mean(wstats[read]["w_in_gt"])),
               "zdist_native_from_mean_patch": float(np.mean(wstats[read]["zdist_native"])), "T2": {}}
        for v, _ in T2_VARIANTS:
            pv = np.array(P[read][v]); dp = np.abs(pv - base)
            rec["T2"][v] = {"mean_abs_dp_fractured": float(dp[y == 1].mean()), "mean_abs_dp_negative": float(dp[y == 0].mean()),
                            "max_abs_dp": float(dp.max()), "decision_flips@0.5": int(((pv >= .5) != (base >= .5)).sum()),
                            "d_auprc": float(average_precision_score(y, pv) - rec["auprc"]),
                            "dp_fractured_signed": summarise(list((base - pv)[y == 1]),
                                                             [c for c, t in zip(comps, y) if t == 1])}
        ok = orows[read]
        oc = [r["comp"] for r in ok]
        orc = {"n": len(ok),
               "zdist": {c: float(np.mean([r[f"zdist|{c}"] for r in ok])) for c in ("native_ell",) + ORACLE},
               "w_in_gt": {c: float(np.mean([r[f"w_in_gt|{c}"] for r in ok])) for c in ("native_ell",) + ORACLE}}
        for a, bb in (("gt_rank", "shift_rank"), ("gt_binary", "shift_binary")):
            dp = [r[f"p|{a}"] - r[f"p|{bb}"] for r in ok]
            dl = np.array([r[f"logit|{a}"] - r[f"logit|{bb}"] for r in ok])
            dh = np.array([r[f"h|{a}"] - r[f"h|{bb}"] for r in ok])
            orc[f"{a}_minus_{bb}"] = {"dp": summarise(dp, oc), "mean_abs_dp": float(np.mean(np.abs(dp))),
                                      "dlogit": float(dl.mean()), "dlogit_h_les_route": float(dh.mean()),
                                      "dlogit_fusion_route": float((dl - dh).mean())}
        rec["oracle"] = orc
        res[read] = rec
    return res, {r: orows[r] for r in READS}


def main(a):
    torch.set_num_threads(a.threads)
    t0 = time.time()
    data = FoldData(a.fold, "crossfit")
    sfx = "_smoke" if a.limit else ""
    global READS
    READS = tuple(a.reads)
    report = {"fold": a.fold, "role": a.role, "reads": READS, "note": __doc__.split("\n\n")[1], "arms": {}}
    for tag in a.arms:
        model, started, ck = load_arm(tag, a.fold, a.device)
        assert model.cfg["vg_ground"] and model.cfg["vg_pool_read"] == "feature"
        res, rows = run_arm(model, data, a.role, a.limit)
        rec = {"arm": started["arm"], "overrides": started["overrides"], "checkpoint_sha256": ck,
               "head_norms": head_norms(model), "reads": res}
        report["arms"][tag] = rec
        dump_json(RUNS / "mechanism" / f"fold{a.fold}" / f"sharpen_{tag.replace('/', '_')}{sfx}.json", {**rec, "rows": rows})
        print(json.dumps({"arm": tag, "sec": round(time.time() - t0), **{
            r: {"auprc": round(v["auprc"], 4), "T2_perm_s0_absdp_frac": round(v["T2"]["perm_s0"]["mean_abs_dp_fractured"], 4),
                "rank_dp": round(v["oracle"]["gt_rank_minus_shift_rank"]["dp"]["mean"], 4),
                "zdist_rank": round(v["oracle"]["zdist"]["gt_rank"], 2)} for r, v in res.items()}}), flush=True)
        del model
        if a.device == "mps":
            torch.mps.empty_cache()
    report["seconds"] = time.time() - t0
    dump_json(RUNS / "mechanism" / f"fold{a.fold}" / (a.out or f"sharpen_summary{sfx}.json"), report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--role", default="model_val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reads", nargs="+", default=list(READS))
    main(ap.parse_args())
