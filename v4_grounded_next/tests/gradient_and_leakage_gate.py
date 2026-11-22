"""Fast pre-training gate for VNextGroundedModel (fold-0 TRAIN fixture, CPU fp32).

  P  parity    : G=C=R=off reproduces the sealed VNextModel (V4) fracture logits
  A  gradients : L_ground and L_contrast each reach visual LoRA + fusion layers
  B  causality : M_hat -> p_f, x_f, closed logits (autograd + finite interventions)
  F  flag ctrl : flag-only arm is invariant to crop content, crop arm is not
  C  leakage   : eval mode refuses any GT mask / box / legacy key
  M  mps       : optional (--mps) CPU vs MPS forward/backward parity probe

Usage:  source v4_grounded_next/env.sh && $PY tests/gradient_and_leakage_gate.py [--mps] [--with-llm]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get("PYTHONPYCACHEPREFIX") and os.environ.get("HF_HUB_OFFLINE") == "1", "source v4_grounded_next/env.sh"

import numpy as np
import torch

from vg.data import load_image, public_batch, training_batch
from vg.detector import load_detector, predict_scored
from vg.model import VNextGroundedModel
from vg.paths import CANONICAL_Q, RUNS, V4_REPORT, dump_json, manifest_fold, read_json, sha256
from vg.registry import arm_model_cfg

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise AssertionError(name + " " + str(detail))


def build(arm, payload, base_cfg, device="cpu"):
    torch.manual_seed(42)
    m = VNextGroundedModel(arm_model_cfg(base_cfg, arm), payload["answer_vocab"], device=device)
    own = m.state_dict()
    state = {k: v for k, v in payload["state"].items() if k in own}
    res = m.load_state_dict(state, strict=False)
    assert not res.unexpected_keys
    m.sync_global_head()
    return m.to(device)


def fixture():
    roles = manifest_fold(0)
    recs = [r for r in read_json(roles["train"]) if r["question"] == CANONICAL_Q]
    v4 = read_json(V4_REPORT / "fold_0/boxes.json")
    frac = [r for r in recs if r["fractured"] and v4[r["image_id"]]][:2]
    neg_with = [r for r in recs if not r["fractured"] and v4[r["image_id"]]][:1]
    neg_without = [r for r in recs if not r["fractured"] and not v4[r["image_id"]]][:1]
    rows = frac + neg_with + neg_without
    receipt = read_json(V4_REPORT / "fold_0/detector/receipt.json")
    assert sha256(receipt["checkpoint"]) == receipt["sha256"]
    det, _ = load_detector(receipt["checkpoint"])
    images = [load_image(r["image"]) for r in rows]
    props = {r["image_id"]: predict_scored(det, im)[0] for r, im in zip(rows, images)}
    for r in rows:  # scored runtime == sealed V4 boxes
        a = np.asarray([p[:4] for p in props[r["image_id"]]], dtype=float).reshape(-1, 4)
        b = np.asarray(v4[r["image_id"]], dtype=float).reshape(-1, 4)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-2), r["image_id"]
    return rows, images, props


def grads_of(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)
    g = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters() if p.grad is not None}
    assert all(np.isfinite(v) for v in g.values())
    return g


def nonzero(g, pred):
    hits = {n: v for n, v in g.items() if pred(n) and v > 0}
    return hits


def main(args):
    t0 = time.time()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = read_json(V4_REPORT / "fold_0/local_real_noaux.json")
    base = copy.deepcopy(cfg["model"])
    base["use_llm"] = bool(args.with_llm)
    base["grad_checkpoint"] = False
    payload = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
    rows, images, props = fixture()
    print(f"fixture: {[r['image_id'] for r in rows]} fractured={[r['fractured'] for r in rows]} "
          f"m={[bool(props[r['image_id']]) for r in rows]}", flush=True)

    # ---------------------------------------------------------------- P parity with sealed V4
    from experiments.vnext_model_v1 import VNextModel
    from src.data.fracatlas_vqa import FracAtlasVQADataset
    torch.manual_seed(42)
    sealed = VNextModel(base, payload["answer_vocab"], device="cpu")
    sealed.load_state_dict({k: v for k, v in payload["state"].items() if k in sealed.state_dict()}, strict=False)
    ours = build("V4", payload, base)
    sealed.eval(); ours.eval()
    with torch.no_grad():
        masks = [FracAtlasVQADataset._boxes_to_masks([p[:4] for p in props[r["image_id"]]], im.size)
                 for r, im in zip(rows, images)]
        ref = sealed._encode_core(images, [r["question"] for r in rows], [None] * 4, masks, "contour", False)
        mine = ours._encode(public_batch(rows, images, props, "test_fixture"))
    diff = float((ref["fracture_logit"] - mine["logit_local"]).abs().max())
    lt = all(torch.equal(a, b) for a, b in zip(sealed.local_tokens.state_dict().values(), ours.local_tokens.state_dict().values()))
    check("P.v4_parity_fracture_logit", diff < 1e-4 and lt, f"max|dlogit|={diff:.2e}, LocalTokens init identical={lt}")
    del sealed
    # identity-at-init: G / GC arms start exactly at V4 (zero-init lesion routes and h_les)
    for arm in ("V4-G", "V4-GC"):
        mg = build(arm, payload, base).eval()
        with torch.no_grad():
            lg_ = mg._encode(public_batch(rows, images, props, "test_fixture"))["logit_local"]
        d_ = float((lg_ - mine["logit_local"]).abs().max())
        check(f"P.identity_init_{arm}_equals_V4", d_ < 1e-6, f"max|dlogit|={d_:.2e}")
        del mg

    # ---------------------------------------------------------------- A gradients (V4-GCR)
    model = build("V4-GCR", payload, base)
    model.train()
    for mod in model.modules():  # deterministic gradient probe; dropout off, BN-free model
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    tb = training_batch(rows, images, props, "test_fixture", model.cfg)
    n_neg = [len(c["negatives"]) if c else 0 for c in tb["contrast_crops"]]
    check("A.fixture_supervision", sum(m.sum() > 0 for m in tb["gt_masks"]) == 2 and n_neg[:2] == [4, 4],
          f"gt mask px>0 on fractured, negatives per fractured={n_neg[:2]} relax={[c['relax'] if c else None for c in tb['contrast_crops'][:2]]}")
    out = model(tb, stage=3)
    check("A.losses_finite", all(torch.isfinite(out[k]).all() for k in ("loss", "loss_ground", "loss_contrast", "loss_recovery")),
          f"loss={float(out['loss']):.4f} ground={float(out['loss_ground']):.4f} contrast={float(out['loss_contrast']):.4f} "
          f"recovery={float(out['loss_recovery']):.4f}")
    # L_ground alone
    canon = list(range(4))
    core = model._encode(tb)
    lg = model._ground_loss(core, tb, canon)[0]
    g = grads_of(model, lg)
    lora = nonzero(g, lambda n: n.startswith("visual.") and "lora" in n)
    fuse = nonzero(g, lambda n: n.startswith("local_tokens.attention") or n.startswith("local_tokens.question"))
    gh = nonzero(g, lambda n: n.startswith("ground_head."))
    check("A.ground_to_visual_lora", len(lora) > 0, f"{len(lora)} LoRA tensors, sum|g|={sum(lora.values()):.3e}")
    check("A.ground_to_fusion", {"local_tokens.attention.in_proj_weight", "local_tokens.question.weight"} <= set(fuse),
          f"{sorted(fuse)}")
    check("A.ground_to_head", len(gh) > 0, f"{len(gh)} tensors")
    leak = nonzero(g, lambda n: n.startswith("fusion.") or n.startswith("closed_head.") or n.startswith("global_fracture_head."))
    check("A.ground_does_not_touch_downstream", not leak, f"unexpected={sorted(leak)[:4]}")
    # L_contrast alone
    lc = model._contrast_loss(tb, [0, 1])
    g = grads_of(model, lc)
    lora_c = nonzero(g, lambda n: n.startswith("visual.") and "lora" in n)
    check("A.contrast_to_visual_lora", len(lora_c) > 0, f"{len(lora_c)} LoRA tensors, sum|g|={sum(lora_c.values()):.3e}")
    check("A.contrast_to_fusion_query", "local_tokens.question.weight" in g and g["local_tokens.question.weight"] > 0,
          f"|g(local_tokens.question)|={g.get('local_tokens.question.weight', 0):.3e}")
    check("A.contrast_projections", g.get("contrast_img.1.weight", 0) > 0 and g.get("contrast_query.weight", 0) > 0)
    # full loss reaches everything trainable that should train
    out = model(tb, stage=3)
    g = grads_of(model, out["loss"])
    for prefix in ("recovery_gate.", "global_fracture_head.", "lesion_pool.", "les_head.", "fusion.", "fracture_head.",
                   "ground_head.c_attn."):
        check(f"A.total_reaches_{prefix.rstrip('.')}", any(n.startswith(prefix) and v > 0 for n, v in g.items()))

    # ---------------------------------------------------------------- S shortcut closure
    gh_ = model.ground_head
    check("S.ground_head_has_no_box_channel", gh_.conv[0].in_channels == model.cfg["vg_hidden"] and not gh_.use_footprint,
          f"conv in_channels={gh_.conv[0].in_channels}")
    with torch.no_grad():
        pt = torch.randn(2, 196, model.cfg["d_model"]); qq = torch.randn(2, model.cfg["d_model"])
        cr = torch.randn(2, model.cfg["vg_max_k"], 4, model.cfg["d_model"]); va = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
        e1 = gh_(pt, qq, cr, va, None); e2 = gh_(pt, qq, cr, va, torch.rand(2, 1, 14, 14))
        e3 = gh_(pt, qq, torch.randn_like(cr), va, None)  # different crop content (LN makes scale moot)
    check("S.ground_head_ignores_box_map", torch.equal(e1, e2))
    check("S.ground_head_reads_crop_content", float((e1[0] - e3[0]).abs().max()) > 0 and torch.equal(e1[1], e3[1]),
          f"row with crop moves by {float((e1[0] - e3[0]).abs().max()):.2e}; m=0 row unchanged")
    unused = [n for n in ("lesion_pool.evidence.weight", "lesion_pool.evidence.bias") if g.get(n, 0) != 0]
    check("S.no_scalar_evidence_into_lesion_token", not unused and not hasattr(model, "w_map"), str(unused))
    with torch.enable_grad():
        model.eval()
        cs = model._encode(public_batch(rows, images, props, "test_fixture"))
        ev_g = torch.autograd.grad(cs["logit_local"].sum(), cs["evidence"], allow_unused=True)[0]
        model.train()
    check("S.fracture_logit_independent_of_Mhat_scalars", ev_g is None or float(ev_g.abs().sum()) == 0.0,
          "d logit_local / d (LSE, max, mean of M_hat) is structurally zero")

    # ---------------------------------------------------------------- B causality M_hat -> p_f / x_f
    # The M_hat -> x_f / logit routes are zero-initialised (identity at init) and open as
    # soon as their weights move.  Wiring is tested on a copy-in-place with small random
    # values in those weights (what a few optimizer steps produce).
    with torch.no_grad():
        gen = torch.Generator().manual_seed(7)
        lp = model.lesion_pool
        for lin in (lp.token, lp.emphasis, model.les_head[1]):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=gen) * 0.02)
    model.eval()
    pb = public_batch(rows, images, props, "test_fixture")
    with torch.no_grad():
        base_core = model._encode(pb)
    ell0 = base_core["ell"].detach().clone().requires_grad_(True)
    with torch.enable_grad(), model.intervene(ell=ell0):
        c1 = model._encode(pb)
        gl = torch.autograd.grad(c1["logit_local"].sum(), ell0, retain_graph=True)[0]
        gx = torch.autograd.grad(c1["fus"]["x_f"].pow(2).sum(), ell0, retain_graph=True)[0]
        gc = torch.autograd.grad(c1["logits_closed"].pow(2).sum(), ell0, retain_graph=True)[0]
        gp = torch.autograd.grad(c1["p_final"].sum(), ell0)[0]
    check("B.dlogit_local/dM", float(gl.abs().sum()) > 0, f"{float(gl.abs().sum()):.3e}")
    check("B.dx_f/dM (feeds ClosedHead+LLM prefix)", float(gx.abs().sum()) > 0, f"{float(gx.abs().sum()):.3e}")
    check("B.dclosed_logits/dM", float(gc.abs().sum()) > 0, f"{float(gc.abs().sum()):.3e}")
    check("B.dp_final/dM", float(gp[:3].abs().sum()) > 0, f"{float(gp.abs().sum()):.3e}")
    ivs = {"no_lesion": torch.full_like(base_core["ell"], -10.0)}
    blob = torch.full_like(base_core["ell"], -10.0); blob[:, :, 4:7, 4:7] = 6.0
    ivs["blob"] = blob
    deltas = {}
    with torch.no_grad():
        for k, v in ivs.items():
            with model.intervene(ell=v):
                cc = model._encode(pb)
            deltas[k] = {"p_local": float((cc["p_local"] - base_core["p_local"]).abs().max()),
                         "x_f": float((cc["fus"]["x_f"] - base_core["fus"]["x_f"]).abs().max()),
                         "closed": float((cc["logits_closed"] - base_core["logits_closed"]).abs().max())}
        dlocal = float((cc["p_local"] - base_core["p_local"]).abs().max())
    check("B.intervention_changes_p_local", all(d["p_local"] > 1e-3 for d in deltas.values()), str(deltas))
    check("B.intervention_changes_x_f", all(d["x_f"] > 1e-4 for d in deltas.values()))
    m0 = ~base_core["valid"].any(1)
    check("B.m0_rows_use_pure_global", bool(torch.allclose(base_core["p_final"][m0], base_core["p_global"][m0])) and
          bool((base_core["alpha"][m0] == 0).all()), f"m=0 rows={int(m0.sum())}")

    # ---------------------------------------------------------------- F flag control
    from src.data.fracatlas_vqa import FracAtlasVQADataset as D
    shifted = {r["image_id"]: [b + [p[4]] for b, p in zip(D._matched_placebo_boxes([p[:4] for p in props[r["image_id"]]], im.size) or [], props[r["image_id"]])]
               for r, im in zip(rows, images)}
    flag = build("V4-GCR-flag", payload, base).eval()
    with torch.no_grad():
        fa = flag._encode(public_batch(rows, images, props, "test_fixture"))["p_final"]
        fb = flag._encode(public_batch(rows, images, shifted, "placebo"))["p_final"]
        ca = model._encode(public_batch(rows, images, props, "test_fixture"))["p_final"]
        cb = model._encode(public_batch(rows, images, shifted, "placebo"))["p_final"]
    check("F.flag_invariant_to_crop_content_and_location", float((fa - fb).abs().max()) < 1e-6, f"{float((fa - fb).abs().max()):.2e}")
    check("F.crop_arm_sensitive_to_crop_content", float((ca - cb).abs().max()) > 1e-5, f"{float((ca - cb).abs().max()):.2e}")
    del flag

    # ---------------------------------------------------------------- C leakage
    model.eval()
    gt = training_batch(rows, images, props, "test_fixture", model.cfg)
    probes = {"gt_masks": {"gt_masks": gt["gt_masks"]},
              "gt_boxes": {"gt_boxes": [r["boxes"] for r in rows]},
              "contrast_crops": {"contrast_crops": gt["contrast_crops"]},
              "legacy_masks": {"masks": [np.ones((1, 4, 4))] * 4},
              "legacy_boxes": {"boxes": [[[0, 0, 5, 5]]] * 4},
              "gt_as_proposal_source": {"proposal_source": ["gt"] * 4}}
    for name, extra in probes.items():
        for fn in ("predict_batch", "infer", "forward"):
            b = dict(pb, **extra)
            raised = False
            try:
                getattr(model, fn)(b) if fn != "forward" else model(dict(b, answers=["no"] * 4, answer_types=["CLOSED"] * 4))
            except AssertionError:
                raised = True
            check(f"C.eval_refuses_{name}.{fn}", raised)
    model.train()
    try:
        model.infer(pb); raised = False
    except AssertionError:
        raised = True
    check("C.infer_requires_eval", raised)
    model.eval()
    res = model.predict_batch(pb)
    check("C.clean_eval_ok", all(0 <= r["p_yes"] <= 1 for r in res), str([round(r["p_yes"], 4) for r in res]))

    # ---------------------------------------------------------------- M mps parity
    if args.mps and torch.backends.mps.is_available():
        mm = build("V4-GCR", payload, base, device="mps").eval()
        model_cpu = build("V4-GCR", payload, base).eval()
        with torch.no_grad():
            pc = model_cpu._encode(pb)["p_final"]
            pm = mm._encode(pb)["p_final"].cpu()
        check("M.forward_cpu_vs_mps", float((pc - pm).abs().max()) < 1e-3, f"{float((pc - pm).abs().max()):.2e}")
        mm.train(); model_cpu.train()
        for mod in list(mm.modules()) + list(model_cpu.modules()):
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        gc_ = grads_of(model_cpu, model_cpu._ground_loss(model_cpu._encode(tb), tb, canon)[0])
        gm_ = grads_of(mm, mm._ground_loss(mm._encode(tb), tb, canon)[0])
        rel = max(abs(gc_[k] - gm_[k]) / max(gc_[k], 1e-8) for k in gc_ if k.startswith("ground_head."))
        check("M.ground_grad_cpu_vs_mps", rel < 1e-2, f"max rel err={rel:.2e}")

    report = {"passed": all(v["pass"] for v in RESULTS.values()), "checks": RESULTS, "wall_seconds": time.time() - t0,
              "init_from": cfg["init_from"], "torch": torch.__version__, "device": "cpu", "fixture": [r["image_id"] for r in rows]}
    dump_json(RUNS / "gates/gradient_and_leakage_gate.json", report)
    print(f"ALL {len(RESULTS)} CHECKS PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--with-llm", action="store_true")
    main(ap.parse_args())
