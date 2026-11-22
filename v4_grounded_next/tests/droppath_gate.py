"""Gate for vg_xf_droppath (training-only DropPath on the FractureHead(x_f) route).

  D.eval_identity        eval logits with p=0.3 == p=0 (same weights, h_les made non-zero)
  D.dropped_rows_exact   train mode, dropped row: logit == b_out + h_les(z_les)
  D.kept_rows_exact      kept row: logit == b_out + (f(x_f) - b_out)/(1-p) + h_les(z_les)
  D.dropped_rows_no_xf_grad   d logit[dropped] / d FractureHead hidden weights == 0
  D.dropped_rows_use_h_les    d logit[dropped] / d h_les weights != 0
  D.dropped_rows_reach_M_hat  d logit[dropped] / d ground_head params != 0 (via z_les)

    $PY -m tests.droppath_gate
"""
from __future__ import annotations

import time

import torch

from tests.gradient_and_leakage_gate import check, fixture, RESULTS
from vg.data import public_batch
from vg.paths import RUNS, dump_json
from vg.trainer import build_model

P = 0.3


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    rows, images, props = fixture()
    model, _ = build_model("V4-GC", "0", device="cpu", overrides={"vg_xf_droppath": P}, use_llm=False)
    with torch.no_grad():  # open the zero-init lesion routes so h_les / t_les are non-trivial
        for p_ in (model.les_head[1].weight, model.lesion_pool.token.weight, model.lesion_pool.emphasis.weight):
            p_.add_(torch.randn_like(p_) * 0.02)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    b = public_batch(rows, images, props, "test_fixture")

    model.eval()
    with torch.no_grad():
        e1 = model._encode(dict(b))["logit_local"]
        model.cfg["vg_xf_droppath"] = 0.0
        e0 = model._encode(dict(b))["logit_local"]
        model.cfg["vg_xf_droppath"] = P
    check("D.eval_identity", torch.equal(e1, e0), f"max|d|={float((e1 - e0).abs().max()):.1e}")

    model.train()
    found = None
    for seed in range(50):  # a draw with at least one dropped and one kept row
        torch.manual_seed(seed)
        core = model._encode(dict(b))
        k = core["xf_keep"]
        if (k == 0).any() and (k == 1).any():
            found = seed; break
    assert found is not None
    keep = core["xf_keep"].detach()
    f = model.fracture_head(core["fus"]["x_f"]).detach()
    h = model.les_head(core["z_les"]).squeeze(-1).detach()
    bo = model.fracture_head.net[-1].bias.detach()
    lg = core["logit_local"]
    dr, kp = keep == 0, keep == 1
    d_drop = float((lg.detach()[dr] - (bo + h[dr])).abs().max())
    d_keep = float((lg.detach()[kp] - (bo + (f[kp] - bo) / (1 - P) + h[kp])).abs().max())
    check("D.dropped_rows_exact", d_drop < 1e-5, f"max|d|={d_drop:.1e}, dropped={int(dr.sum())}/{len(keep)}")
    check("D.kept_rows_exact", d_keep < 1e-5, f"max|d|={d_keep:.1e}")

    i = int(torch.nonzero(dr)[0])
    named = dict(model.named_parameters())
    targets = {"xf": ["fracture_head.net.0.weight"], "h": ["les_head.1.weight"],
               "ground": [n for n in named if n.startswith("ground_head.") and named[n].requires_grad]}
    flat = [n for v in targets.values() for n in v]
    gs = torch.autograd.grad(lg[i], [named[n] for n in flat], allow_unused=True)
    gmap = {n: (0.0 if g is None else float(g.abs().sum())) for n, g in zip(flat, gs)}
    check("D.dropped_rows_no_xf_grad", gmap["fracture_head.net.0.weight"] == 0.0, f"{gmap['fracture_head.net.0.weight']:.1e}")
    check("D.dropped_rows_use_h_les", gmap["les_head.1.weight"] > 0, f"{gmap['les_head.1.weight']:.2e}")
    gg = sum(gmap[n] for n in targets["ground"])
    check("D.dropped_rows_reach_M_hat", gg > 0, f"sum|grad| ground_head={gg:.2e}")

    dump_json(RUNS / "gates/droppath_gate.json", {"passed": True, "checks": RESULTS, "p": P, "seconds": time.time() - t0})
    print(f"ALL {len(RESULTS)} CHECKS PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
