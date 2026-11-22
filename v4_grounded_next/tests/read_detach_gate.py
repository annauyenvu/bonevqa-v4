"""Gate for vg_read_gamma / vg_read_detach / --readout-scope h_les.

  R.gamma1_exact          gamma=1, detach=False: z_les and logits identical to the default model
  R.gamma2_formula        gamma=2: w == M_hat^2 / sum M_hat^2 (direct power, fp64 reference)
  R.gamma_no_underflow    gamma=4 on ell = -30 everywhere: w finite, sums to 1, uniform
  R.detach_eval_identity  detach changes nothing in the forward (same logits)
  R.detach_task_no_ground grad of the fracture logit wrt GroundingHead params == 0 under detach
  R.detach_task_reaches   ... but reaches h_les, t_les, emphasis and the G1 patches (visual LoRA)
  R.nodetach_task_ground  without detach the same gradient wrt GroundingHead is non-zero
  R.ground_loss_reaches   an objective on ell (as L_ground) still reaches the GroundingHead under detach
  R.scope_h_les           make_optimizer(scope='h_les'): only les_head.* at lr*mult

    $PY -m tests.read_detach_gate
"""
from __future__ import annotations

import time

import torch

from tests.gradient_and_leakage_gate import check, fixture, RESULTS
from vg.data import public_batch
from vg.paths import RUNS, dump_json
from vg.trainer import build_model, make_optimizer


def opened(model):
    with torch.no_grad():  # make the zero-init routes non-trivial
        for p_ in (model.les_head[1].weight, model.lesion_pool.token.weight, model.lesion_pool.emphasis.weight):
            p_.add_(torch.randn_like(p_) * 0.02)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    return model


def grads(model, scalar, prefixes):
    named = dict(model.named_parameters())
    names = [n for n in named if n.startswith(prefixes) and named[n].requires_grad]
    gs = torch.autograd.grad(scalar, [named[n] for n in names], allow_unused=True, retain_graph=True)
    return {n: (0.0 if g is None else float(g.abs().sum())) for n, g in zip(names, gs)}


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    rows, images, props = fixture()
    b = public_batch(rows, images, props, "test_fixture")

    torch.manual_seed(0)
    base, _ = build_model("V4-GC", "0", device="cpu", use_llm=False)
    sd = opened(base).state_dict()
    variants = {}
    for name, ov in (("g1", {"vg_read_gamma": 1.0}), ("g2", {"vg_read_gamma": 2.0}),
                     ("g2d", {"vg_read_gamma": 2.0, "vg_read_detach": True}),
                     ("g1d", {"vg_read_detach": True})):
        m, _ = build_model("V4-GC", "0", device="cpu", overrides=ov, use_llm=False)
        m.load_state_dict(sd)  # identical weights to `base`
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        variants[name] = m.eval()
    base.eval()
    with torch.no_grad():
        cb = base._encode(dict(b)); c1 = variants["g1"]._encode(dict(b))
    check("R.gamma1_exact", torch.equal(cb["z_les"], c1["z_les"]) and torch.equal(cb["logit_local"], c1["logit_local"]),
          f"max|dz|={float((cb['z_les'] - c1['z_les']).abs().max()):.1e}")

    with torch.no_grad():
        c2 = variants["g2"]._encode(dict(b))
        w = variants["g2"].lesion_pool.last_w.double()
        m_hat = torch.sigmoid(c2["ell"].reshape(len(w), -1).double())
        ref = m_hat ** 2 / (m_hat ** 2).sum(1, keepdim=True)
    check("R.gamma2_formula", float((w - ref).abs().max()) < 1e-6, f"max|d|={float((w - ref).abs().max()):.1e}")

    lp = variants["g2"].lesion_pool
    old = lp.gamma; lp.gamma = 4.0
    ell = torch.full((2, 1, 14, 14), -30.0)
    g = torch.randn(2, 197, 768)
    with torch.no_grad():
        _, z, _ = lp(g, ell)
    w4 = lp.last_w
    lp.gamma = old
    check("R.gamma_no_underflow", bool(torch.isfinite(w4).all()) and float((w4.sum(1) - 1).abs().max()) < 1e-5
          and float((w4 - 1 / 196).abs().max()) < 1e-6, f"sum={w4.sum(1).tolist()}")

    with torch.no_grad():
        c2d = variants["g2d"]._encode(dict(b))
    check("R.detach_eval_identity", torch.equal(c2["logit_local"], c2d["logit_local"]),
          f"max|d|={float((c2['logit_local'] - c2d['logit_local']).abs().max()):.1e}")

    gh = ("ground_head.",)
    ro = ("les_head.", "lesion_pool.token.", "lesion_pool.emphasis.")
    for name, detach in (("g2d", True), ("g2", False)):
        m = variants[name].train()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        core = m._encode(dict(b))
        s = core["logit_local"].sum()
        gg = grads(m, s, gh)
        if detach:
            check("R.detach_task_no_ground", max(gg.values()) == 0.0, f"max|grad| ground_head={max(gg.values()):.1e}")
            gr = grads(m, s, ro)
            vis = grads(m, s, ("visual.",))
            check("R.detach_task_reaches", all(gr[n] > 0 for n in gr if n.endswith("weight")) and sum(vis.values()) > 0,
                  f"readout={ {k: f'{v:.1e}' for k, v in gr.items() if k.endswith('weight')} } visual={sum(vis.values()):.1e}")
            # L_ground reads ell directly (Dice/BCE on sigmoid(ell)); proxy objective on ell
            ellsum = grads(m, torch.sigmoid(core["ell"]).sum(), gh)
            check("R.ground_loss_reaches", sum(ellsum.values()) > 0, f"sum|grad|={sum(ellsum.values()):.1e}")
        else:
            check("R.nodetach_task_ground", sum(gg.values()) > 0, f"sum|grad| ground_head={sum(gg.values()):.1e}")

    m = variants["g2d"]
    _, opt = make_optimizer(m, 1e-4, 10.0, "h_les")
    named = {id(p): n for n, p in m.named_parameters()}
    hi = [named[id(p)] for p in opt.param_groups[1]["params"]]
    check("R.scope_h_les", hi and all(n.startswith("les_head.") for n in hi) and opt.param_groups[1]["lr"] == 1e-3
          and not any(n.startswith("lesion_pool.") for n in hi), f"high-lr params={hi}")

    dump_json(RUNS / "gates/read_detach_gate.json", {"passed": True, "checks": RESULTS, "seconds": time.time() - t0})
    print(f"ALL {len(RESULTS)} CHECKS PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
