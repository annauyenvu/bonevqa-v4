"""Parameter travel from the shared initialisation, L causal checks, dead parameters, P x T interaction, v1 ablations."""
import json, sys
from pathlib import Path
import numpy as np, torch

R2 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2"); sys.path.insert(0, str(R2))
from experiments.route_audit_dyn_v1 import load_arm, group, ap, CAN
from experiments.v5 import study_v5 as S

torch.set_num_threads(3)
V5 = R2 / "reports/v5_studies/factorial_fold0/fold_0"
out = {}
init = torch.load(R2 / "reports/perfold_PREtask_roundtrip_v3_1/fold_0_PREtask.pt", map_location="cpu", weights_only=False)["state"]
st = {a: torch.load(V5 / f"{a}_seed42/LAST.pt", map_location="cpu", weights_only=False)["state"] for a in ("B", "T", "P", "PT", "V4", "V", "L", "VLPT")}
tr = {}
for a, s in st.items():
    g = {}
    for n, v in s.items():
        e = g.setdefault(group(n), dict(d_init=0.0, n_init=0.0, d_B=0.0, new=0))
        if n in init:
            e["d_init"] += float((v.float() - init[n].float()).pow(2).sum()); e["n_init"] += float(init[n].float().pow(2).sum()); e["d_B"] += float((v.float() - st["B"][n].float()).pow(2).sum())
        else:
            e["new"] += 1
    tr[a] = {k: dict(rel_init=(v["d_init"] / max(v["n_init"], 1e-12)) ** .5, dist_B=v["d_B"] ** .5, new=v["new"]) for k, v in g.items()}
out["travel"] = tr
for a in tr:
    print(a, {k: (round(v["rel_init"], 3), round(v["dist_B"], 2)) for k, v in tr[a].items() if k in ("visual.", "fusion.", "fracture_head.", "qformer.")}, flush=True)
# frozen vs trainable: which trainable parameters got no gradient at all
model, spec, _, _ = load_arm("L")
ds, collate = S.batcher("0", spec)
recs = ds["model_val"].records
can = [i for i, r in enumerate(recs) if r["question"] == CAN]
y = np.array([recs[i]["answer"] == "yes" for i in can], int)


def score(m):
    p = []
    with torch.no_grad():
        for off in range(0, len(can), 16):
            b = collate([ds["model_val"][i] for i in can[off: off + 16]], role="model_val")
            core = m._encode_core(b["images"], b["questions"], [None] * len(b["images"]), [np.zeros((0, 1, 1), np.uint8)] * len(b["images"]), m.cfg["prompt_kind"], False)
            p.append(core["fracture_logit"].float().numpy())
    return np.concatenate(p)


clean = score(model)
sd = model.state_dict()
keep = {k: v.clone() for k, v in sd.items() if k.startswith("v5_lens_fusion")}
torch.manual_seed(0)
for k in keep:
    sd[k].copy_(torch.randn_like(sd[k].float()).to(sd[k].dtype))
rand = score(model)
for k, v in keep.items():
    sd[k].copy_(v)
lora = {k: v.clone() for k, v in sd.items() if k.startswith("visual.") and k in st["B"]}
for k in lora:
    sd[k].copy_(st["B"][k])
swapped = score(model)
for k, v in lora.items():
    sd[k].copy_(v)
out["L_causal"] = dict(ap_clean=ap(y, clean), lens_module_randomised_max_dlogit=float(np.abs(rand - clean).max()), lora_from_B_ap=ap(y, swapped), lora_from_B_mean_abs_dlogit=float(np.abs(swapped - clean).mean()),
                       lora_from_B_flip=float(((swapped >= 0) != (clean >= 0)).mean()))
print(out["L_causal"], flush=True)
named = [n for n, p in model.named_parameters() if p.requires_grad and n.startswith("visual.")]
out["visual_trainable_sample"] = [n for n in named if "lora" not in n]
del model
# P x T interaction on test (only complete 2x2 inside the factorial)
def fp(a, role="test"):
    return [r for r in json.load(open(V5 / f"{a}_seed42/{role}_predictions.json"))["records"] if r["question"] == CAN]
for role in ("model_val", "test"):
    R = {a: fp(a, role) for a in ("B", "T", "P", "PT")}
    y2 = np.array([r["gt"] == "yes" for r in R["B"]], int)
    P = {a: np.array([r["p_yes"] for r in v]) for a, v in R.items()}
    rng = np.random.default_rng(20261101)
    f = lambda s: (ap(y2[s], P["PT"][s]) - ap(y2[s], P["P"][s])) - (ap(y2[s], P["T"][s]) - ap(y2[s], P["B"][s]))
    d = [f(rng.integers(0, len(y2), len(y2))) for _ in range(2000)]
    out[f"PxT_{role}"] = dict(ap={a: ap(y2, p) for a, p in P.items()}, interaction=f(np.arange(len(y2))), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    print(role, out[f"PxT_{role}"], flush=True)
# detector availability on TRAIN (what V4 was trained with)
boxes = json.load(open(R2 / "reports/local_noaux_refinement_v1/fold_0/boxes.json"))
avail = {}
for role in ("train", "model_val", "calibration", "test"):
    rs = [r for r in ds[role].records if r["question"] == CAN]
    a = np.array([bool(boxes.get(r["image_id"])) for r in rs]); yy = np.array([r["answer"] == "yes" for r in rs])
    avail[role] = dict(n=len(rs), active_pos=float(a[yy].mean()), active_neg=float(a[~yy].mean()))
out["availability_by_role"] = avail
print(avail, flush=True)
json.dump(out, open(R2 / "reports/route_audit_v1/extra.json", "w"), indent=1)
