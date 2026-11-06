"""V4 / placebo: does the local attention prefer the crop that overlaps the lesion?  Fold-0 model_val + test positives with GT boxes."""
import json, sys
from pathlib import Path
import numpy as np, torch

R2 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2"); sys.path.insert(0, str(R2))
from experiments.route_audit_dyn_v1 import load_arm, CAN
from experiments.v5 import study_v5 as S

torch.set_num_threads(4)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)


out = {}
for arm in ("V4", "v4_placebo"):
    model, spec, _, _ = load_arm(arm)
    ds, collate = S.batcher("0", spec)
    mha = model.local_tokens.attention
    fwd = mha.forward
    seen = []
    def f(q, k, v, *a, **kw):
        kw.update(need_weights=True, average_attn_weights=True)
        o, w = fwd(q, k, v, *a, **kw)
        seen.append(w.detach()); return o, w
    mha.forward = f
    for role in ("model_val", "test"):
        recs = ds[role].records
        rows = [i for i, r in enumerate(recs) if r["question"] == CAN and r["answer"] == "yes" and r.get("boxes")]
        res = dict(n=len(rows), no_proposal=0, any_hit=[], top_hit=[], chance=[], top_share=[], n_crops=[])
        with torch.no_grad():
            for i in rows:
                b = collate([ds[role][i]], role=role)
                bx = b["boxes"][0] or []
                if not bx:
                    res["no_proposal"] += 1; continue
                seen.clear()
                pb = S.public(b, spec)
                ml = model._prepare_masks(b["images"], pb["masks"], pb["boxes"], b["image_ids"])
                model._encode_core(b["images"], b["questions"], [None], ml, model.cfg["prompt_kind"], False)
                per = seen[0].mean(1).reshape(-1, 4).sum(-1)[: len(bx)]
                best = [max(iou(x[:4], g[:4]) for g in recs[i]["boxes"]) for x in bx]
                t = int(per.argmax())
                res["any_hit"].append(max(best) >= .1); res["top_hit"].append(best[t] >= .1)
                res["chance"].append(float(np.mean([v >= .1 for v in best]))); res["top_share"].append(float(per[t])); res["n_crops"].append(len(bx))
        # only rows with a choice inform the attention
        multi = [k for k, n in enumerate(res["n_crops"]) if n > 1 and res["any_hit"][k]]
        out[f"{arm}_{role}"] = dict(n_pos=res["n"], no_proposal=res["no_proposal"], any_crop_hits_lesion=float(np.mean(res["any_hit"])), top_attention_crop_hits=float(np.mean(res["top_hit"])),
                                    random_crop_hits=float(np.mean(res["chance"])), rows_with_choice=len(multi), top_hit_given_choice=float(np.mean([res["top_hit"][k] for k in multi])) if multi else None,
                                    chance_given_choice=float(np.mean([res["chance"][k] for k in multi])) if multi else None, top_share=float(np.mean(res["top_share"])))
        print(arm, role, out[f"{arm}_{role}"], flush=True)
    del model
json.dump(out, open(R2 / "reports/route_audit_v1/crop_attention.json", "w"), indent=1)
