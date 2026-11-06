"""Static route audit: every BoneVQA checkpoint (v1-v5), switches, route parameters, learned gates.  CPU, read-only."""
import json, os, subprocess, sys
from pathlib import Path
import torch

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
OUT = ROOT / "refactor_v2/reports/route_audit_v1/static_checkpoints.json"
paths = subprocess.run(["find", str(ROOT), "-type", "f", "-size", "+100M", "(", "-name", "*.pt", "-o", "-name", "*.pth", ")",
                        "-not", "-path", "*/hf_cache/*", "-not", "-path", "*/references/*", "-not", "-path", "*/deps/*",
                        "-not", "-name", "optimizer*", "-not", "-name", "best.pt", "-not", "-name", "smoke.pt"],
                       capture_output=True, text=True).stdout.split("\n")
paths = sorted(p for p in paths if p)
# exp4 / mandatory / .evaluation_runtime keep only best checkpoints
paths += sorted(subprocess.run(["find", str(ROOT / ".evaluation_runtime"), str(ROOT / "exp4_multiseed"), str(ROOT / "mandatory_experiments_20261011"),
                                str(ROOT / "refactor_v2/checkpoints/fracatlas_clean_none_seed42_v2"), "-type", "f", "-name", "best.pt", "-o", "-name", "checkpoint_best.pt"],
                               capture_output=True, text=True).stdout.split())
paths = sorted(set(p for p in paths if "hf_cache" not in p and "deps" not in p))
SW = ["use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "use_region_in_prompt", "textual_prompt_style", "lens_training_only",
      "visual_route", "visual_force_zero", "lens_force_zero", "latent_force_zero", "latent_answer_bank_mode", "prompt_protocol_version", "prompt_variant",
      "use_fracture_head", "prompt_kind", "v5_local", "v5_hag", "v5_local_view", "v5_lens", "v5_latent", "v5_pf", "vnext_enabled", "vnext_aux_weight", "use_llm"]
rows = []
for k, p in enumerate(paths):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        rows.append(dict(path=p, error=repr(e)[:200])); continue
    if not isinstance(ck, dict):
        rows.append(dict(path=p, error=type(ck).__name__)); continue
    state = next((ck[x] for x in ("trainable", "state_dict", "model", "model_state", "trainable_state_dict", "state") if isinstance(ck.get(x), dict)), ck)
    cfg = next((ck[x] for x in ("cfg", "config", "model_cfg") if isinstance(ck.get(x), dict)), {})
    if isinstance(cfg.get("model"), dict):
        cfg = cfg["model"]
    r = dict(path=p, top=[x for x in ck if not isinstance(ck[x], torch.Tensor)][:12], n_tensors=sum(isinstance(v, torch.Tensor) for v in state.values()),
             sw={s: cfg.get(s) for s in SW if s in cfg}, gates={}, norms={})
    for n, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "gamma" in n and v.ndim == 0:
            r["gates"][n] = float(v)
        for g in ("prompt_mixer.visual_proj", "prompt_mixer.lens_proj", "lens_fusion", "fusion.lp_", "latent.", "local_tokens.", "v5_lens_fusion", "v5_gm", "v5_lf", "v5_prior", "v5_out_norm"):
            if n.startswith(g):
                r["norms"].setdefault(g, 0.0)
                r["norms"][g] += float(v.float().pow(2).sum())
        if n in ("local_tokens.gate.bias",):
            r["gates"][n] = float(v)
        if n == "local_tokens.gate.weight":
            r["gates"]["local_tokens.gate.weight_norm"] = float(v.norm())
    r["norms"] = {a: b ** 0.5 for a, b in r["norms"].items()}
    rows.append(r)
    print(k + 1, "/", len(paths), p.split("BONEVQA/")[1], r["sw"].get("use_visual_prompt"), r["gates"], flush=True)
    del ck, state
json.dump(rows, open(OUT, "w"), indent=1)
