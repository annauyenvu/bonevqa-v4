"""Route audit on real checkpoints, CPU, read-only (no weights saved, no study artifact touched).

    python route_audit_dyn_v1.py <arm> [role]

Layers: (1) wiring: gradients per loss, parameter travel, padding/attention-mask invariance, hidden-state shift;
(2) inference interventions: zero / matched noise / swap / location shift / dose; (3) attention mass, Grad-CAM,
attention rollout.  Every arm first reproduces its saved predictions, otherwise the audit of that arm fails.
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
R2 = ROOT / "refactor_v2"
OUT = R2 / "reports/route_audit_v1"
V5 = R2 / "reports/v5_studies/factorial_fold0/fold_0"
V4 = R2 / "reports/local_noaux_refinement_v1/fold_0"
CAN = "Is there a fracture in this X-ray?"
BATCH, N_OPEN, N_OTHER, THREADS = 16, 60, 150, int(__import__("os").environ.get("AUDIT_THREADS", "8"))
TEMPLATE = {"clinical": "This is a bone X-ray image. Answer the question briefly. ",
            "irrelevant": "This input follows a standard presentation format. Answer the question briefly. ",
            "clinical_paraphrase": "Review the provided bone X-ray image and give a concise answer. "}


# ------------------------------------------------------------------ arms
def load_arm(arm):
    from experiments.v5.model_v5 import BoneVQAv5
    from experiments.v5 import study_v5 as S

    if arm in ("B", "T", "P", "PT", "V4", "V", "L", "VLPT"):
        path, spec = V5 / f"{arm}_seed42/LAST.pt", S.arm_spec(arm)
        model = BoneVQAv5.load(str(path), device="cpu").eval()
        saved = V5 / f"{arm}_seed42"
    elif arm in ("v4_B", "v4_real", "v4_placebo"):
        name = {"v4_B": "B", "v4_real": "local_real_noaux", "v4_placebo": "local_placebo_noaux"}[arm]
        path, saved = V4 / f"{name}/LAST.pt", V4 / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = {k: v for k, v in payload["config"].items() if not k.startswith("vnext_")}
        spec = {"model": {}, "proposals": "none"}
        if arm != "v4_B":
            cfg["v5_local"] = True
            spec = {"model": {"v5_local": True}, "proposals": "detector" if arm == "v4_real" else "placebo"}
        model = BoneVQAv5(cfg, payload["answer_vocab"], device="cpu")
        _, unexpected = model.load_state_dict(payload["state"], strict=False)
        assert not unexpected, unexpected
        model.eval()
    else:
        raise ValueError(arm)
    return model, spec, Path(path), saved


# -------------------------------------------------------------- intervention plumbing
class Plumbing:
    """Patches instance methods once; ``self.a`` selects the active intervention (empty = identical forward)."""

    def __init__(self, model):
        self.m, self.a, self.rng, self.memo = model, {}, torch.Generator().manual_seed(20261101), {}
        self.same = None
        self.v_fwd = model.visual.forward
        model.visual.forward = self.visual
        self.q_text = model._question_text
        model._question_text = self.question_text
        self.t_fwd = model.text.forward
        model.text.forward = self.text
        c = model.cfg
        if c["v5_local"]:
            self.lt_fwd = model.local_tokens.forward
            model.local_tokens.forward = self.local
        if c["v5_hag"] != "off":
            self.hag = model._hag_prefix
            model._hag_prefix = self.hag_prefix
        if c["v5_latent"]:
            self.lf_fwd = model.v5_lf.forward
            model.v5_lf.forward = self.lf
            if c["v5_pf"] != "none":
                self.pf_fwd = model.v5_prior.forward
                model.v5_prior.forward = self.pf

    # memoised image encoder: interventions downstream of the ViT reuse the same tokens
    def visual(self, pixel):
        if self.a.get("no_memo"):
            return self.v_fwd(pixel)
        keys = [hash(x.numpy().tobytes()) for x in pixel]
        miss = sorted({i for i, k in enumerate(keys) if k not in self.memo}, key=lambda i: i)
        first = {}
        for i in miss:
            first.setdefault(keys[i], i)
        if first:
            ids = list(first.values())
            out = self.v_fwd(pixel[ids])
            for j, i in enumerate(ids):
                self.memo[keys[i]] = (out["tokens"][j], out["embed"][j])
        return {"tokens": torch.stack([self.memo[k][0] for k in keys]), "embed": torch.stack([self.memo[k][1] for k in keys])}

    def noise_like(self, x, valid=None):
        ref = x if valid is None else x[valid]
        mean, std = ref.mean(0 if ref.ndim == 2 else (0, 1)), ref.std(0 if ref.ndim == 2 else (0, 1))
        return mean + std * torch.randn(x.shape, generator=self.rng)

    def donor(self, n):
        """Row -> another row of the batch showing a different image."""
        out = []
        for i in range(n):
            j = next((k % n for k in range(i + 1, i + n) if self.same[k % n] != self.same[i]), i)
            out.append(j)
        return torch.tensor(out)

    def question_text(self, question, region):
        mode = self.a.get("text")
        if mode == "raw":
            return question
        if mode in TEMPLATE:
            return TEMPLATE[mode] + question
        return self.q_text(question, region)

    def text(self, texts, device):
        out = self.t_fwd(texts, device)
        mode = self.a.get("text_tokens")
        if mode and self.m.cfg.get("use_textual_prompt"):
            n = len(self.m.text.tokenizer(TEMPLATE["clinical"].strip(), add_special_tokens=False)["input_ids"])
            out = dict(out)
            if mode == "mask":
                mask = out["mask"].clone()
                mask[:, 1: 1 + n] = False
                out["mask"] = mask
            else:
                tok = out["tokens"].clone()
                tok[:, 1: 1 + n] = self.noise_like(tok[:, 1: 1 + n])
                out["tokens"] = tok
        return out

    def local(self, g, t, m, crops, valid):
        mode = self.a.get("local")
        if mode == "crop_zero":
            crops = torch.zeros_like(crops)
        elif mode == "crop_noise" and valid.any():
            crops = self.noise_like(crops, valid) * valid[:, :, None, None] + crops * (~valid)[:, :, None, None]
        elif mode == "crop_swap":
            d = self.donor(g.shape[0])
            crops, valid = crops[d], valid[d]
        elif mode == "global_swap":
            d = self.donor(g.shape[0])
            g, t, m = g[d], t, m
        fi, aux, active, delta = self.lt_fwd(g, t, m, crops, valid)
        scale = 0.0 if mode == "delta_zero" else float(self.a.get("local_scale", 1.0))
        return g + delta * scale, aux, active, delta * scale

    def hag_prefix(self, g, raw, slots, base):
        mode = self.a.get("hag")
        if mode == "zero" or raw is None:
            if mode == "zero":
                return self.hag(g, None, [[] for _ in slots], base)
        elif mode == "noise":
            raw = self.noise_like(raw)
        elif mode == "swap":
            d = self.donor(len(slots)).tolist()
            new_slots = [slots[d[i]] for i in range(len(slots))]
            return self.hag(g, raw, new_slots, base)
        return self.hag(g, raw, slots, base)

    def _lat(self, x, mode):
        if mode == "zero":
            return torch.zeros_like(x)
        if mode == "noise":
            return self.noise_like(x)
        if mode == "swap":
            return x[self.donor(x.shape[0])]
        return x

    def lf(self, *args):
        return self._lat(self.lf_fwd(*args), self.a.get("lf"))

    def pf(self, *args):
        return self._lat(self.pf_fwd(*args), self.a.get("pf"))


def interventions(arm, model):
    c = model.cfg
    plan = {"img_blank": {"image": "blank"}, "img_swap": {"image": "swap"}, "q_swap": {"question": "swap"}}
    if c["v5_local"]:
        plan.update({"local_delta_zero": {"local": "delta_zero"}, "local_crop_zero": {"local": "crop_zero"},
                     "local_crop_noise": {"local": "crop_noise"}, "local_crop_swap": {"local": "crop_swap"},
                     "local_global_swap": {"local": "global_swap"},
                     "box_none": {"boxes": "none"}, "box_shift": {"boxes": "shift"}, "box_grid": {"boxes": "grid"},
                     "box_gt_on_positives": {"boxes": "gt"}})
        for s in (0.25, 0.5, 2.0, 4.0, 16.0) if arm in ("V4", "v4_real") else (0.5, 2.0):
            plan[f"local_scale_{s:g}"] = {"local_scale": s}
    if c["v5_hag"] != "off":
        plan.update({"hag_zero": {"hag": "zero"}, "hag_noise": {"hag": "noise"}, "hag_swap": {"hag": "swap"},
                     "v_all_zero": {"local": "delta_zero", "hag": "zero"}})
    if c["v5_latent"]:
        plan.update({"lf_zero": {"lf": "zero"}, "lf_noise": {"lf": "noise"}, "lf_swap": {"lf": "swap"}})
        if c["v5_pf"] != "none":
            plan.update({"pf_zero": {"pf": "zero"}, "pf_noise": {"pf": "noise"}, "pf_swap": {"pf": "swap"},
                         "p_all_zero": {"lf": "zero", "pf": "zero"}})
    if c.get("use_textual_prompt"):
        plan.update({"t_removed": {"text": "raw"}, "t_irrelevant": {"text": "irrelevant"}, "t_paraphrase": {"text": "clinical_paraphrase"},
                     "t_tokens_masked": {"text_tokens": "mask"}, "t_tokens_noise": {"text_tokens": "noise"}})
    else:
        plan["t_added"] = {"text": "clinical"}  # a template the arm never saw: negative control for T
    return plan


# ------------------------------------------------------------------ data
def shift_boxes(boxes, size):
    w, h = size
    out = []
    for b in boxes or []:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x = (b[0] + w / 2) % max(w - bw, 1)
        y = (b[1] + h / 2) % max(h - bh, 1)
        out.append([x, y, x + bw, y + bh])
    return out


def rows_for(ds, role, collate):
    recs = ds[role].records
    idx = list(range(len(recs)))
    random.Random(20261101).shuffle(idx)
    opens = [i for i in idx if recs[i]["answer_type"] == "OPEN"][:N_OPEN]
    others = [i for i in idx if recs[i]["answer_type"] != "OPEN" and recs[i]["question"] != CAN][:N_OTHER]
    keep = [i for i in idx if recs[i]["question"] == CAN] + others + opens
    random.Random(1).shuffle(keep)
    return recs, keep


def run_batch(model, plumb, b, spec, a, role_boxes):
    """Public forward of predict_batch without generation; returns endpoint tensors."""
    plumb.a = a
    n = len(b["images"])
    images, questions = list(b["images"]), list(b["questions"])
    ids = b["image_ids"]
    plumb.same = ids
    boxes = [list(x) if x else [] for x in b["boxes"]]
    d = plumb.donor(n).tolist()
    if a.get("image") == "blank":
        images = [Image.new("RGB", im.size, (128, 128, 128)) for im in images]
    elif a.get("image") == "swap":
        images = [images[j] for j in d]
        boxes = [boxes[j] for j in d]
        plumb.same = [ids[j] for j in d]
    if a.get("question") == "swap":
        questions = [questions[j] for j in d]
    if a.get("boxes") == "none":
        boxes = [[] for _ in boxes]
    elif a.get("boxes") == "shift":
        boxes = [shift_boxes(x, im.size) for x, im in zip(boxes, images)]
    elif a.get("boxes") == "grid":
        boxes = [[[0, 0, im.width / 2, im.height / 2], [im.width / 2, 0, im.width, im.height / 2], [0, im.height / 2, im.width / 2, im.height]] for im in images]
    elif a.get("boxes") == "gt":
        boxes = [[g[:4] for g in (gt or [])] or x for gt, x in zip(b["gt_boxes"], boxes)]
    from experiments.v5.study_v5 import _box_masks

    masks = [_box_masks(x, im.size) for x, im in zip(boxes, images)] if spec["proposals"] != "none" else [None] * n
    mask_list = model._prepare_masks(images, masks, boxes if spec["proposals"] != "none" else [None] * n, ids)
    core = model._encode_core(images, questions, [None] * n, mask_list, model.cfg["prompt_kind"], False,
                              context_group_ids=b.get("component_ids"))
    out = {"fracture": core["fracture_logit"].float(), "closed": core["logits_closed"].float(), "type": core["type_logit"].float(),
           "x_f": core["fus"]["x_f"].float(), "f_i": core["vis"]["f_i"].float(), "prefix": core["vis"]["prefix_vis"].float()}
    opens = [i for i, t in enumerate(b["answer_types"]) if t == "OPEN"]
    if opens and model.llm is not None:
        sub = model._subset_core(core, opens)
        embeds, att, _ = model._build_llm_inputs(model._llm_prefix(sub), [core["q_texts"][i] for i in opens], None)
        pos = (att.cumsum(1) - 1).clamp_min(0)
        out["open_idx"] = opens
        out["llm"] = model.llm(inputs_embeds=embeds, attention_mask=att, position_ids=pos).logits[:, -1].float().log_softmax(-1)
    return out


def intervene_all(arm, model, spec, ds, collate, role, saved, log):
    plumb = Plumbing(model)
    recs, keep = rows_for(ds, role, collate)
    plan = interventions(arm, model)
    if plumb.a is not None and not model.cfg["v5_local"]:
        pass
    yes, no = model.answer_to_idx["yes"], model.answer_to_idx["no"]
    canon = [i for i, r in enumerate(recs) if r["question"] == CAN]
    res = {name: dict(frac=np.full(len(recs), np.nan), top=np.full(len(recs), -1), logit=np.full(len(recs), np.nan), yes=np.full(len(recs), np.nan),
                      kl=np.full(len(recs), np.nan), t1=np.full(len(recs), -1), xf=np.full(len(recs), np.nan), fi=np.full(len(recs), np.nan),
                      pre=np.full(len(recs), np.nan)) for name in ["clean"] + list(plan)}
    t0 = time.time()
    with torch.no_grad():
        for off in range(0, len(keep), BATCH):
            ids = keep[off: off + BATCH]
            b = collate([ds[role][i] for i in ids], role=role)
            b["gt_boxes"] = [recs[i].get("boxes") for i in ids]
            b["answer_types"] = [recs[i]["answer_type"] for i in ids]
            plumb.memo = {}
            clean = run_batch(model, plumb, b, spec, {}, None)
            for name, a in [("clean", {})] + list(plan.items()):
                try:
                    out = clean if name == "clean" else run_batch(model, plumb, b, spec, a, None)
                except Exception as e:
                    log(f"FAILED {arm} {name}: {e!r}")
                    raise
                r = res[name]
                for k, i in enumerate(ids):
                    r["logit"][i] = float(out["fracture"][k])
                    r["top"][i] = int(out["closed"][k].argmax())
                    y = out["closed"][k, [yes, no]].softmax(0)
                    r["yes"][i] = float(y[0])
                    r["xf"][i] = float((out["x_f"][k] - clean["x_f"][k]).norm() / clean["x_f"][k].norm())
                    r["fi"][i] = float((out["f_i"][k] - clean["f_i"][k]).norm() / clean["f_i"][k].norm())
                    r["pre"][i] = float((out["prefix"][k] - clean["prefix"][k]).norm() / clean["prefix"][k].norm())
                if "llm" in out:
                    for p, k in enumerate(out["open_idx"]):
                        i = ids[k]
                        r["kl"][i] = float((clean["llm"][p].exp() * (clean["llm"][p] - out["llm"][p])).sum())
                        r["t1"][i] = int(out["llm"][p].argmax())
            if off % (BATCH * 10) == 0:
                log(f"{arm} interventions {off}/{len(keep)} {time.time() - t0:.0f}s")
    plumb.a = {}
    return plumb, recs, canon, res, keep


# ------------------------------------------------------------------ metrics
def ap(y, s):
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    return float((tp[y == 1] / (np.nonzero(y == 1)[0] + 1)).mean()) if y.sum() else float("nan")


def auroc(y, s):
    from scipy.stats import rankdata

    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(recs, canon, res, keep):
    y = np.array([1 if recs[i]["answer"].strip().lower() == "yes" else 0 for i in canon])
    comp = np.array([recs[i]["image_id"] for i in canon])
    rng = np.random.default_rng(20261101)
    closed = [i for i in keep if recs[i]["answer_type"] == "CLOSED" and recs[i]["question"] != CAN]
    table = {}
    base = res["clean"]
    p0 = 1 / (1 + np.exp(-base["logit"][canon]))
    draws = [rng.integers(0, len(canon), len(canon)) for _ in range(1000)]
    for name, r in res.items():
        p = 1 / (1 + np.exp(-r["logit"][canon]))
        row = dict(ap=ap(y, p), auroc=auroc(y, p), sens=float((p[y == 1] >= .5).mean()), spec=float((p[y == 0] < .5).mean()),
                   mean_abs_dlogit=float(np.abs(r["logit"][canon] - base["logit"][canon]).mean()), frac_flip=float(((p >= .5) != (p0 >= .5)).mean()),
                   closed_top_flip=float(np.mean(r["top"][closed] != base["top"][closed])), xf_shift=float(np.nanmean(r["xf"])), fi_shift=float(np.nanmean(r["fi"])),
                   prefix_shift=float(np.nanmean(r["pre"])), open_first_token_kl=float(np.nanmean(r["kl"])),
                   open_first_token_flip=float(np.mean(r["t1"][r["t1"] >= 0] != base["t1"][r["t1"] >= 0])) if (r["t1"] >= 0).any() else None,
                   mean_logit_pos=float(r["logit"][canon][y == 1].mean()), mean_logit_neg=float(r["logit"][canon][y == 0].mean()))
        if name != "clean":
            d = [ap(y[s], p[s]) - ap(y[s], p0[s]) for s in draws if y[s].sum() > 0]
            row["dap_vs_clean"] = row["ap"] - table["clean"]["ap"]
            row["dap_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        table[name] = row
    return table


def replay_check(recs, canon, res, saved, role):
    ref = json.load(open(saved / f"{role}_predictions.json"))["records"]
    p = 1 / (1 + np.exp(-res["clean"]["logit"][canon]))
    d = [abs(float(a) - ref[i]["p_yes"]) for a, i in zip(p, canon)]
    assert all(recs[i]["image_id"] == ref[i]["image_id"] for i in canon)
    return dict(n=len(d), max_abs_p_yes_diff=max(d), passed=max(d) < 1e-4)


# ------------------------------------------------------------------ wiring
GROUPS = ["visual.", "text.", "qformer.", "fusion.", "closed_head.", "fracture_head.", "type_head.", "prefix_proj.", "llm.", "img_proj.", "txt_proj.",
          "local_tokens.attention", "local_tokens.question", "local_tokens.gate", "local_tokens.norm", "local_tokens.mil_score", "local_tokens.aux_head",
          "gamma_hag", "v5_gm", "v5_lf", "v5_prior", "v5_out_norm", "v5_lens_fusion"]


def group(name):
    return next((g for g in GROUPS if name.startswith(g)), "other")


def gradient_flow(arm, model, spec, ds, collate, cfg_init):
    from experiments.v5.study_v5 import public

    model.eval()  # dropout off; gradients still flow
    recs = ds["train"].records
    random.seed(0)
    ids = random.Random(0).sample(range(len(recs)), 8)
    b = collate([ds["train"][i] for i in ids], role="train")
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    report = {}

    def per_group(loss):
        grads = torch.autograd.grad(loss, list(named.values()), retain_graph=True, allow_unused=True)
        g = {}
        for (n, p), v in zip(named.items(), grads):
            k = group(n)
            e = g.setdefault(k, dict(params=0, none=0, zero=0, norm2=0.0))
            e["params"] += 1
            if v is None:
                e["none"] += 1
            else:
                nv = float(v.float().pow(2).sum())
                e["norm2"] += nv
                e["zero"] += int(nv == 0)
        return {k: dict(params=v["params"], none=v["none"], zero=v["zero"], norm=v["norm2"] ** .5) for k, v in g.items()}

    # the LM term is expensive, so one forward for all losses
    out = model(b, stage=2)
    report["losses"] = {k: float(v) for k, v in out.items() if k.startswith("loss") and v.ndim == 0}
    report["total"] = per_group(out["loss"])
    report["answer_only"] = per_group(out["loss_answer"])
    if model.cfg["v5_lens"]:
        nb = dict(b)
        from src.models.bonevqa import normalize_batch

        nb = normalize_batch(b)
        report["lens_qa"] = per_group(model.lens_loss(nb, "qa"))
        report["lens_region"] = per_group(model.lens_loss(nb, "region"))
    # parameter travel since the shared initialisation
    init = torch.load(cfg_init, map_location="cpu", weights_only=False)["state"] if cfg_init else {}
    tr = {}
    state = model.trainable_state_dict()
    for n, v in state.items():
        k = group(n)
        e = tr.setdefault(k, dict(d2=0.0, n2=0.0, new=0))
        if n in init:
            e["d2"] += float((v.float() - init[n].float()).pow(2).sum())
            e["n2"] += float(init[n].float().pow(2).sum())
        else:
            e["new"] += 1
            e["n2"] += float(v.float().pow(2).sum())
    report["travel"] = {k: dict(rel=(v["d2"] / max(v["n2"], 1e-12)) ** .5, new_params=v["new"], norm=v["n2"] ** .5) for k, v in tr.items()}
    model.zero_grad(set_to_none=True)
    return report


@torch.no_grad()
def mask_invariance(arm, model, spec, ds, collate, role, plumb):
    """Padded content must not matter; a row must not depend on its batch; empty ROI must give an exact zero delta."""
    recs = ds[role].records
    ids = random.Random(3).sample(range(len(recs)), 8)
    b = collate([ds[role][i] for i in ids], role=role)
    b["gt_boxes"] = [None] * 8
    b["answer_types"] = ["CLOSED"] * 8
    plumb.memo = {}
    base = run_batch(model, plumb, b, spec, {}, None)
    rep = {}
    # row alone vs inside the batch (text padding, crop padding, HAG padding)
    d = []
    for k in range(8):
        one = {x: (b[x][k: k + 1] if isinstance(b.get(x), list) else b.get(x)) for x in b}
        o = run_batch(model, plumb, one, spec, {}, None)
        d.append(float((o["fracture"][0] - base["fracture"][k]).abs()))
    rep["row_alone_vs_batch_max_dlogit"] = max(d)
    # padded text tokens: replace by noise, output must be unchanged
    orig = model.text.forward

    def noisy_pad(texts, device):
        out = orig(texts, device)
        out = dict(out)
        tok = out["tokens"].clone()
        pad = ~out["mask"]
        tok[pad] = torch.randn(int(pad.sum()), tok.shape[-1]) * 5
        out["tokens"] = tok
        return out

    model.text.forward = noisy_pad
    o = run_batch(model, plumb, b, spec, {}, None)
    model.text.forward = orig
    rep["text_padding_noise_max_dlogit"] = float((o["fracture"] - base["fracture"]).abs().max())
    if model.cfg["v5_local"]:
        lt = model.local_tokens.forward

        def noisy_crops(g, t, m, crops, valid):
            crops = crops.clone()
            crops[~valid] = torch.randn(int((~valid).sum()), crops.shape[2], crops.shape[3]) * 5
            return lt(g, t, m, crops, valid)

        model.local_tokens.forward = noisy_crops
        o = run_batch(model, plumb, b, spec, {}, None)
        model.local_tokens.forward = lt
        rep["crop_padding_noise_max_dlogit"] = float((o["fracture"] - base["fracture"]).abs().max())
        captured = {}
        def spy(g, t, m, crops, valid):
            out = lt(g, t, m, crops, valid)
            captured["delta"], captured["active"], captured["g"] = out[3], out[2], g
            return out
        model.local_tokens.forward = spy
        run_batch(model, plumb, b, spec, {"boxes": "none"}, None)
        rep["no_roi_delta_abs_max"] = float(captured["delta"].abs().max())
        model.local_tokens.forward = lt
    return rep


# ------------------------------------------------------------------ attention mass
class AttnSpy:
    """need_weights=True on selected MultiheadAttention modules (outputs identical up to float noise)."""

    def __init__(self, modules):
        self.seen, self.undo = {}, []
        for name, mha in modules.items():
            fwd = mha.forward

            def f(q, k, v, *a, _n=name, _f=fwd, **kw):
                kw["need_weights"] = True
                kw["average_attn_weights"] = True
                o, w = _f(q, k, v, *a, **kw)
                self.seen.setdefault(_n, []).append(w.detach())
                return o, w

            mha.forward = f
            self.undo.append((mha, fwd))

    def close(self):
        for m, f in self.undo:
            m.forward = f


@torch.no_grad()
def attention_mass(arm, model, spec, ds, collate, role, plumb, keep, n=96):
    c = model.cfg
    recs = ds[role].records
    mods = {"fusion.img_ca": model.fusion.img_ca.attn, "fusion.txt_ca": model.fusion.txt_ca.attn}
    for i, (_, ca) in enumerate(model.qformer.layers):
        mods[f"qformer.ca{i}"] = ca.attn
    if c["v5_local"]:
        mods["local_tokens"] = model.local_tokens.attention
    if c["v5_latent"]:
        mods.update({"lf.ca_l": model.v5_lf.ca_l.attn, "lf.ca_i": model.v5_lf.ca_i.attn, "lf.ca_mm": model.v5_lf.ca_mm.attn, "gm.cross": model.v5_gm.cross.attn})
    spy = AttnSpy(mods)
    res = {k: [] for k in ("template_share", "question_share", "cls_sep_share", "template_uniform", "img_patch_share", "hag_local_share", "hag_local_uniform",
                           "local_crop_top", "local_crop_valid", "local_delta_rel", "local_gate", "lf_mm_img_share", "lf_mm_img_uniform", "gm_entropy_ratio",
                           "eq12_lp", "eq12_fi", "eq12_fl", "lf_l_entropy", "lf_i_entropy")}
    n_tpl = len(model.text.tokenizer(TEMPLATE["clinical"].strip(), add_special_tokens=False)["input_ids"]) if c.get("use_textual_prompt") else 0
    orig_lt = model.local_tokens.forward if c["v5_local"] else None
    cap = {}
    if c["v5_local"]:
        def lt(g, t, m, crops, valid):
            out = orig_lt(g, t, m, crops, valid)
            cap["delta"], cap["g"], cap["valid"], cap["sig"] = out[3], g, valid, torch.sigmoid(model.local_tokens.gate(g.mean(1))).squeeze(-1)
            return out
        model.local_tokens.forward = lt
    lapa = model._lapa if c["v5_latent"] else None
    if c["v5_latent"]:
        def wl(e_i, fus, txt):
            out = lapa(e_i, fus, txt)
            x = out["x_ii"][:, fus["f_fi"].shape[1] + fus["f_fl"].shape[1]:].mean(1)
            res["eq12_lp"].append(float((c["v5_alpha"] * x).norm(dim=-1).mean()))
            res["eq12_fi"].append(float((c["v5_theta"] * fus["f_fi"][:, 0]).norm(dim=-1).mean()))
            res["eq12_fl"].append(float((c["v5_beta"] * fus["f_fl"][:, 0]).norm(dim=-1).mean()))
            return out
        model._lapa = wl
    canon_first = [i for i in keep if recs[i]["answer_type"] == "CLOSED"][:n]
    for off in range(0, len(canon_first), BATCH):
        ids = canon_first[off: off + BATCH]
        b = collate([ds[role][i] for i in ids], role=role)
        b["gt_boxes"] = [None] * len(ids)
        b["answer_types"] = ["CLOSED"] * len(ids)
        spy.seen.clear()
        plumb.memo = {}
        tx = {}
        tf = model.text.forward
        def txt(t, d):
            o = tf(t, d)
            tx["mask"] = o["mask"]
            return o
        model.text.forward = txt
        run_batch(model, plumb, b, spec, {}, None)
        model.text.forward = tf
        w = spy.seen["fusion.img_ca"][0]                   # image queries -> text tokens
        mask = tx["mask"].float()
        lens = mask.sum(1)
        for k in range(w.shape[0]):
            wk = w[k].mean(0)
            l = int(lens[k])
            cls_sep = float(wk[0] + wk[l - 1])
            tpl = float(wk[1: 1 + n_tpl].sum())
            res["template_share"].append(tpl)
            res["cls_sep_share"].append(cls_sep)
            res["question_share"].append(1 - tpl - cls_sep)
            res["template_uniform"].append(n_tpl / l)
        wv = spy.seen["fusion.txt_ca"][0].mean(1)          # text queries -> image tokens
        res["img_patch_share"].append(float(wv[:, 1:].sum(1).mean()))
        if c["v5_hag"] != "off":
            q = spy.seen["qformer.ca1"][-1]                # the HAG call, after the global-only base prefix
            mem = q.shape[-1]
            n_tok = 197
            s = q.mean(1)
            res["hag_local_share"].append(float(s[:, n_tok:].sum(1).mean()))
            res["hag_local_uniform"].append(float(1 - n_tok / mem))
        if c["v5_local"] and "local_tokens" in spy.seen:
            a = spy.seen["local_tokens"][0].mean(1)        # [B, k*4]
            per = a.reshape(a.shape[0], -1, 4).sum(-1)
            v = cap["valid"]
            for k in range(per.shape[0]):
                nv = int(v[k].sum())
                if nv:
                    res["local_crop_top"].append(float(per[k][v[k]].max()))
                    res["local_crop_valid"].append(nv)
            act = v.any(1)
            if act.any():
                res["local_delta_rel"].append(float((cap["delta"][act].norm(dim=(1, 2)) / cap["g"][act].norm(dim=(1, 2))).mean()))
                res["local_gate"].append(float(cap["sig"][act].mean()))
        if c["v5_latent"]:
            mm = spy.seen["lf.ca_mm"][0].mean(1)
            nimg = 197
            res["lf_mm_img_share"].append(float(mm[:, :nimg].sum(1).mean()))
            res["lf_mm_img_uniform"].append(float(nimg / mm.shape[1]))
            g = spy.seen["gm.cross"][0].mean(1)[0].clamp_min(1e-12)
            res["gm_entropy_ratio"].append(float(-(g * g.log()).sum() / np.log(g.shape[0])))
            ei = spy.seen["lf.ca_i"][0].mean(1).clamp_min(1e-12)
            res["lf_i_entropy"].append(float((-(ei * ei.log()).sum(1) / np.log(ei.shape[1])).mean()))
    spy.close()
    if c["v5_local"]:
        model.local_tokens.forward = orig_lt
    if c["v5_latent"]:
        model._lapa = lapa
    return {k: float(np.mean(v)) for k, v in res.items() if v}


# ------------------------------------------------------------------ Grad-CAM + rollout
def lesion_grid(gt, size, side=14):
    g = np.zeros((side, side), bool)
    w, h = size
    for b in gt:
        x1, x2 = int(np.floor(b[0] / w * side)), int(np.ceil(b[2] / w * side))
        y1, y2 = int(np.floor(b[1] / h * side)), int(np.ceil(b[3] / h * side))
        g[max(y1, 0):min(y2, side), max(x1, 0):min(x2, side)] = True
    return g


def spatial_maps(arm, model, spec, ds, collate, role, plumb, boxes_key="boxes"):
    """Grad-CAM-like map on the global patch tokens for the fracture logit, and attribution share global vs crops."""
    recs = ds[role].records
    rows = [i for i, r in enumerate(recs) if r["question"] == CAN and r["answer"].strip().lower() == "yes" and r.get("boxes")]
    out = dict(n=len(rows), cam_lift=[], cam_hit=[], ctrl_lift=[], crop_share=[], rollout_lift=[], rollout_hit=[], rollout_ctrl=[])
    trunk = model.visual.trunk
    blocks = list(trunk.blocks)
    attns = []
    for blk in blocks:
        blk.attn.fused_attn = False
    hooks = [blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(i[0].detach())) for blk in blocks]
    v_fwd = plumb.v_fwd
    leaves = []

    def leaf(pixel):
        o = v_fwd(pixel)
        t = o["tokens"].detach().requires_grad_(True)
        leaves.append(t)
        return {"tokens": t, "embed": o["embed"]}

    keep = model.visual.forward
    model.visual.forward = leaf
    for param in model.parameters():
        param.requires_grad_(False)
    grids = {}
    for nrow, i in enumerate(rows):
        b = collate([ds[role][i]], role=role)
        attns.clear()
        leaves.clear()
        box = b["boxes"][0] or []
        from experiments.v5.study_v5 import _box_masks

        im = b["images"][0]
        masks = [_box_masks(box, im.size)] if spec["proposals"] != "none" else [None]
        ml = model._prepare_masks([im], masks, [box] if spec["proposals"] != "none" else [None], b["image_ids"])
        core = model._encode_core([im], b["questions"], [None], ml, model.cfg["prompt_kind"], False, context_group_ids=b.get("component_ids"))
        logit = core["fracture_logit"].sum()
        grads = torch.autograd.grad(logit, leaves, allow_unused=True)
        g0 = grads[0][0]                                   # global view of this row
        t0 = leaves[0][0]
        cam = (g0 * t0).sum(-1)[1:].reshape(14, 14).clamp_min(0).detach().numpy()
        tot = float((g0 * t0).abs().sum())
        crop = 0.0
        if len(leaves) > 2 or (len(leaves) == 2 and spec["proposals"] != "none"):
            for g, t in zip(grads[1:], leaves[1:]):
                if g is not None:
                    crop += float((g * t).abs().sum())
            out["crop_share"].append(crop / max(tot + crop, 1e-12))
        g = lesion_grid(recs[i]["boxes"], im.size)
        grids[i] = (g, cam)
        area = max(g.mean(), 1e-6)
        out["cam_lift"].append(float(cam[g].sum() / max(cam.sum(), 1e-12) / area))
        out["cam_hit"].append(bool(g.flat[int(cam.argmax())]))
        # rollout on the global view only (first ViT call)
        k = len(blocks)
        a = [x[0].mean(0) for x in attns[:k]]
        r = torch.eye(a[0].shape[0])
        for m in a:
            m = 0.5 * m + 0.5 * torch.eye(m.shape[0])
            r = m / m.sum(-1, keepdim=True) @ r
        ro = r[0, 1:].reshape(14, 14).numpy()
        out["rollout_lift"].append(float(ro[g].sum() / ro.sum() / area))
        out["rollout_hit"].append(bool(g.flat[int(ro.argmax())]))
        grids[i] = (g, cam, ro)
    # control: lesion mask of another positive image, same maps
    ids = list(grids)
    for p, i in enumerate(ids):
        g = grids[ids[(p + 1) % len(ids)]][0]
        _, cam, ro = grids[i]
        area = max(g.mean(), 1e-6)
        out["ctrl_lift"].append(float(cam[g].sum() / max(cam.sum(), 1e-12) / area))
        out["rollout_ctrl"].append(float(ro[g].sum() / ro.sum() / area))
    for h in hooks:
        h.remove()
    for blk in blocks:
        blk.attn.fused_attn = True
    model.visual.forward = keep
    summary = {k: (float(np.mean(v)) if v else None) for k, v in out.items() if k != "n"}
    summary["n"] = out["n"]
    return summary


# ------------------------------------------------------------------ main
def main():
    from experiments.v5 import study_v5 as S

    arm = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "model_val"
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    folder = OUT / "arms"
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def log(m):
        print(time.strftime("%H:%M:%S"), m, flush=True)

    model, spec, ck, saved = load_arm(arm)
    fold = "0"
    ds, collate = S.batcher(fold, spec)
    report = dict(arm=arm, role=role, checkpoint=str(ck), proposals=spec["proposals"])
    done = folder / f"{arm}_{role}.json"
    if done.exists() and "interventions" in json.load(open(done)):  # resume after a later stage failed
        report = json.load(open(done))
        plumb = Plumbing(model)
        recs, keep = rows_for(ds, role, collate)
    else:
        plumb, recs, canon, res, keep = intervene_all(arm, model, spec, ds, collate, role, saved, log)
        report["replay"] = replay_check(recs, canon, res, saved, role)
        log(f"replay {report['replay']}")
        assert report["replay"]["passed"], "audit harness does not reproduce the saved predictions"
        report["interventions"] = summarize(recs, canon, res, keep)
        np.savez_compressed(folder / f"{arm}_{role}_rows.npz", **{f"{n}__{k}": v for n, r in res.items() for k, v in r.items() if k in ("logit", "top", "t1", "kl")}, canon=np.array(canon))
    # routing data for the threshold sensitivity (router is the type head)
    json.dump(report, open(folder / f"{arm}_{role}.json", "w"), indent=1)
    log("interventions done")
    report["mask_invariance"] = mask_invariance(arm, model, spec, ds, collate, role, plumb)
    log(f"masks {report['mask_invariance']}")
    report["attention"] = attention_mass(arm, model, spec, ds, collate, role, plumb, keep)
    log(f"attention {report['attention']}")
    json.dump(report, open(folder / f"{arm}_{role}.json", "w"), indent=1)
    report["spatial"] = spatial_maps(arm, model, spec, ds, collate, role, plumb)
    log(f"spatial {report['spatial']}")
    json.dump(report, open(folder / f"{arm}_{role}.json", "w"), indent=1)
    # gradients need trainable parameters again
    del model, plumb
    model, spec, ck, saved = load_arm(arm)
    cfg = json.load(open(saved / "config.json")) if (saved / "config.json").exists() else None
    init = cfg.get("init_from") if cfg else None
    report["gradients"] = gradient_flow(arm, model, spec, ds, collate, init)
    report["seconds"] = time.time() - start
    json.dump(report, open(folder / f"{arm}_{role}.json", "w"), indent=1)
    log(f"gradients done {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
