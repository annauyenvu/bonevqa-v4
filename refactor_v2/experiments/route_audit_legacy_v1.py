"""Route interventions on v1/v2/v3 checkpoints with the code each generation ran on.  CPU, read-only.
    python route_audit_legacy_v1.py <name> <checkpoint> <code: exp4|src> [boxes.json]
Cohort: component-clean test (route sensitivity; v1 exp4 runs saw the leaky split, so their AP is not a clean estimate)."""
import json, random, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
OUT = ROOT / "refactor_v2/reports/route_audit_v1/legacy"
OUT.mkdir(parents=True, exist_ok=True)
name, ck, code = sys.argv[1:4]
boxes_file = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / "refactor_v2/detector_clean/boxes.json")
sys.path[:0] = [str(ROOT / "exp4_multiseed/code") if code == "exp4" else str(ROOT / "refactor_v2"), str(ROOT / ".evaluation_runtime/deps")]
from src.models.bonevqa import BoneVQAModel
ImageFile.LOAD_TRUNCATED_IMAGES = True  # src/data/common.py resets it on import; this cohort is route sensitivity only

torch.set_num_threads(3)
DATA = ROOT / ".evaluation_runtime/data"
recs = json.load(open(DATA / "fracatlas_component_clean/vqa_test.json"))
CAN = "Is there a fracture in this X-ray?"
idx = list(range(len(recs)))
random.Random(20261101).shuffle(idx)
keep = [i for i in idx if recs[i]["question"] == CAN][:300]
keep += [i for i in idx if recs[i]["answer_type"] == "CLOSED" and recs[i]["question"] != CAN][:250] + [i for i in idx if recs[i]["answer_type"] == "OPEN"][:100]
random.Random(1).shuffle(keep)
boxes = json.load(open(boxes_file))
boxes = boxes.get("boxes", boxes) if isinstance(boxes, dict) else boxes
try:
    model = BoneVQAModel.load(ck, device="cpu", cfg_override={"sam": {"enabled": False}})
except Exception as e:
    print("load via class failed", repr(e)[:300]); raise
model.eval()
c = model.cfg
yes, no = model.answer_to_idx["yes"], model.answer_to_idx["no"]
memo = {}
vf = model.visual.forward


def visual(pixel):
    keys = [hash(x.numpy().tobytes()) for x in pixel]
    first = {}
    for i, k in enumerate(keys):
        if k not in memo:
            first.setdefault(k, i)
    if first:
        ids = list(first.values())
        o = vf(pixel[ids])
        for j, i in enumerate(ids):
            memo[keys[i]] = (o["tokens"][j], o["embed"][j])
    return {"tokens": torch.stack([memo[k][0] for k in keys]), "embed": torch.stack([memo[k][1] for k in keys])}


model.visual.forward = visual


def rect(bs, im):
    out = []
    for b in bs[:3]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b[:4]]
        x1, x2, y1, y2 = max(0, x1), min(im.width, x2), max(0, y1), min(im.height, y2)
        if x2 > x1 and y2 > y1:
            m = np.zeros((im.height, im.width), np.uint8)
            m[y1:y2, x1:x2] = 1
            out.append(m)
    return np.stack(out) if out else np.zeros((0, im.height, im.width), np.uint8)


def shift(bs, im):
    out = []
    for b in bs:
        w, h = b[2] - b[0], b[3] - b[1]
        x, y = (b[0] + im.width / 2) % max(im.width - w, 1), (b[1] + im.height / 2) % max(im.height - h, 1)
        out.append([x, y, x + w, y + h])
    return out


def run(batch, a):
    saved = {k: c.get(k) for k in a if k in c or k.startswith(("use_", "visual_", "lens_", "latent_"))}
    gam = {}
    for k, v in a.items():
        if k.startswith("use_") or k.endswith("force_zero"):
            c[k] = v
        if k.startswith("gamma"):
            p = getattr(model.prompt_mixer if k != "gamma_latent" else model.fusion, k)
            gam[k] = float(p)
            p.data.mul_(v)
    ims, qs, ids, bx = batch
    n = len(ims)
    d = [next((j % n for j in range(i + 1, i + n) if ids[j % n] != ids[i]), i) for i in range(n)]
    if a.get("image") == "blank":
        ims = [Image.new("RGB", im.size, (128, 128, 128)) for im in ims]
    if a.get("image") == "swap":
        ims, bx = [ims[j] for j in d], [bx[j] for j in d]
    bs = bx
    if a.get("box") == "none":
        bs = [[] for _ in bx]
    if a.get("box") == "shift":
        bs = [shift(b, im) for b, im in zip(bx, ims)]
    masks = [rect(b, im) for b, im in zip(bs, ims)]
    core = model._encode_core(ims, qs, [None] * n, masks, c["prompt_kind"], False)
    o = dict(closed=core["logits_closed"].float(), xf=core["fus"]["x_f"].float(), prefix=core["vis"]["prefix_vis"].float(), f_i=core["vis"]["f_i"].float(), q=core["q_texts"], core=core)
    o["frac"] = core["fracture_logit"].float() if core.get("fracture_logit") is not None else None
    for k, v in saved.items():
        c[k] = v
    for k, v in gam.items():
        p = getattr(model.prompt_mixer if k != "gamma_latent" else model.fusion, k)
        p.data.fill_(v)
    return o


def llm_first(o, opens):
    if not opens:
        return None
    sub = model._subset_core(o["core"], opens)
    e, att, _ = model._build_llm_inputs(model._llm_prefix(sub), [o["q"][i] for i in opens], None)
    return model.llm(inputs_embeds=e, attention_mask=att, position_ids=(att.cumsum(1) - 1).clamp_min(0)).logits[:, -1].float().log_softmax(-1)


plan = {"img_blank": {"image": "blank"}, "img_swap": {"image": "swap"}}
if c["use_visual_prompt"]:
    plan.update(V_off={"use_visual_prompt": False}, box_none={"box": "none"}, box_shift={"box": "shift"})
if c["use_lens"]:
    plan.update(L_off={"use_lens": False}, box_none={"box": "none"}, box_shift={"box": "shift"})
if c["use_latent_prompt"]:
    plan.update(P_off={"use_latent_prompt": False})
if c["use_textual_prompt"]:
    plan.update(T_off={"use_textual_prompt": False})
state = model.state_dict()
if "prompt_mixer.gamma_visual" in state and c["use_visual_prompt"]:
    plan.update(V_gate_x0={"gamma_visual": 0.0}, V_gate_x100={"gamma_visual": 100.0})
    plan.pop("V_off", None)
if "prompt_mixer.gamma_lens" in state and c["use_lens"]:
    plan.update(L_gate_x0={"gamma_lens": 0.0}, L_gate_x100={"gamma_lens": 100.0})
    plan.pop("L_off", None)
if "fusion.gamma_latent" in state and c["use_latent_prompt"]:
    plan.update(P_gate_x0={"gamma_latent": 0.0}, P_gate_x100={"gamma_latent": 100.0})
    plan.pop("P_off", None)
res = {k: dict(logit=np.full(len(recs), np.nan), top=np.full(len(recs), -1), xf=[], pre=[], fi=[], kl=[], t1=[]) for k in ["clean"] + list(plan)}
base_t1 = {}
t0 = time.time()
with torch.no_grad():
    for off in range(0, len(keep), 16):
        ids = keep[off: off + 16]
        ims = [ImageOps.exif_transpose(Image.open(DATA / "fracatlas/FracAtlas" / recs[i]["image"]).convert("RGB")) if (DATA / "fracatlas/FracAtlas" / recs[i]["image"]).exists() else None for i in ids]
        assert all(x is not None for x in ims)
        qs = [recs[i]["question"] for i in ids]
        iid = [recs[i]["image_id"] for i in ids]
        bx = [[b[:4] for b in (boxes.get(x) or [])] for x in iid]
        opens = [k for k, i in enumerate(ids) if recs[i]["answer_type"] == "OPEN"]
        memo.clear()
        clean = run((ims, qs, iid, bx), {})
        l0 = llm_first(clean, opens)
        for nm, a in [("clean", {})] + list(plan.items()):
            o = clean if nm == "clean" else run((ims, qs, iid, bx), a)
            l = l0 if nm == "clean" else llm_first(o, opens)
            r = res[nm]
            for k, i in enumerate(ids):
                y = o["closed"][k, [yes, no]].softmax(0)
                r["logit"][i] = float(o["frac"][k]) if o["frac"] is not None else float(torch.logit(y[0].clamp(1e-6, 1 - 1e-6)))
                r["top"][i] = int(o["closed"][k].argmax())
            r["xf"].append(float(((o["xf"] - clean["xf"]).norm(dim=-1) / clean["xf"].norm(dim=-1)).mean()))
            r["pre"].append(float(((o["prefix"] - clean["prefix"]).norm(dim=(1, 2)) / clean["prefix"].norm(dim=(1, 2))).mean()))
            if o["f_i"].shape == clean["f_i"].shape:
                r["fi"].append(float(((o["f_i"] - clean["f_i"]).norm(dim=(1, 2)) / clean["f_i"].norm(dim=(1, 2))).mean()))
            if l is not None:
                r["kl"].append(float((l0.exp() * (l0 - l)).sum(-1).mean()))
                r["t1"].append(float((l.argmax(-1) != l0.argmax(-1)).float().mean()))
        if off % 160 == 0:
            print(name, off, len(keep), round(time.time() - t0), flush=True)


def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]; tp = np.cumsum(y)
    return float((tp[y == 1] / (np.nonzero(y == 1)[0] + 1)).mean())


can = [i for i in keep if recs[i]["question"] == CAN]
clo = [i for i in keep if recs[i]["answer_type"] == "CLOSED" and recs[i]["question"] != CAN]
y = np.array([recs[i]["answer"].strip().lower() == "yes" for i in can], int)
tab = {}
p0 = res["clean"]["logit"][can]
for nm, r in res.items():
    p = r["logit"][can]
    tab[nm] = dict(ap=ap(y, p), frac_flip=float(((p >= 0) != (p0 >= 0)).mean()), mean_abs_dlogit=float(np.abs(p - p0).mean()), closed_flip=float(np.mean(r["top"][clo] != res["clean"]["top"][clo])),
                   xf_shift=float(np.mean(r["xf"])), prefix_shift=float(np.mean(r["pre"])), fi_shift=float(np.mean(r["fi"])) if r["fi"] else None, open_kl=float(np.mean(r["kl"])) if r["kl"] else None, open_t1_flip=float(np.mean(r["t1"])) if r["t1"] else None,
                   closed_acc=float(np.mean([model.answer_vocab[int(r["top"][i])].strip().lower() == recs[i]["answer"].strip().lower() for i in clo])))
json.dump(dict(name=name, checkpoint=ck, code=code, switches={k: c.get(k) for k in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "lens_training_only", "prompt_kind", "visual_route")}, gates={k: float(v) for k, v in state.items() if "gamma" in k}, table=tab, n_canonical=len(can), positives=int(y.sum())),
          open(OUT / f"{name}.json", "w"), indent=1)
for k, v in tab.items():
    print(k, {a: (round(b, 4) if b is not None else None) for a, b in v.items()})
