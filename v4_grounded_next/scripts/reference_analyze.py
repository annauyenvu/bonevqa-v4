"""Does reference information (age, sex, site, view, region, hardware) improve the answers?  No model is loaded.

    source v4_grounded_next/env.sh
    $PY -m scripts.reference_analyze --out reference_fold0 --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa \
        J_v4cf_yolojoint_btxrd K_v4cf_kg K0_v4cf_kgempty

Inputs: runs/joint/fold_0/ref_<name>_<role>.json (scripts/reference_score.py), the BTXRD and FracAtlas TRAIN records
(for the priors), kg_rgo/out/v2/bone_kg_v2.json, and for FracAtlas runs/yolo_ablation/fold_0/<name>_{calibration,test}.json.

Three ways of giving the model the reference information, all without retraining:
  wording      the information is written into the question (age_sex; age_sex_site)
  data prior   naive Bayes over the metadata, fitted on TRAIN images only, fused with the model's output:
                 diagnosis   log p_model(c | image) + lam * sum_f log P(f | c)
                 yes / no    logit p_model + lam * (log-odds of the metadata prior - log-odds of the base rate)
  KG prior     diagnosis only: lam * (age band matches a typical_age edge + site matches a typical_site edge) of graph v2
lam is chosen on the validation role (BTXRD model_val; FracAtlas calibration) by log-loss and then applied to
test and external.  "metadata only" uses no image at all.  Paired bootstrap over pseudo-patient groups (BTXRD) or
components (FracAtlas), 2,000 resamples, differences against the plain model output.
For the tumor question the lesion site is NOT used: BTXRD gives a site only for tumour images (label leak).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from vg.joint import BT_ROLE
from vg.paths import ROOT, RUNS, dump_json, read_json

N_BOOT, SEED = 2000, 20261114
LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
KG_LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
MALIGNANT = {"osteosarcoma", "other malignant bone tumor"}
BAND = ((0, 9, "age_child"), (10, 19, "age_adolescent"), (20, 39, "age_young_adult"), (40, 59, "age_middle"), (60, 200, "age_older"))
FA_DATA = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3"


def band(age):
    return next(b for lo, hi, b in BAND if lo <= int(age) <= hi)


def feats(r, use):
    f = {}
    if "age" in use:
        f["age"] = band(r["age"])
    if "sex" in use:
        f["sex"] = r["gender"]
    if "region" in use:
        f["region"] = r["region"]
    if "view" in use:
        f["view"] = "+".join(sorted(r["views"]))
    if "site" in use:
        f["site"] = "+".join(sorted(list(r["bones"]) + list(r["joints"]))) or "none"
    if "hardware" in use:
        f["hardware"] = int(bool(r["hardware"]))
    return f


class NaiveBayes:
    """Categorical naive Bayes with add-one smoothing; unseen feature values share the smoothing mass."""

    def __init__(self, rows, labels, use):
        self.use, self.classes = use, sorted(set(labels))
        self.prior = Counter(labels)
        self.n = len(labels)
        self.count = defaultdict(Counter)
        self.values = defaultdict(set)
        for r, y in zip(rows, labels):
            for k, v in feats(r, use).items():
                self.count[(k, y)][v] += 1
                self.values[k].add(v)

    def loglik(self, r):   # sum_f log P(f | c), per class
        f = feats(r, self.use)
        return np.array([sum(math.log((self.count[(k, c)][v] + 1) / (self.prior[c] + len(self.values[k]) + 1)) for k, v in f.items())
                         for c in self.classes])

    def logprior(self):
        return np.array([math.log(self.prior[c] / self.n) for c in self.classes])


def kg_scores(candidates):
    g = read_json(ROOT / "kg_rgo/out/v2/bone_kg_v2.json")
    parent = {e["source"]: e["target"] for e in g["edges"] if e["relation"] == "part_of"}
    ages, sites = defaultdict(set), defaultdict(set)
    for e in g["edges"]:
        if e["relation"] == "typical_age":
            ages[e["source"]].add(e["target"])
        elif e["relation"] == "typical_site":
            x = e["target"]
            while x:
                sites[e["source"]].add(x)
                x = parent.get(x)
    node = {c: g["label_map"]["btxrd"][{"other benign bone tumor": "other bt", "other malignant bone tumor": "other mt"}.get(c, c)]
            for c in candidates}

    def score(r):
        b = band(r["age"])
        here = {x.replace(" ", "_").replace("-", "_") for x in list(r["bones"]) + list(r["joints"])}
        out = []
        for c in candidates:
            n = node[c]
            if n not in ages and n not in sites:      # category nodes carry no age / site edges: neutral
                out.append(1.0)
            else:
                out.append(float(b in ages[n]) + float(bool(here & sites[n])))
        return np.array(out)
    return score, node


def nll(logits, y):
    z = logits - logits.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def diag_metrics(pred, y, n_class):
    ok = pred == y
    rec = [ok[y == c].mean() for c in range(n_class) if (y == c).any()]
    return {"accuracy": float(ok.mean()), "balanced_accuracy": float(np.mean(rec))}


def bin_metrics(p, y):
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p))}


def boot(groups, fn_ref, fn_new, rng):
    """Paired group bootstrap of (new - ref) for every metric the functions return."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    gl = list(members.values())
    diffs = defaultdict(list)
    for _ in range(N_BOOT):
        idx = np.array([i for k in rng.integers(0, len(gl), len(gl)) for i in gl[k]])
        try:
            a, b = fn_ref(idx), fn_new(idx)
        except ValueError:
            continue
        for k in a:
            diffs[k].append(b[k] - a[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}


def analyse_btxrd(name, vqa_dir, folder, rng):
    train = read_json(Path(vqa_dir) / "vqa_train.json")
    per_image = {r["image_id"]: r for r in train if r.get("canonical") == "tumor"}
    dx_train = [(per_image[r["image_id"]], r["answer"]) for r in train if r.get("canonical") == "diagnosis"]
    tum_train = [r for r in per_image.values() if r["tumor"]]
    R = {role: read_json(folder / f"ref_{name}_{role}.json") for role in BT_ROLE if role != "train"}
    cands = R["test"]["candidates"]
    cidx = {c: i for i, c in enumerate(cands)}
    kg_score, kg_node = kg_scores(cands)
    out = {"candidates": cands, "kg_node": kg_node, "diagnosis": {}, "malignancy": {}, "tumor": {}}

    # ---------------------------------------------------------------- diagnosis
    nbs = {"age + sex": NaiveBayes([r for r, _ in dx_train], [y for _, y in dx_train], ("age", "sex")),
           "site": NaiveBayes([r for r, _ in dx_train], [y for _, y in dx_train], ("site",)),
           "age + sex + site + region + view": NaiveBayes([r for r, _ in dx_train], [y for _, y in dx_train],
                                                          ("age", "sex", "site", "region", "view"))}
    for nb in nbs.values():
        assert nb.classes == cands

    def pack(role):
        rows = R[role]["endpoints"]["diagnosis"]
        y = np.array([cidx[r["truth"]] for r in rows])
        return rows, y, [(r["center"], r["group"]) for r in rows]

    v_rows, v_y, _ = pack("model_val")
    methods = {}   # label -> function(rows) -> logits (N, C)
    for variant, lab in (("plain", "image only (plain question)"), ("age_sex", "wording: age, sex in the question"),
                         ("age_sex_site", "wording: age, sex, site in the question")):
        methods[lab] = lambda rows, v=variant: np.array([r[v] for r in rows])
    chosen = {}
    for lab, nb in nbs.items():
        ll = lambda rows, nb=nb: np.array([nb.loglik(r) for r in rows])
        base = np.array([r["plain"] for r in v_rows])
        lam = min(LAMS, key=lambda t: nll(base + t * ll(v_rows), v_y))
        chosen[f"image + data prior ({lab})"] = lam
        methods[f"image + data prior ({lab})"] = lambda rows, ll=ll, lam=lam: np.array([r["plain"] for r in rows]) + lam * ll(rows)
        methods[f"metadata only ({lab})"] = lambda rows, ll=ll, nb=nb: nb.logprior()[None] + ll(rows)
    ks = lambda rows: np.array([kg_score(r) for r in rows])
    lam = min(KG_LAMS, key=lambda t: nll(np.array([r["plain"] for r in v_rows]) + t * ks(v_rows), v_y))
    chosen["image + KG prior (age, site edges)"] = lam
    methods["image + KG prior (age, site edges)"] = lambda rows, lam=lam: np.array([r["plain"] for r in rows]) + lam * ks(rows)
    methods["KG prior only (age, site edges)"] = lambda rows: ks(rows)
    prior = nbs["site"].logprior()
    methods["majority class (no image, no metadata)"] = lambda rows: np.tile(prior, (len(rows), 1))
    is_mal = np.array([c in MALIGNANT for c in cands])
    for role in ("model_val", "test", "external"):
        rows, y, groups = pack(role)
        ref_pred = methods["image only (plain question)"](rows).argmax(1)
        table = {}
        for lab, fn in methods.items():
            pred = fn(rows).argmax(1)
            m = diag_metrics(pred, y, len(cands))
            m["benign_vs_malignant_accuracy"] = float((is_mal[pred] == is_mal[y]).mean())
            m["classes_predicted"] = int(len(set(pred.tolist())))
            m["recall_by_class"] = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(cands) if (y == i).any()}
            if lab != "image only (plain question)":
                m["ci95_vs_image_only"] = boot(groups, lambda ix: diag_metrics(ref_pred[ix], y[ix], len(cands)),
                                               lambda ix, pred=pred: diag_metrics(pred[ix], y[ix], len(cands)), rng)
                m["diff_vs_image_only"] = {k: m[k] - table["image only (plain question)"][k] for k in ("accuracy", "balanced_accuracy")}
            table[lab] = m
        out["diagnosis"][role] = {"n": len(y), "n_by_class": {c: int((y == i).sum()) for i, c in enumerate(cands)}, "methods": table}
    out["diagnosis"]["lambda_chosen_on_val"] = chosen

    # ---------------------------------------------------------------- yes / no endpoints
    for ep, train_rows, key, sets in (
            ("malignancy", tum_train, "malignant", {"age + sex": ("age", "sex"), "site": ("site",),
                                                    "age + sex + site + region + view": ("age", "sex", "site", "region", "view")}),
            ("tumor", list(per_image.values()), "tumor", {"age + sex": ("age", "sex"), "age + sex + region + view": ("age", "sex", "region", "view")})):
        ytr = [int(bool(r[key])) for r in train_rows]
        base_lo = math.log(sum(ytr) / (len(ytr) - sum(ytr)))
        nb2 = {lab: NaiveBayes(train_rows, ytr, use) for lab, use in sets.items()}
        lo = lambda rows, nb: np.array([(lambda v: v[1] - v[0])(nb.loglik(r)) for r in rows])   # metadata log-odds minus base rate
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        bce = lambda z, y: float(np.mean(np.log1p(np.exp(-np.where(y == 1, z, -z)))))
        vr = R["model_val"]["endpoints"][ep]
        vy = np.array([int(r["truth"] == "yes") for r in vr])
        vz = logit(np.array([r["plain"] for r in vr]))
        m2 = {"image only (plain question)": lambda rows: logit(np.array([r["plain"] for r in rows])),
              "wording: age, sex in the question": lambda rows: logit(np.array([r["age_sex"] for r in rows]))}
        if ep == "malignancy":
            m2["wording: age, sex, site in the question"] = lambda rows: logit(np.array([r["age_sex_site"] for r in rows]))
        chosen2 = {}
        for lab, nb in nb2.items():
            lam = min(LAMS, key=lambda t: bce(vz + t * lo(vr, nb), vy))
            chosen2[f"image + data prior ({lab})"] = lam
            m2[f"image + data prior ({lab})"] = lambda rows, nb=nb, lam=lam: logit(np.array([r["plain"] for r in rows])) + lam * lo(rows, nb)
            m2[f"metadata only ({lab})"] = lambda rows, nb=nb: base_lo + lo(rows, nb)
        m2["proposal flag only"] = lambda rows: np.array([float(r["has_proposal"]) for r in rows])
        for role in ("model_val", "test", "external"):
            rows = R[role]["endpoints"][ep]
            y = np.array([int(r["truth"] == "yes") for r in rows])
            groups = [(r["center"], r["group"]) for r in rows]
            ref = m2["image only (plain question)"](rows)
            table = {}
            for lab, fn in m2.items():
                z = fn(rows)
                m = bin_metrics(z, y)
                if lab != "image only (plain question)":
                    m["ci95_vs_image_only"] = boot(groups, lambda ix: bin_metrics(ref[ix], y[ix]), lambda ix, z=z: bin_metrics(z[ix], y[ix]), rng)
                    m["diff_vs_image_only"] = {k: m[k] - table["image only (plain question)"][k] for k in ("auprc", "auroc")}
                table[lab] = m
            out[ep][role] = {"n": len(y), "positives": int(y.sum()), "methods": table}
        out[ep]["lambda_chosen_on_val"] = chosen2
    return out


def analyse_fracatlas(name, rng):
    """Fracture endpoint: P(fracture) fused with a prior from region, views and (separately) hardware."""
    abl = RUNS / "yolo_ablation" / "fold_0"
    files = {role: abl / f"{name}_{role}.json" for role in ("calibration", "test")}
    if not all(f.exists() for f in files.values()):
        return {"skipped": "no calibration / test predictions for this run"}
    meta = {}
    for role in ("train", "calibration", "test"):
        for r in read_json(FA_DATA / f"fold_0/vqa_{role}.json"):
            meta[r["image_id"]] = {"region": r["region"], "views": r["views"], "hardware": r["hardware"], "fractured": r["fractured"],
                                   "role": role, "age": 0, "gender": "", "bones": [], "joints": []}
    comp = {}
    for gid, images in enumerate(read_json(FA_DATA / "manifest.json")["components"]):
        for im in images:
            comp[str(im).rsplit(".", 1)[0]] = gid
    train = [m for m in meta.values() if m["role"] == "train"]
    ytr = [int(bool(m["fractured"])) for m in train]
    base_lo = math.log(sum(ytr) / (len(ytr) - sum(ytr)))
    sets = {"region + view": ("region", "view"), "region + view + hardware": ("region", "view", "hardware"), "hardware": ("hardware",)}
    nbs = {lab: NaiveBayes(train, ytr, use) for lab, use in sets.items()}
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    bce = lambda z, y: float(np.mean(np.log1p(np.exp(-np.where(y == 1, z, -z)))))
    lo = lambda ids, nb: np.array([(lambda v: v[1] - v[0])(nb.loglik(meta[i])) for i in ids])
    rows = {role: read_json(f)["rows"] for role, f in files.items()}
    cal_ids = [r["image_id"] for r in rows["calibration"]]
    cz, cy = logit(np.array([r["p_final"] for r in rows["calibration"]])), np.array([r["y"] for r in rows["calibration"]])
    ids = [r["image_id"] for r in rows["test"]]
    z0, y = logit(np.array([r["p_final"] for r in rows["test"]])), np.array([r["y"] for r in rows["test"]])
    groups = [comp.get(i, i) for i in ids]
    table = {"image only": bin_metrics(z0, y)}
    chosen = {}
    for lab, nb in nbs.items():
        lam = min(LAMS, key=lambda t: bce(cz + t * lo(cal_ids, nb), cy))
        chosen[lab] = lam
        for label, z in ((f"image + data prior ({lab})", z0 + lam * lo(ids, nb)), (f"metadata only ({lab})", base_lo + lo(ids, nb))):
            m = bin_metrics(z, y)
            m["ci95_vs_image_only"] = boot(groups, lambda ix: bin_metrics(z0[ix], y[ix]), lambda ix, z=z: bin_metrics(z[ix], y[ix]), rng)
            m["diff_vs_image_only"] = {k: m[k] - table["image only"][k] for k in ("auprc", "auroc")}
            table[label] = m
    return {"n": len(y), "positives": int(y.sum()), "lambda_chosen_on_calibration": chosen, "methods": table}


def fmt(x, signed=False):
    return (f"{x:+.3f}" if signed else f"{x:.3f}").replace(".", ",").replace("-", "−")


def markdown(res):
    L = ["# Thông tin tham chiếu có giúp không? (fold 0)", "",
         "Chênh lệch so với đầu ra gốc của chính mô hình đó (chỉ ảnh, câu hỏi gốc), khoảng tin cậy 95% trong ngoặc vuông.", ""]
    for name, r in res["runs"].items():
        L += [f"## {name}", ""]
        b = r["btxrd"]
        for role in ("test", "external"):
            d = b["diagnosis"][role]
            L += [f"### Chẩn đoán loại u — BTXRD {role} (n = {d['n']})", "",
                  "| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |", "|---|---|---|---|---|---|"]
            for lab, m in d["methods"].items():
                ci = m.get("ci95_vs_image_only")
                diff = f"{fmt(m['diff_vs_image_only']['accuracy'], True)} [{fmt(ci['accuracy'][0], True)}; {fmt(ci['accuracy'][1], True)}]" if ci else "—"
                L.append(f"| {lab} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['benign_vs_malignant_accuracy'])} | {m['classes_predicted']} | {diff} |")
            L.append("")
        for ep, title in (("malignancy", "Lành hay ác"), ("tumor", "Có u không")):
            for role in ("test", "external"):
                d = b[ep][role]
                L += [f"### {title} — BTXRD {role} (n = {d['n']}, dương {d['positives']})", "",
                      "| Cách | AUPRC | AUROC | Chênh AUROC |", "|---|---|---|---|"]
                for lab, m in d["methods"].items():
                    ci = m.get("ci95_vs_image_only")
                    diff = f"{fmt(m['diff_vs_image_only']['auroc'], True)} [{fmt(ci['auroc'][0], True)}; {fmt(ci['auroc'][1], True)}]" if ci else "—"
                    L.append(f"| {lab} | {fmt(m['auprc'])} | {fmt(m['auroc'])} | {diff} |")
                L.append("")
        f = r["fracatlas"]
        if "methods" in f:
            L += [f"### Gãy xương — FracAtlas test (n = {f['n']}, gãy {f['positives']})", "", "| Cách | AUPRC | AUROC | Chênh AUPRC |", "|---|---|---|---|"]
            for lab, m in f["methods"].items():
                ci = m.get("ci95_vs_image_only")
                diff = f"{fmt(m['diff_vs_image_only']['auprc'], True)} [{fmt(ci['auprc'][0], True)}; {fmt(ci['auprc'][1], True)}]" if ci else "—"
                L.append(f"| {lab} | {fmt(m['auprc'])} | {fmt(m['auroc'])} | {diff} |")
            L.append("")
    return "\n".join(L)


def main(a):
    folder = Path(a.dir) if a.dir else RUNS / "joint" / f"fold_{a.fold}"
    out = folder / f"{a.out}.json"
    assert not out.exists(), out
    rng = np.random.default_rng(SEED)
    res = {"n_boot": N_BOOT, "seed": SEED, "runs": {}}
    for name in a.names:
        res["runs"][name] = {"btxrd": analyse_btxrd(name, a.btxrd_vqa, folder, rng), "fracatlas": analyse_fracatlas(name, rng)}
    dump_json(out, res, exclusive=True)
    md = folder / f"{a.out}.md"
    md.write_text(markdown(res))
    print(md.read_text()[: a.print_chars], flush=True)
    print(json.dumps({"written": [str(out), str(md)]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--btxrd-vqa", required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--print-chars", type=int, default=6000)
    main(ap.parse_args())
