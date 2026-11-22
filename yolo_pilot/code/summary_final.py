"""Summary tables of the thesis final configuration (scope freeze, 20/11/2026).  Reads stored JSON only, except
compare_arms for JF vs B (paired component bootstrap, ~20 s per fold, CPU) when that file is missing.

    python3 -I code/summary_final.py          # -> runs/summary_final/{summary.json, bang_tong_hop.md}

Table 1  FracAtlas test, folds 0-3: A, B, D2, J, JF (AUPRC, AUROC; JF vs A, B, J with 95% CI) and the same without
         the orthopaedic-hardware images (98% of them are labelled fractured: a shortcut).
Table 2  BTXRD test / external, folds 0-3: D2, J, JF (tumour AUROC, malignancy AUROC, diagnosis accuracy; JF vs J).
Table 3  Trust layer (calibrated confidence + detector score, 80% target coverage, fitted on val): J and JF.
Table 4  Fold 0 extras: J seeds 42/43/44, KG control (K, K0), lesion-site information, grounding (placebo / oracle),
         BTXRD threshold selection (inference only).
Missing files print as "—" (the table can be run while folds are still training).
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
VG = ROOT / "v4_grounded_next"
AB = VG / "runs/yolo_ablation"
JT = VG / "runs/joint"
PILOT = ROOT / "yolo_pilot"
OUT = PILOT / "runs/summary_final"
FOLDS = [0, 1, 2, 3, 4]
JF = "JF_v4cf_yolojoint_btxrd_r0.02"
ARMS = {"A": "A_v4_sealed_frcnn", "B": "B_v4cf_frcnn", "D2": "D2_v4cf_yolojoint", "J": "J_v4cf_yolojoint_btxrd", "JF": JF}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def fmt_d(d):
    if not d:
        return "—"
    lo, hi = d["ci95"]
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{d['diff']:+.3f} [{lo:+.3f}; {hi:+.3f}]{star}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); y, s = y[o], s[o]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last]; fp = (last + 1) - tp
    prec = tp / (tp + fp); rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def auroc(y, s):
    from math import isnan  # noqa: F401
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s))
    ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def rows(fold, name):
    d = load(AB / f"fold_{fold}/{name}_test.json")
    return None if d is None else {r["image_id"]: (r["y"], r["p_final"]) for r in d["rows"]}


def v4cmd(*args):
    cmd = f"source {VG}/env.sh > /dev/null; cd {VG}; $PY -m scripts.compare_arms " + " ".join(args)
    subprocess.run(["zsh", "-c", cmd], capture_output=True, text=True)


# ------------------------------------------------------------------------------------------------ table 1
def table1():
    out = {}
    for f in FOLDS:
        D = AB / f"fold_{f}"
        if (D / f"{JF}_test.json").exists() and not (D / "compare_test_JF_vsB.json").exists():
            v4cmd("--fold", str(f), "--role", "test", "--out", "compare_test_JF_vsB", ARMS["B"], JF)
        cf = load(D / "compare_test_final.json") or {}
        o = {"arms": {}, "JF_vs_A": None, "JF_vs_B": None, "JF_vs_J": None, "no_hardware": {}}
        for k, n in ARMS.items():
            m = (cf.get("arms") or {}).get(n) or (load(D / f"{n}_test.json") or {}).get("metrics")
            o["arms"][k] = None if m is None else {x: m.get(x) for x in ("auprc", "auroc", "sens@0.5", "spec@0.5",
                                                                         "m_rate_fractured", "m_rate_nonfractured")}
        o["JF_vs_A"] = ((cf.get("diff_vs_reference") or {}).get(JF) or {}).get("auprc")
        for key, fn in (("JF_vs_B", "compare_test_JF_vsB.json"), ("JF_vs_J", "compare_test_final_vsJ.json")):
            c = load(D / fn)
            o[key] = None if c is None else c["diff_vs_reference"][JF]["auprc"]
            if c is not None:
                o[key + "_auroc"] = c["diff_vs_reference"][JF]["auroc"]
        gt = load(PILOT / f"datasets/fa_fold{f}/gt.json")
        keep = {i for i, g in gt.items() if g["role"] == "test" and not g.get("hardware")}
        o["n_hardware_test"] = sum(1 for g in gt.values() if g["role"] == "test" and g.get("hardware"))
        for k, n in ARMS.items():
            r = rows(f, n)
            if r is None:
                o["no_hardware"][k] = None; continue
            ids = sorted(i for i in r if i in keep)
            y = np.array([r[i][0] for i in ids]); s = np.array([r[i][1] for i in ids], float)
            o["no_hardware"][k] = {"auprc": ap(y, s), "auroc": auroc(y, s), "n": len(ids)}
        # JF - B without hardware, image-level paired bootstrap
        rb, rj = rows(f, ARMS["B"]), rows(f, JF)
        if rb and rj:
            ids = sorted(i for i in rb if i in keep and i in rj)
            y = np.array([rb[i][0] for i in ids]); sb = np.array([rb[i][1] for i in ids], float)
            sj = np.array([rj[i][1] for i in ids], float)
            rng = np.random.default_rng(20261120); d = []
            for _ in range(1000):
                b = rng.integers(0, len(ids), len(ids))
                if y[b].sum() == 0:
                    continue
                d.append(ap(y[b], sj[b]) - ap(y[b], sb[b]))
            o["no_hardware"]["JF_vs_B"] = {"diff": ap(y, sj) - ap(y, sb), "ci95": [float(np.percentile(d, 2.5)),
                                                                                  float(np.percentile(d, 97.5))]}
        out[f] = o
    return out


# ------------------------------------------------------------------------------------------------ table 2
def table2():
    out = {}
    for f in FOLDS:
        o = {}
        for role in ("test", "external"):
            o[role] = {}
            for k in ("D2", "J", "JF"):
                d = load(JT / f"fold_{f}/{ARMS[k]}_btxrd_{role}.json")
                if d is None:
                    o[role][k] = None; continue
                m = d["metrics"]
                o[role][k] = {"tumor_auroc": m["tumor"]["auroc"], "tumor_auprc": m["tumor"]["auprc"],
                              "malignancy_auroc": m["malignancy"]["auroc"], "diagnosis_accuracy": m["diagnosis"]["accuracy"],
                              "diagnosis_macro_f1": m["diagnosis"].get("macro_f1"),
                              "tumor_proposal_rate_pos": m["tumor"].get("m_rate_pos"),
                              "tumor_proposal_rate_neg": m["tumor"].get("m_rate_neg")}
            c = load(JT / f"fold_{f}/compare_btxrd_{role}_JF_vs_J.json")
            if c:
                e = c["endpoints"]
                o[role]["JF_vs_J"] = {"tumor_auroc": e["tumor"]["diff_vs_reference"][JF]["auroc"],
                                      "malignancy_auroc": e["malignancy"]["diff_vs_reference"][JF]["auroc"],
                                      "diagnosis_accuracy": e["diagnosis"]["diff_vs_reference"][JF]["accuracy"]}
        out[f] = o
    return out


# ------------------------------------------------------------------------------------------------ table 3
def table3():
    out = {}
    for f in FOLDS:
        o = {}
        for k in ("J", "JF"):
            t = load(JT / f"fold_{f}/trust_{ARMS[k]}_eval.json")
            if t is None:
                o[k] = None; continue
            r = {}
            for ep, roles in t["endpoints"].items():
                for role, v in roles.items():
                    sig = v.get("signals", {})
                    pt = (sig.get("head_detector") or sig.get("head") or v).get("0.80")
                    if pt:
                        r[f"{ep}/{role}"] = {"coverage": pt["coverage"], "acc_all": pt["accuracy_all"],
                                             "acc_answered": pt["accuracy_answered"], "gain_ci95": pt["gain_ci95"],
                                             "signal": "head+detector" if "head_detector" in sig else "head"}
            o[k] = r
        out[f] = o
    return out


# ------------------------------------------------------------------------------------------------ table 4 (fold 0)
def table4():
    o = {}
    s = load(AB / "fold_0/compare_test_J_seeds.json")
    seeds = {}
    for n, lab in (("J_v4cf_yolojoint_btxrd", "42"), ("J_s43_v4cf_yolojoint_btxrd", "43"), ("J_s44_v4cf_yolojoint_btxrd", "44")):
        e = {"fracatlas_auprc": s["arms"][n]["auprc"] if s else None, "fracatlas_auroc": s["arms"][n]["auroc"] if s else None}
        for role in ("test", "external"):
            d = load(JT / f"fold_0/{n}_btxrd_{role}.json")
            if d:
                m = d["metrics"]
                e[f"{role}_tumor_auroc"] = m["tumor"]["auroc"]; e[f"{role}_malignancy_auroc"] = m["malignancy"]["auroc"]
                e[f"{role}_diagnosis_accuracy"] = m["diagnosis"]["accuracy"]
        seeds[lab] = e
    o["seeds"] = seeds
    kg = {}
    ck = load(AB / "fold_0/compare_test_K_vs_J.json")
    for n, lab in (("J_v4cf_yolojoint_btxrd", "J"), ("K_v4cf_kg", "K (KG prompt)"), ("K0_v4cf_kgempty", "K0 (KG rỗng)")):
        fa = (ck or {}).get("arms", {}).get(n) or (load(AB / f"fold_0/{n}_test.json") or {}).get("metrics", {})
        e = {"fracatlas_auprc": fa.get("auprc")}
        for role in ("test", "external"):
            d = load(JT / f"fold_0/{n}_btxrd_{role}.json")
            if d:
                m = d["metrics"]
                e[f"{role}_tumor_auroc"] = m["tumor"]["auroc"]; e[f"{role}_malignancy_auroc"] = m["malignancy"]["auroc"]
                e[f"{role}_diagnosis_accuracy"] = m["diagnosis"]["accuracy"]
        kg[lab] = e
    for role in ("test", "external"):
        c = load(JT / f"fold_0/compare_btxrd_{role}_K_vs_J.json")
        if c:
            kg[f"K_vs_J_{role}"] = {ep: c["endpoints"][ep]["diff_vs_reference"]["K_v4cf_kg"].get(
                "auroc", c["endpoints"][ep]["diff_vs_reference"]["K_v4cf_kg"].get("accuracy")) for ep in c["endpoints"]}
    o["kg"] = kg
    ref = load(JT / "fold_0/reference_fold0_1117_0703.json")
    site = {}
    if ref:
        dg = ref["runs"]["J_v4cf_yolojoint_btxrd"]["btxrd"]["diagnosis"]
        for role in ("test", "external"):
            for meth in ("image only (plain question)", "image + data prior (site)", "metadata only (site)",
                         "image + data prior (age + sex + site + region + view)", "image + KG prior (age, site edges)",
                         "wording: age, sex, site in the question", "majority class (no image, no metadata)"):
                v = dg[role]["methods"][meth]
                site.setdefault(meth, {})[role] = {"accuracy": v["accuracy"], "diff": v.get("diff_vs_image_only", {}).get("accuracy"),
                                                   "ci95": v.get("ci95_vs_image_only", {}).get("accuracy")}
    o["lesion_site"] = site
    g = load(PILOT / "runs/sweep_analysis/fold0.json")
    gr = {}
    if g:
        for lvl, v in g["levels"].items():
            iv = v.get("intervention_vs_native", {})
            pl = next((x for k, x in iv.items() if k.endswith("_placebo")), None)
            orc = next((x for k, x in iv.items() if k.endswith("_oracleloc")), None)
            gt_ = v.get("grounding_test", {})
            gr[lvl] = {"v4_test_auprc": v["v4_test"]["auprc"], "v4_model_val_auprc": v["v4_model_val"]["auprc"],
                       "box_precision@0.5": v["detector_test"]["box_precision@0.5"],
                       "image_hit_rate@0.3": v["detector_test"]["image_hit_rate@0.3"],
                       "normal_with_box": v["detector_test"]["m_rate_nonfractured"],
                       "fractured_without_box": gt_.get("share_none"), "sens_without_box": gt_.get("sens@0.5_none"),
                       "tp_with_box_on_fracture": gt_.get("tp_with_box_on_fracture"),
                       "placebo_auprc": pl["auprc"] if pl else None, "oracle_auprc": orc["auprc"] if orc else None}
        gr["yolo_best_f1@0.5_model_val"] = {k: g["detector_best_f1@0.5_model_val"].get(k) for k in
                                            ("threshold", "m_rate_nonfractured", "f1@0.5")}
    o["grounding"] = gr
    o["btxrd_level_selection"] = load(JT / "final_btxrd_level.json")
    d1 = load(AB / "fold_1/compare_test_sweep0.02_vsD2.json")
    o["d2_level_0.02"] = {"fold0_vs_D2": (load(AB / "fold_0/compare_test_sweep_vsD2.json") or {}).get("diff_vs_reference", {}).get("D2_sweep_r0.02", {}).get("auprc"),
                          "fold1_vs_D2": d1["diff_vs_reference"]["D2_sweep_r0.02"]["auprc"] if d1 else None}
    return o



# ------------------------------------------------------------------------------------------------ table 0 (detector)
def table0():
    """YOLOv8 on its own (no V4): outer joint detector of each fold, at the detector's own best operating point
    (max box F1@0.5 on model_val, per class) and at the proposal threshold fed to V4.  Read-only."""
    import sys
    sys.path.insert(0, str(PILOT / "code"))
    from sweep_analysis import box_stats
    sel = load(JT / "final_btxrd_level.json")
    bl = sel["chosen_level"] if sel else "0.02"
    raw = load(PILOT / "datasets/btxrd_pilot/gt.json")
    gbt = {i: {"fractured": bool(g["tumor"]), "boxes": [b[:4] for b in g["boxes"]], "role": g["role"]} for i, g in raw.items()}
    keys = ("threshold", "box_precision@0.5", "box_recall@0.5", "f1@0.5", "image_hit_rate@0.3", "m_rate_fractured",
            "m_rate_nonfractured", "false_boxes_per_normal_image")
    out = {}
    for f in FOLDS:
        pre = "" if f == 0 else f"f{f}_"
        o = {}
        for ds, cls, gt, roles, pfile in (
                ("fracatlas", 0, load(PILOT / f"datasets/fa_fold{f}/gt.json"), ("test",),
                 VG / f"runs/crossfit_yolo_joint_sweep/r0.02/fold_{f}/proposals_fold{f}.json"),
                ("btxrd", 1, gbt, ("test", "external"),
                 VG / f"runs/crossfit_yolo_joint_sweep/r{bl}/btxrd_fold{f}/proposals_btxrd_insample.json")):
            pred = load(PILOT / f"runs/pred/{pre}fa_btxrd_cls{cls}.json")["boxes"]
            ids = {r: sorted(i for i, g in gt.items() if g["role"] == r) for r in ("val",) + roles}
            sc = sorted({b[4] for i in ids["val"] for b in pred.get(i, [])})
            grid = sorted(set(np.quantile(sc, np.linspace(0, 0.999, 300)).tolist()))
            best = max((box_stats(pred, gt, ids["val"], t) for t in grid), key=lambda r: r["f1@0.5"])
            pt = load(pfile)
            t_v4 = pt["thresholds"]["outer"] if pt else None
            e = {"own_best_threshold": best["threshold"], "own_best_val": {k: best[k] for k in keys}, "v4_threshold": t_v4}
            for r in roles:
                e[f"own_best_{r}"] = {k: v for k, v in box_stats(pred, gt, ids[r], best["threshold"]).items() if k in keys}
                if t_v4 is not None:
                    e[f"v4_level_{r}"] = {k: v for k, v in box_stats(pred, gt, ids[r], t_v4).items() if k in keys}
            o[ds] = e
        out[f] = o
    out["btxrd_v4_level"] = bl
    return out


def detector_md(t0):
    L = ["## 0. Detector YOLOv8 riêng (không qua V4)", "",
         "Detector chính (joint FracAtlas + BTXRD, YOLOv8s) của mỗi fold, **không huấn luyện lại, không bị thay đổi** bởi "
         "bất kỳ nhánh VQA nào. Điểm vận hành riêng của detector = ngưỡng có F1 hộp @IoU 0,5 cao nhất trên model_val "
         "(chọn theo từng fold, từng lớp). Đây là ngưỡng dùng để **hiển thị hộp** cho người đọc phim. Cột \"mức đưa vào V4\" "
         "là ngưỡng proposal của JF (FracAtlas 0,02; BTXRD mức chọn ở bảng 4e), chỉ để V4 dùng.", "",
         "| Fold | Tập | Ngưỡng | Precision hộp @0,5 | Recall hộp @0,5 | F1 @0,5 | Ảnh bệnh có hộp trúng (@0,3) | Ảnh bình thường có hộp |",
         "|---|---|---|---|---|---|---|---|"]
    for ds, lab, roles in (("fracatlas", "FracAtlas gãy", ("test",)), ("btxrd", "BTXRD u", ("test", "external"))):
        for r in roles:
            for kind, klab in (("own_best", "riêng của detector"), ("v4_level", "mức đưa vào V4")):
                vals = []
                for f in FOLDS:
                    e = t0[f][ds].get(f"{kind}_{r}")
                    if not e:
                        continue
                    vals.append(e)
                    L.append(f"| {f} | {lab} {r}, {klab} | {fmt(e['threshold'])} | {fmt(e['box_precision@0.5'])} | "
                             f"{fmt(e['box_recall@0.5'])} | {fmt(e['f1@0.5'])} | {fmt(e['image_hit_rate@0.3'])} | {fmt(e['m_rate_nonfractured'])} |")
                if vals:
                    m = lambda k: fmt(mean([v[k] for v in vals]))  # noqa: E731
                    L.append(f"| **TB** | {lab} {r}, {klab} | | {m('box_precision@0.5')} | {m('box_recall@0.5')} | {m('f1@0.5')} | "
                             f"{m('image_hit_rate@0.3')} | {m('m_rate_nonfractured')} |")
    L += ["", "Nguyên tắc: ngưỡng đưa vào V4 chỉ quyết định V4 nhìn thấy hộp nào; nó không thay đổi detector, và hộp hiển thị "
          "cho người dùng dùng điểm vận hành riêng của detector. Không tối ưu một nhánh VQA trung gian bằng cách làm giảm hiệu năng detector.", ""]
    return "\n".join(L) + "\n"

# ------------------------------------------------------------------------------------------------ markdown
def markdown(t1, t2, t3, t4):
    L = ["# Bảng tổng hợp cấu hình cuối cùng (JF), fold 0–4", "",
         f"Tạo lúc {os.popen('date +%F_%T').read().strip()} từ các file JSON đã lưu. "
         "Mỗi fold một seed (42). Dấu * là CI 95% không chứa 0.", "",
         "**Các nhánh:** A = V4 gốc niêm phong (Faster R-CNN). B = V4-CF huấn luyện lại, Faster R-CNN. "
         "D2 = V4-CF + YOLOv8 joint (ngưỡng theo tỉ lệ của Faster R-CNN). J = D2 + câu hỏi–đáp BTXRD. "
         "**JF = cấu hình cuối cùng**: J với ngưỡng riêng của YOLOv8 (FracAtlas mức 0,02, chọn trên model_val fold 0; "
         "BTXRD chọn trên model_val fold 0 chỉ ở bước suy luận, xem bảng 4e). A, B, D2, J chỉ là tham chiếu.", "",
         "## 1. FracAtlas test (phát hiện gãy xương)", "",
         "| Fold | A | B | D2 | J | **JF** | JF − A | JF − B | JF − J |", "|---|---|---|---|---|---|---|---|---|"]
    for f in FOLDS:
        a = t1[f]["arms"]
        L.append(f"| {f} | " + " | ".join(fmt((a[k] or {}).get("auprc")) for k in ARMS) +
                 f" | {fmt_d(t1[f]['JF_vs_A'])} | {fmt_d(t1[f]['JF_vs_B'])} | {fmt_d(t1[f]['JF_vs_J'])} |")
    FJ = [f for f in FOLDS if (t1[f]["arms"].get("JF") or {}).get("auprc") is not None] or FOLDS
    TBL = "**TB**" if FJ == FOLDS else f"**TB fold {','.join(map(str, FJ))}**"
    L.append(f"| {TBL} | " + " | ".join(fmt(mean([(t1[f]["arms"][k] or {}).get("auprc") for f in FJ])) for k in ARMS) +
             " | " + " | ".join(fmt(mean([(t1[f][x] or {}).get("diff") for f in FJ])) for x in ("JF_vs_A", "JF_vs_B", "JF_vs_J")) + " |")
    L += ["", "AUPRC. AUROC:", "", "| Fold | A | B | D2 | J | **JF** |", "|---|---|---|---|---|---|"]
    for f in FOLDS:
        L.append(f"| {f} | " + " | ".join(fmt((t1[f]["arms"][k] or {}).get("auroc")) for k in ARMS) + " |")
    L.append(f"| {TBL} | " + " | ".join(fmt(mean([(t1[f]["arms"][k] or {}).get("auroc") for f in FJ])) for k in ARMS) + " |")
    L += ["", "**Bỏ ảnh có dụng cụ kim loại** (98% ảnh có dụng cụ được gán nhãn gãy: lối tắt). AUPRC, bootstrap theo ảnh:", "",
          "| Fold | Ảnh có dụng cụ (test) | A | B | D2 | J | **JF** | JF − B |", "|---|---|---|---|---|---|---|---|"]
    for f in FOLDS:
        nh = t1[f]["no_hardware"]
        L.append(f"| {f} | {t1[f]['n_hardware_test']} | " + " | ".join(fmt((nh.get(k) or {}).get("auprc")) for k in ARMS) +
                 f" | {fmt_d(nh.get('JF_vs_B'))} |")
    L.append(f"| {TBL} | | " + " | ".join(fmt(mean([(t1[f]["no_hardware"].get(k) or {}).get("auprc") for f in FJ])) for k in ARMS) + " | |")
    L += ["", "## 2. BTXRD (u xương): test / external", ""]
    for role, lab in (("test", "test (cùng nguồn)"), ("external", "external (trung tâm khác)")):
        L += [f"**{lab}**", "", "| Fold | Có u: AUROC D2 / J / **JF** | Ác tính: AUROC D2 / J / **JF** | Chẩn đoán: acc D2 / J / **JF** | JF − J (u / ác tính / chẩn đoán) |",
              "|---|---|---|---|---|"]
        for f in FOLDS:
            r = t2[f][role]
            cell = lambda m: " / ".join(fmt((r.get(k) or {}).get(m)) for k in ("D2", "J", "JF"))  # noqa: E731
            dj = r.get("JF_vs_J") or {}
            L.append(f"| {f} | {cell('tumor_auroc')} | {cell('malignancy_auroc')} | {cell('diagnosis_accuracy')} | "
                     f"{fmt_d(dj.get('tumor_auroc'))} / {fmt_d(dj.get('malignancy_auroc'))} / {fmt_d(dj.get('diagnosis_accuracy'))} |")
        FJ2 = [f for f in FOLDS if (t2[f][role].get("JF") or {}).get("tumor_auroc") is not None] or FOLDS
        mc = lambda m: " / ".join(fmt(mean([(t2[f][role].get(k) or {}).get(m) for f in FJ2])) for k in ("D2", "J", "JF"))  # noqa: E731
        TB2 = "**TB**" if FJ2 == FOLDS else f"**TB fold {','.join(map(str, FJ2))}**"
        L += [f"| {TB2} | {mc('tumor_auroc')} | {mc('malignancy_auroc')} | {mc('diagnosis_accuracy')} | |", ""]
    L += ["B không có cột BTXRD: B không học BTXRD và không có hộp u xương. D2 không học câu hỏi–đáp BTXRD "
          "(chỉ có hộp u từ YOLOv8), nên cột D2 cho thấy phần đóng góp của dữ liệu BTXRD; chẩn đoán của D2 = 0.", "",
          "## 3. Lớp tin cậy (xác minh / từ chối trả lời)", "",
          "Độ tin cậy đã hiệu chỉnh + điểm hộp của YOLOv8, khớp trên val, điểm vận hành ~80% câu được trả lời. "
          "Ô = độ phủ: độ chính xác tất cả → độ chính xác phần được trả lời.", "",
          "| Fold | Nhánh | Gãy (test) | U (test) | U (external) | Ác tính (test) | Ác tính (external) | Chẩn đoán (test) | Chẩn đoán (external) |",
          "|---|---|---|---|---|---|---|---|---|"]
    keys = ["fracture/test", "tumor/test", "tumor/external", "malignancy/test", "malignancy/external", "diagnosis/test", "diagnosis/external"]
    for f in FOLDS:
        for k in ("J", "JF"):
            r = t3[f].get(k)
            cells = ["—" if not r or kk not in r else f"{r[kk]['coverage']:.0%}: {r[kk]['acc_all']:.3f} → {r[kk]['acc_answered']:.3f}" for kk in keys]
            L.append(f"| {f} | {k} | " + " | ".join(cells) + " |")
    L += ["", "Từ chối trả lời đổi độ phủ lấy độ chính xác; nó không làm tăng AUROC.", "",
          "## 4. Kết quả bổ trợ trên fold 0", "", "**a. J với 3 seed (42 / 43 / 44)**", "",
          "| Seed | FracAtlas AUPRC | U AUROC test / ext | Ác tính AUROC test / ext | Chẩn đoán acc test / ext |", "|---|---|---|---|---|"]
    for s, e in t4["seeds"].items():
        L.append(f"| {s} | {fmt(e.get('fracatlas_auprc'))} | {fmt(e.get('test_tumor_auroc'))} / {fmt(e.get('external_tumor_auroc'))} | "
                 f"{fmt(e.get('test_malignancy_auroc'))} / {fmt(e.get('external_malignancy_auroc'))} | "
                 f"{fmt(e.get('test_diagnosis_accuracy'))} / {fmt(e.get('external_diagnosis_accuracy'))} |")
    L += ["", "Seed chỉ đổi thứ tự lô BTXRD và dropout, nên độ dao động theo seed là cận dưới.", "",
          "**b. Đối chứng KG**", "", "| Nhánh | FracAtlas AUPRC | U AUROC test / ext | Ác tính AUROC test / ext | Chẩn đoán acc test / ext |",
          "|---|---|---|---|---|"]
    for lab, e in t4["kg"].items():
        if lab.startswith("K_vs_J"):
            continue
        L.append(f"| {lab} | {fmt(e.get('fracatlas_auprc'))} | {fmt(e.get('test_tumor_auroc'))} / {fmt(e.get('external_tumor_auroc'))} | "
                 f"{fmt(e.get('test_malignancy_auroc'))} / {fmt(e.get('external_malignancy_auroc'))} | "
                 f"{fmt(e.get('test_diagnosis_accuracy'))} / {fmt(e.get('external_diagnosis_accuracy'))} |")
    for role in ("test", "external"):
        d = t4["kg"].get(f"K_vs_J_{role}")
        if d:
            L.append(f"\nK − J ({role}): u {fmt_d(d.get('tumor'))}, ác tính {fmt_d(d.get('malignancy'))}, chẩn đoán {fmt_d(d.get('diagnosis'))}")
    L += ["", "**c. Thông tin tham chiếu: vị trí tổn thương (chẩn đoán BTXRD, J, chấm theo ứng viên)**", "",
          "| Cách dùng | Acc test (Δ so với chỉ ảnh) | Acc external (Δ) |", "|---|---|---|"]
    for meth, v in t4["lesion_site"].items():
        cell = lambda r: fmt(v[r]["accuracy"]) + ("" if v[r]["diff"] is None else f" ({fmt_d({'diff': v[r]['diff'], 'ci95': v[r]['ci95']})})")  # noqa: E731
        L.append(f"| {meth} | {cell('test')} | {cell('external')} |")
    L += ["", "Vị trí tổn thương không được dùng cho câu hỏi có u hay không: BTXRD chỉ ghi vị trí cho ảnh có u.", "",
          "**d. Grounding (D2, fold 0, test): V4 có dựa vào vị trí hộp không?**", "",
          "| Mức (ảnh bình thường có hộp, model_val) | AUPRC model_val / test | Precision hộp @0,5 | Ảnh gãy có hộp trúng | Ảnh gãy không hộp (độ nhạy) | Placebo ΔAUPRC | Oracle ΔAUPRC |",
          "|---|---|---|---|---|---|---|"]
    for lvl, v in t4["grounding"].items():
        if lvl.startswith("yolo_best"):
            continue
        lab = "0.284 (tỉ lệ của Faster R-CNN)" if lvl.startswith("0.284") else lvl
        L.append(f"| {lab} | {fmt(v['v4_model_val_auprc'])} / {fmt(v['v4_test_auprc'])} | {fmt(v['box_precision@0.5'])} | "
                 f"{fmt(v['image_hit_rate@0.3'])} | {fmt(v['fractured_without_box'])} ({fmt(v['sens_without_box'])}) | "
                 f"{fmt_d(v['placebo_auprc'])} | {fmt_d(v['oracle_auprc'])} |")
    yb = t4["grounding"].get("yolo_best_f1@0.5_model_val")
    if yb:
        L.append(f"\nĐiểm F1@0,5 tốt nhất của riêng YOLOv8 trên model_val: ngưỡng {yb['threshold']:.3f}, "
                 f"{yb['m_rate_nonfractured']:.1%} ảnh bình thường có hộp.")
    d = t4["d2_level_0.02"]
    L.append(f"\nD2 mức 0,02 so với D2 (ngưỡng theo Faster R-CNN), test AUPRC: fold 0 {fmt_d(d['fold0_vs_D2'])}, fold 1 {fmt_d(d['fold1_vs_D2'])}.")
    sel = t4["btxrd_level_selection"]
    L += ["", "**e. Chọn ngưỡng BTXRD (chỉ suy luận, JF fold 0, BTXRD model_val; không nhìn test / external)**", ""]
    if sel:
        L += ["| Tỉ lệ ảnh bình thường có hộp | U AUROC | Ác tính AUROC | Chẩn đoán acc |", "|---|---|---|---|"]
        for l, c in sel["candidates"].items():
            mark = " **(chọn)**" if l == sel["chosen_level"] else ""
            L.append(f"| {l}{mark} | {fmt(c['tumor_auroc'])} | {fmt(c['malignancy_auroc'])} | {fmt(c['diagnosis_accuracy'])} |")
        L.append(f"\nQuy tắc khai báo trước: {sel['rule']}.")
    else:
        L.append("Chưa chạy.")
    L += ["", "## 5. Hạn chế cần ghi trong luận văn", "",
          "- **Proposal BTXRD lúc huấn luyện V4 là in-sample.** Ba detector cross-fit của mỗi fold chỉ chia FracAtlas train; "
          "BTXRD train được đưa nguyên vào cả ba. Vì vậy ảnh BTXRD lúc huấn luyện nhận hộp từ detector đã học chính ảnh đó "
          "(tỉ lệ ảnh có u được đặt hộp cao hơn lúc test). V4 có thể tin hộp u quá mức, nhất là trên tập external. "
          "Sửa: chia luôn BTXRD train trong 3 detector cross-fit (bản IEEE).",
          "- Mỗi fold chỉ một seed. Seed chỉ đổi thứ tự lô BTXRD và dropout, nên độ dao động là cận dưới.",
          "- Mức ngưỡng FracAtlas 0,02 được chọn trên model_val của fold 0 rồi cố định cho các fold khác; ở fold 1, "
          "model_val lại chuộng mức khác. Ngưỡng BTXRD chọn trên model_val fold 0 chỉ ở bước suy luận; JF được huấn luyện với BTXRD mức 0,02.",
          "- Detector YOLOv8s, ảnh 640, 60 epoch; chưa tinh chỉnh kích thước mô hình hay độ phân giải ảnh.",
          "- Lợi ích của YOLOv8 đến từ việc có hộp với độ tin cậy cao (placebo ở mức 0,02 không làm giảm), "
          "không phải từ việc V4 định vị đúng ổ gãy.",
          "- Ảnh có dụng cụ kim loại là lối tắt cho nhãn gãy; kết quả khi bỏ nhóm này nằm ở bảng 1.",
          "- Chẩn đoán tên bệnh BTXRD còn yếu, nhất là trên tập external; bốn lớp hiếm không bao giờ được dự đoán.",
          "- Fold 4 chưa chạy (tạm dừng theo phạm vi đóng băng 20/11)."]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0, t1, t2, t3, t4 = table0(), table1(), table2(), table3(), table4()
    (OUT / "summary.json").write_text(json.dumps({"detector": t0, "fracatlas": t1, "btxrd": t2, "trust": t3, "fold0_extras": t4}, indent=2))
    md = markdown(t1, t2, t3, t4).replace("## 1. FracAtlas test", detector_md(t0) + "## 1. FracAtlas test", 1)
    (OUT / "bang_tong_hop.md").write_text(md)
    print("wrote", OUT / "bang_tong_hop.md")


if __name__ == "__main__":
    main()
