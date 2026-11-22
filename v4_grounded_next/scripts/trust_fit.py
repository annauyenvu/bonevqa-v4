"""Fit the trust layer of a joint run on its validation scores, then read it on test / external.  No model, no training.

    source v4_grounded_next/env.sh; cd v4_grounded_next
    $PY -m scripts.trust_fit J_v4cf_yolojoint_btxrd

Inputs (all written earlier by inference-only scripts):
    runs/yolo_ablation/fold_<f>/<name>_{calibration,test}.json     fracture, P(yes) = p_final        (scripts.eval_arm)
    runs/joint/fold_<f>/ref_<name>_{model_val,test,external}.json  tumor, malignancy, diagnosis      (scripts.reference_score)
    runs/crossfit_yolo_joint/...                                   the proposal files the run was evaluated with
Outputs in runs/joint/fold_<f>/:
    trust_<name>.json        the fitted layer (vg.trust.TrustLayer.load)
    trust_<name>_eval.json   what it does on test and external
    trust_<name>_eval.md     the same as tables, in Vietnamese
The fit never sees test or external rows.  FracAtlas uses the calibration role; BTXRD has no separate calibration split,
so its model_val role is used (the run saved its last checkpoint; model_val was only monitored).
"""
from __future__ import annotations

import argparse
import datetime
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from vg import trust as T
from vg.paths import CV, RUNS, dump_json, guard_write, read_json, sha256

N_BOOT, SEED = 2000, 20261114
VI = {"fracture": "Gãy xương (FracAtlas)", "tumor": "Có u (BTXRD)", "malignancy": "Lành / ác (BTXRD)", "diagnosis": "Chẩn đoán loại u (BTXRD)"}
SIG_VI = {"head": "chỉ đầu ra có / không", "head_detector": "đầu ra + điểm detector"}


def sources(name, fold):
    abl, jnt, cf = RUNS / "yolo_ablation" / f"fold_{fold}", RUNS / "joint" / f"fold_{fold}", RUNS / "crossfit_yolo_joint"
    return {"fracture": {"calibration": abl / f"{name}_calibration.json", "test": abl / f"{name}_test.json"},
            "btxrd": {role: jnt / f"ref_{name}_{role}.json" for role in ("model_val", "test", "external")},
            "fracatlas_proposals": cf / f"fold_{fold}" / f"proposals_fold{fold}.json",
            "btxrd_proposals": cf / ("btxrd" if str(fold) == "0" else f"btxrd_fold{fold}") / "proposals_btxrd_insample.json"}


def load_scores(name, fold):
    """{endpoint: {role: dict(p, y, s, groups, ids)}} and the diagnosis block, labels included (read after inference)."""
    src = sources(name, fold)
    fa, bt = read_json(src["fracatlas_proposals"]), read_json(src["btxrd_proposals"])
    comp = {}
    for gid, images in enumerate(read_json(CV / "manifest.json")["components"]):
        for im in images:
            comp[str(im).rsplit(".", 1)[0]] = gid
    out = {"fracture": {}, "tumor": {}, "malignancy": {}}
    for role, f in src["fracture"].items():
        rows = read_json(f)["rows"]
        out["fracture"]["val" if role == "calibration" else role] = {
            "p": np.array([r["p_final"] for r in rows]), "y": np.array([r["y"] for r in rows]),
            "s": np.array([T.top_box_score(fa[role][r["image_id"]]) for r in rows]),
            "groups": [comp[r["image_id"]] for r in rows], "ids": [r["image_id"] for r in rows]}
    refs = {role: read_json(f) for role, f in src["btxrd"].items()}
    for ep in ("tumor", "malignancy"):
        for role, R in refs.items():
            rows = R["endpoints"][ep]
            out[ep]["val" if role == "model_val" else role] = {
                "p": np.array([r["plain"] for r in rows]), "y": np.array([int(r["truth"] == "yes") for r in rows]),
                "s": np.array([T.top_box_score(bt[role][r["image_id"]]) for r in rows]),
                "groups": [(r["center"], r["group"]) for r in rows], "ids": [r["image_id"] for r in rows]}
    cands = refs["test"]["candidates"]
    dx = {("val" if role == "model_val" else role): {
        "z": np.array([r["plain"] for r in R["endpoints"]["diagnosis"]]),
        "y": np.array([cands.index(r["truth"]) for r in R["endpoints"]["diagnosis"]]),
        "groups": [(r["center"], r["group"]) for r in R["endpoints"]["diagnosis"]]} for role, R in refs.items()}
    files = [src["fracture"]["calibration"], src["btxrd"]["model_val"], src["fracatlas_proposals"], src["btxrd_proposals"]]
    return out, dx, cands, {str(f): sha256(f) for f in files}, refs["test"].get("checkpoint_sha256"), refs["test"].get("tag")


def boot_gain(groups, ok, keep, rng):
    mem = defaultdict(list)
    for i, g in enumerate(groups):
        mem[g].append(i)
    gl, d = list(mem.values()), []
    for _ in range(N_BOOT):
        ix = np.array([i for k in rng.integers(0, len(gl), len(gl)) for i in gl[k]])
        if keep[ix].any():
            d.append(ok[ix][keep[ix]].mean() - ok[ix].mean())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def selective(conf, tau, ok, groups, rng, positives=None, said_no=None):
    keep = conf >= tau
    out = {"coverage": float(keep.mean()), "accuracy_all": float(ok.mean()),
           "accuracy_answered": float(ok[keep].mean()) if keep.any() else None,
           "gain": float(ok[keep].mean() - ok.mean()) if keep.any() else None, "gain_ci95": boot_gain(groups, ok, keep, rng),
           "share_of_errors_deferred": float((~ok & ~keep).sum() / max((~ok).sum(), 1)),
           "accuracy_deferred": float(ok[~keep].mean()) if (~keep).any() else None}
    if positives is not None:
        missed = positives & said_no
        out.update(missed_positives_all=int(missed.sum()), missed_positives_answered=int((missed & keep).sum()))
    return out


def main(a):
    jnt = RUNS / "joint" / f"fold_{a.fold}"
    art, ev_json, ev_md = jnt / f"trust_{a.name}.json", jnt / f"trust_{a.name}_eval.json", jnt / f"trust_{a.name}_eval.md"
    assert not art.exists() and not ev_json.exists(), f"{art} exists"
    scores, dx, cands, hashes, ck_sha, tag = load_scores(a.name, a.fold)
    spec = {"schema": T.SCHEMA, "name": a.name, "tag": tag, "fold": a.fold, "checkpoint_sha256": ck_sha,
            "fitted": datetime.datetime.now().isoformat(timespec="seconds"), "fitted_on": hashes,
            "fit_roles": {"fracture": "calibration", "tumor": "model_val", "malignancy": "model_val", "diagnosis": "model_val"},
            "endpoints": {ep: T.fit_closed(v["val"]["p"], v["val"]["y"], v["val"]["s"]) for ep, v in scores.items()},
            "diagnosis": T.fit_diagnosis(dx["val"]["z"], dx["val"]["y"], cands)}
    dump_json(art, spec, exclusive=True)
    layer = T.TrustLayer.load(art)
    rng = np.random.default_rng(SEED)
    res = {"name": a.name, "trust_file": str(art), "trust_sha256": sha256(art), "n_boot": N_BOOT, "seed": SEED, "endpoints": {}}
    for ep, by_role in scores.items():
        res["endpoints"][ep] = {}
        for role in [r for r in by_role if r != "val"]:
            v = by_role[role]
            pred = (v["p"] >= 0.5).astype(int)
            ok = pred == v["y"]
            block = {"n": int(len(ok)), "positives": int(v["y"].sum()), "accuracy": float(ok.mean()), "signals": {}}
            for sig in T.SIGNALS:
                conf = layer.confidence(ep, v["p"], v["s"], sig)
                block["signals"][sig] = {"auroc_correct": float(roc_auc_score(ok, conf)), **{
                    T._coverage_key(c): selective(conf, layer.threshold(ep, c, sig), ok, v["groups"], rng, v["y"] == 1, pred == 0)
                    for c in T.COVERAGES}}
            res["endpoints"][ep][role] = block
    res["endpoints"]["diagnosis"] = {}
    for role in ("test", "external"):
        p = T.softmax_t(dx[role]["z"], layer.dx["temperature"])
        y = dx[role]["y"]
        ok = p.argmax(1) == y
        top3 = np.array([y[i] in np.argsort(-p[i])[:3] for i in range(len(y))])
        res["endpoints"]["diagnosis"][role] = {
            "n": int(len(y)), "accuracy": float(ok.mean()), "top3_contains_truth": float(top3.mean()),
            **{T._coverage_key(c): selective(p.max(1), layer.dx["threshold"][T._coverage_key(c)], ok, dx[role]["groups"], rng) for c in T.COVERAGES}}
    dump_json(ev_json, res, exclusive=True)

    f = lambda x, sign=False: "—" if x is None else (f"{x:+.3f}" if sign else f"{x:.3f}").replace(".", ",").replace("-", "−")
    L = [f"# Lớp tin cậy của {a.name} (fold {a.fold})", "",
         "Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.",
         "\"Tỉ lệ trả lời\" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.", ""]
    for cov in T.COVERAGES:
        k = T._coverage_key(cov)
        L += [f"## Ngưỡng giữ {cov:.0%} số ca trên val", "",
              "| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ep in ("fracture", "tumor", "malignancy"):
            for role, b in res["endpoints"][ep].items():
                for sig in T.SIGNALS:
                    s = b["signals"][sig][k]
                    L.append(f"| {VI[ep]} | {role} | {SIG_VI[sig]} | {f(b['accuracy'])} | {f(s['coverage'])} | {f(s['accuracy_answered'])} | "
                             f"{f(s['gain'], True)} [{f(s['gain_ci95'][0], True)}; {f(s['gain_ci95'][1], True)}] | {f(s['share_of_errors_deferred'])} | "
                             f"{s['missed_positives_all']} → {s['missed_positives_answered']} | {f(b['signals'][sig]['auroc_correct'])} |")
        for role, b in res["endpoints"]["diagnosis"].items():
            s = b[k]
            L.append(f"| {VI['diagnosis']} | {role} | điểm 9 đáp án ứng viên | {f(b['accuracy'])} | {f(s['coverage'])} | {f(s['accuracy_answered'])} | "
                     f"{f(s['gain'], True)} [{f(s['gain_ci95'][0], True)}; {f(s['gain_ci95'][1], True)}] | {f(s['share_of_errors_deferred'])} | — | — |")
        L.append("")
    L += ["## Danh sách 3 chẩn đoán phân biệt", "", "| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |", "|---|---|---|"]
    L += [f"| {role} | {f(b['accuracy'])} | {f(b['top3_contains_truth'])} |" for role, b in res["endpoints"]["diagnosis"].items()]
    guard_write(ev_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print({"trust": str(art), "eval": str(ev_json), "md": str(ev_md)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fold", default="0")
    main(ap.parse_args())
