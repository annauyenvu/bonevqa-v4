"""Số liệu của giai đoạn cuối (JF) cho luận văn, đọc từ yolo_pilot/runs/summary_final/summary.json và các tệp đánh giá.

Ghi src/results_jf.json, build.py nạp vào RES['jf'] để văn bản dùng {R:jf....}. Ô của fold chưa có số liệu (fold 4 khi
đang chạy) và mọi trung bình cần fold đó được ghi '…'; dựng lại luận văn sau khi fold 4 xong là các ô tự điền.
python3 -I src/gen_jf.py  (chạy lại summary_final.py trước nếu kết quả mới)
"""
import json
from pathlib import Path

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
S = json.loads((ROOT / "yolo_pilot/runs/summary_final/summary.json").read_text())
JOINT = ROOT / "v4_grounded_next/runs/joint"
FOLDS = ["0", "1", "2", "3", "4"]
NA = "…"
JF = "JF_v4cf_yolojoint_btxrd_r0.02"


def num(x, d=3, sign=False):
    if x is None:
        return NA
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def pct(x):
    return NA if x is None else f"{round(100 * x)}%"


def ci(dd):
    if not dd or dd.get("diff") is None:
        return NA
    lo, hi = dd["ci95"][:2]
    return f"{num(dd['diff'], sign=True)} [{num(lo, sign=True)}; {num(hi, sign=True)}]"


def mean(vals):
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return None
        d = d[p]
    return d


R = {"fa": {}, "nohw": {}, "bt": {}, "tr": {}, "det": {}, "cls": {}, "ba": {}, "count": {}}
fa = S["fracatlas"]
for arm in ("A", "B", "D2", "J", "JF"):
    for m in ("auprc", "auroc"):
        vals = [g(fa, f, "arms", arm, m) for f in FOLDS]
        R["fa"].setdefault(arm, {})[m] = {f: num(v) for f, v in zip(FOLDS, vals)} | {"mean": num(mean(vals))}
    vals = [g(fa, f, "no_hardware", arm, "auprc") for f in FOLDS]
    R["nohw"][arm] = {f: num(v) for f, v in zip(FOLDS, vals)} | {"mean": num(mean(vals))}
R["nohw"]["n"] = {f: (str(g(fa, f, "n_hardware_test")) if g(fa, f, "n_hardware_test") is not None else NA) for f in FOLDS}
R["nohw"]["JFB"] = {f: ci(g(fa, f, "no_hardware", "JF_vs_B")) for f in FOLDS}
for key, src in (("JFA", "JF_vs_A"), ("JFB", "JF_vs_B"), ("JFJ", "JF_vs_J")):
    diffs = [g(fa, f, src, "diff") for f in FOLDS]
    sig = [None if g(fa, f, src) is None else g(fa, f, src, "ci95")[0] > 0 for f in FOLDS]
    R["fa"][key] = {f: ci(g(fa, f, src)) for f in FOLDS} | {"mean": num(mean(diffs), sign=True),
                                                             "nsig": NA if None in sig else str(sum(sig)),
                                                             "npos": NA if None in diffs else str(sum(d > 0 for d in diffs)),
                                                             "min": num(None if None in diffs else min(diffs), sign=True),
                                                             "max": num(None if None in diffs else max(diffs), sign=True)}
for a, b in (("D2", "B"), ("J", "D2"), ("JF", "J"), ("JF", "A")):
    d = [None if g(fa, f, "arms", a, "auprc") is None or g(fa, f, "arms", b, "auprc") is None else g(fa, f, "arms", a, "auprc") - g(fa, f, "arms", b, "auprc") for f in FOLDS]
    R["fa"][f"d_{a}_{b}"] = {"mean": num(mean(d), sign=True)} | {f: num(v, sign=True) for f, v in zip(FOLDS, d)}
    R["fa"][f"d_{a}_{b}"]["nneg"] = NA if None in d else str(sum(v < 0 for v in d))
drop = [None if g(fa, f, "arms", arm, "auprc") is None or g(fa, f, "no_hardware", arm, "auprc") is None else g(fa, f, "arms", arm, "auprc") - g(fa, f, "no_hardware", arm, "auprc")
        for arm in ("A", "B", "D2", "J", "JF") for f in FOLDS]
R["nohw"]["drop_min"] = num(None if None in drop else min(drop)); R["nohw"]["drop_max"] = num(None if None in drop else max(drop))

bt = S["btxrd"]
for role in ("test", "external"):
    R["bt"][role] = {}
    for arm in ("D2", "J", "JF"):
        for m in ("tumor_auroc", "malignancy_auroc", "diagnosis_accuracy"):
            vals = [g(bt, f, role, arm, m) for f in FOLDS]
            R["bt"][role].setdefault(arm, {})[m] = {f: num(v) for f, v in zip(FOLDS, vals)} | {"mean": num(mean(vals))}
    for m in ("tumor_auroc", "malignancy_auroc", "diagnosis_accuracy"):
        d = [g(bt, f, role, "JF_vs_J", m, "diff") for f in FOLDS]
        sig = [None if g(bt, f, role, "JF_vs_J", m) is None else g(bt, f, role, "JF_vs_J", m, "ci95")[0] > 0 for f in FOLDS]
        R["bt"][role].setdefault("JFJ", {})[m] = {f: ci(g(bt, f, role, "JF_vs_J", m)) for f in FOLDS} | {
            "mean": num(mean(d), sign=True), "npos_sig": NA if None in sig else str(sum(sig))}

# chẩn đoán theo lớp, độ chính xác cân bằng (JF, trung bình các fold)
for role in ("test", "external"):
    rec, ba, n = {}, [], {}
    for f in FOLDS:
        p = JOINT / f"fold_{f}" / f"{JF}_btxrd_{role}.json"
        m = json.loads(p.read_text())["metrics"]["diagnosis"] if p.exists() else None
        ba.append(m and m["balanced_accuracy"])
        for k in (m or {}).get("recall_by_class", {}):
            rec.setdefault(k, []).append(m["recall_by_class"][k]); n[k] = m["n_by_class"][k]
        if m is None:
            for k in rec:
                rec[k].append(None)
    R["ba"][role] = num(mean(ba))
    R["cls"][role] = {k.replace(" ", "_"): {"recall": num(mean(v) if len(v) == len(FOLDS) else None), "n": str(n[k])} for k, v in rec.items()}

tr = S["trust"]
for key in ("fracture/test", "tumor/test", "tumor/external", "malignancy/test", "malignancy/external", "diagnosis/test", "diagnosis/external"):
    vals = {m: [g(tr, f, "JF", key, m) for f in FOLDS] for m in ("coverage", "acc_all", "acc_answered")}
    R["tr"][key.replace("/", "_")] = {"cov": pct(mean(vals["coverage"])), "all": num(mean(vals["acc_all"])), "ans": num(mean(vals["acc_answered"]))}

det = S["detector"]
for ds in ("fracatlas", "btxrd"):
    for kind, sets in (("own", ["own_best_test", "own_best_external"]), ("v4", ["v4_level_test", "v4_level_external"])):
        for st in sets:
            if ds == "fracatlas" and st.endswith("external"):
                continue
            role = "external" if st.endswith("external") else "test"
            for m in ("box_precision@0.5", "box_recall@0.5", "f1@0.5", "image_hit_rate@0.3", "m_rate_nonfractured"):
                vals = [g(det, f, ds, st, m) for f in FOLDS]
                R["det"].setdefault(ds, {}).setdefault(role, {}).setdefault(kind, {})[m.split("@")[0].replace("box_", "")] = num(mean(vals))
W = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm"}


def of5(n):
    """Cụm 'cả năm fold' / 'bốn trong năm fold' / 'không fold nào' cho văn bản."""
    if n == NA:
        return NA
    return "cả năm fold" if n == "5" else "không fold nào" if n == "0" else f"{W[n]} trong năm fold"


for key in ("JFA", "JFB", "JFJ"):
    for c in ("npos", "nsig"):
        R["fa"][key][c + "_t"] = of5(R["fa"][key][c])
for k in [k for k in R["fa"] if k.startswith("d_")]:
    n = R["fa"][k]["nneg"]
    R["fa"][k]["nneg_t"] = NA if n == NA else ("không fold nào" if n == "0" else f"{W[n]} fold")
R["count"]["folds"] = "năm"
Path(__file__).with_name("results_jf.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
print("wrote results_jf.json; missing cells:", json.dumps(R, ensure_ascii=False).count(NA))
