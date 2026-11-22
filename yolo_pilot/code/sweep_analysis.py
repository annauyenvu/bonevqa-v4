"""After the fold-0 threshold sweep: when is YOLOv8 at its best, how often are its boxes right, and does V4 use
WHERE the box is (grounding), with no Faster R-CNN anchor in any of the criteria.  Asked 2026-11-19.

    python3 -I code/sweep_analysis.py make <level>      # placebo + oracle-location proposal files of one level
    venv/bin/python -I code/sweep_analysis.py report     # (numpy) writes runs/sweep_analysis/fold0.{json,md}

Levels: 0.02, 0.15, 0.2840909090909091 (= the existing D2, rate of the sealed Faster R-CNN), max.

1. Detector alone (outer YOLOv8, top-3 contract, no V4).  Box-level precision / recall at IoU 0.3 and 0.5 over
   the whole score range on model_val; the detector's own best operating point = max F1 (IoU 0.5) on model_val,
   and the non-fractured proposal rate it implies.  Same quantities on test at each sweep level.
2. V4 per level: model_val AUPRC (level selection), test AUPRC / AUROC, paired component bootstrap vs the
   existing D2 (compare_test_sweep.json, written by lane_sweep.sh).
3. Grounding of the answers on test, per level, two kinds of evidence:
   a. descriptive: fractured images split into box hits the fracture (IoU >= 0.3) / boxes all miss / no box;
      sensitivity at 0.5 in each; "grounded" true positives = answered yes AND a box on the fracture.
   b. interventions (inference only, same checkpoint, only the test proposals change; other roles identical):
      placebo    every box moved to a random position of the same size, on bone (mean grey > 40), IoU 0 with
                 every fracture box and < 0.1 with the original box; count and scores kept.
      oracleloc  fractured images that have proposals get the fracture boxes in their first slots (count and
                 scores kept, extra slots keep the original boxes); every other image unchanged.
      If V4 uses the location, placebo lowers AUPRC and oracleloc raises it.  GT is used only to build the
      intervention files, never seen by the model at training.
"""
import json
import random
import sys
from pathlib import Path

PILOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
VG = Path("/Users/uyen/Downloads/BONEVQA/v4_grounded_next")
SW = VG / "runs/crossfit_yolo_joint_sweep"
ABL = VG / "runs/yolo_ablation/fold_0"
LEVELS = ["0.02", "0.15", "0.2840909090909091", "max"]
SEED = 20261119
sys.path.insert(0, str(PILOT / "code"))
from eval_detectors import iou  # noqa: E402


def names(level):
    """eval_arm name of the native run, checkpoint tag, proposal file of one level."""
    if level.startswith("0.284"):
        return "D2_v4cf_yolojoint", "yolo_fold0/V4-CF_yolojoint", SW / f"r{level}/fold_0/proposals_fold0.json"
    return f"D2_sweep_r{level}", f"yolo_fold0_sweep/V4-CF_yolojoint_r{level}", SW / f"r{level}/fold_0/proposals_fold0.json"


def gt0():
    return json.loads((PILOT / "datasets/fa_fold0/gt.json").read_text())


# ------------------------------------------------------------------ make
def make(level):
    from PIL import Image  # venv only
    import numpy as np
    gt = gt0()
    src = names(level)[2]
    doc = json.loads(src.read_text())
    rng = random.Random(SEED)
    small = {}

    def grey(iid):
        if iid not in small:
            im = Image.open(PILOT / f"datasets/fa_fold0/images/test/{iid}.jpg").convert("L")
            w, h = im.size
            small[iid] = (np.asarray(im.resize((max(1, w // 8), max(1, h // 8)))), w, h)
        return small[iid]

    def placebo_box(iid, b):
        arr, w, h = grey(iid)
        bw, bh = b[2] - b[0], b[3] - b[1]
        fx = gt[iid]["boxes"]
        for strict in (True, False):
            for _ in range(400):
                x1 = rng.uniform(0, max(1.0, w - bw)); y1 = rng.uniform(0, max(1.0, h - bh))
                c = [x1, y1, x1 + bw, y1 + bh]
                if any(iou(c, g) > 0 for g in fx) or iou(c, b) >= 0.1:
                    continue
                patch = arr[int(y1) // 8:int(y1 + bh) // 8 + 1, int(x1) // 8:int(x1 + bw) // 8 + 1]
                if strict and (patch.size == 0 or patch.mean() <= 40):
                    continue
                return c + [b[4]], strict
        return None, False

    placebo, oracle, stats = {}, {}, {"boxes": 0, "on_bone": 0, "off_bone": 0, "kept_unmoved": 0, "oracle_images": 0}
    for iid, props in doc["test"].items():
        new = []
        for b in props:
            stats["boxes"] += 1
            c, strict = placebo_box(iid, b)
            if c is None:
                stats["kept_unmoved"] += 1; new.append(b)
            else:
                stats["on_bone" if strict else "off_bone"] += 1; new.append(c)
        placebo[iid] = new
        fx = gt[iid]["boxes"]
        if props and gt[iid]["fractured"] and fx:
            stats["oracle_images"] += 1
            o = [list(fx[k]) + [props[k][4]] for k in range(min(len(props), len(fx)))]
            oracle[iid] = o + props[len(o):]
        else:
            oracle[iid] = props
    for kind, test in (("placebo", placebo), ("oracleloc", oracle)):
        out = dict(doc)
        out["test"] = test
        out["intervention"] = {"kind": kind, "seed": SEED, "source": str(src), "stats": stats}
        p = src.parent / f"proposals_fold0_{kind}.json"
        if not p.exists():
            p.write_text(json.dumps(out))
    print(level, stats)


# ------------------------------------------------------------------ report
def box_stats(pred, gt, ids, thr, topk=3):
    """Box-level precision / recall under the top-k contract, image-level rates."""
    tp = {0.3: 0, 0.5: 0}; nbox = 0; ngt = 0; gt_hit = {0.3: 0, 0.5: 0}
    img_hit = {0.3: 0, 0.5: 0}; nf_with = 0; nf = 0; fx_with = 0; fx = 0; nf_boxes = 0
    for i in ids:
        props = [b for b in sorted(pred.get(i, []), key=lambda b: -b[4]) if b[4] >= thr][:topk]
        g = gt[i]["boxes"] if gt[i]["fractured"] else []
        nbox += len(props)
        if gt[i]["fractured"]:
            fx += 1; fx_with += bool(props); ngt += len(g)
            for t in tp:
                tp[t] += sum(any(iou(p, x) >= t for x in g) for p in props)
                gt_hit[t] += sum(any(iou(p, x) >= t for p in props) for x in g)
                img_hit[t] += any(iou(p, x) >= t for p in props for x in g)
        else:
            nf += 1; nf_with += bool(props); nf_boxes += len(props)
    r = {"threshold": thr, "m_rate_fractured": fx_with / fx, "m_rate_nonfractured": nf_with / nf,
         "false_boxes_per_normal_image": nf_boxes / nf}
    for t in tp:
        P = tp[t] / nbox if nbox else None
        R = gt_hit[t] / ngt
        r[f"box_precision@{t}"] = P
        r[f"box_recall@{t}"] = R
        r[f"f1@{t}"] = 2 * P * R / (P + R) if P else 0.0
        r[f"image_hit_rate@{t}"] = img_hit[t] / fx
    return r


def report():
    import numpy as np
    gt = gt0()
    ids = {r: sorted(i for i, g in gt.items() if g["role"] == r) for r in ("val", "test")}
    outer = json.loads((PILOT / "runs/pred/fa_btxrd_cls0.json").read_text())["boxes"]
    out = {"detector_curve_model_val": [], "levels": {}}
    scores = sorted({b[4] for i in ids["val"] for b in outer.get(i, [])})
    grid = sorted(set(np.quantile(scores, np.linspace(0, 0.999, 400)).tolist()))
    for t in grid:
        out["detector_curve_model_val"].append(box_stats(outer, gt, ids["val"], t))
    for key in ("f1@0.5", "f1@0.3"):
        best = max(out["detector_curve_model_val"], key=lambda r: r[key])
        out[f"detector_best_{key}_model_val"] = best
        out[f"detector_best_{key}_on_test"] = box_stats(outer, gt, ids["test"], best["threshold"])
    cmp = ABL / "compare_test_sweep.json"
    cmp = json.loads(cmp.read_text()) if cmp.exists() else None
    for lv in LEVELS:
        name, tag, pf = names(lv)
        doc = json.loads(pf.read_text())
        L = {"thresholds": doc["thresholds"], "detector_test": box_stats(outer, gt, ids["test"], doc["thresholds"]["outer"])}
        comp = VG / f"runs/arms/{tag}/fold_0/seed_42/completed.json"
        if comp.exists():
            mv = json.loads(comp.read_text())["model_val"]
            L["v4_model_val"] = {"auprc": mv["auprc"], "auroc": mv["auroc"]}
        ev = ABL / f"{name}_test.json"
        if ev.exists():
            e = json.loads(ev.read_text())
            L["v4_test"] = {k: e["metrics"][k] for k in ("auprc", "auroc", "spec@0.5")}
            rows = {r["image_id"]: r for r in e["rows"]}
            groups = {"hit": [], "miss": [], "none": []}
            for i in ids["test"]:
                if not gt[i]["fractured"]:
                    continue
                props = doc["test"][i]
                g = "none" if not props else ("hit" if any(iou(p, x) >= 0.3 for p in props for x in gt[i]["boxes"]) else "miss")
                groups[g].append(rows[i]["p_final"])
            nfx = sum(len(v) for v in groups.values())
            tps = {g: sum(p >= .5 for p in v) for g, v in groups.items()}
            ntp = sum(tps.values())
            L["grounding_test"] = {
                "fractured": nfx,
                **{f"share_{g}": len(v) / nfx for g, v in groups.items()},
                **{f"sens@0.5_{g}": (tps[g] / len(v) if v else None) for g, v in groups.items()},
                "grounded_tp_over_fractured": tps["hit"] / nfx,
                "tp_with_box_on_fracture": tps["hit"] / ntp if ntp else None,
                "tp_with_wrong_boxes_only": tps["miss"] / ntp if ntp else None,
                "tp_without_box": tps["none"] / ntp if ntp else None}
            nrm = [i for i in ids["test"] if not gt[i]["fractured"]]
            fp_b = [rows[i]["p_final"] >= .5 for i in nrm if doc["test"][i]]
            fp_n = [rows[i]["p_final"] >= .5 for i in nrm if not doc["test"][i]]
            L["grounding_test"]["fp_rate_normal_with_box"] = float(np.mean(fp_b)) if fp_b else None
            L["grounding_test"]["fp_rate_normal_without_box"] = float(np.mean(fp_n)) if fp_n else None
            for kind in ("placebo", "oracleloc"):
                f = ABL / f"{name}_{kind}_test.json"
                if f.exists():
                    k = {r["image_id"]: r["p_final"] for r in json.loads(f.read_text())["rows"]}
                    hit = [i for i in ids["test"] if gt[i]["fractured"] and doc["test"][i]
                           and any(iou(p, x) >= 0.3 for p in doc["test"][i] for x in gt[i]["boxes"])]
                    L[f"{kind}_dp_fractured_hit"] = float(np.mean([k[i] - rows[i]["p_final"] for i in hit])) if hit else None
                    L[f"{kind}_flips@0.5"] = int(sum((k[i] >= .5) != (rows[i]["p_final"] >= .5) for i in rows))
            lc = ABL / f"compare_test_loc_r{lv}.json"
            if lc.exists():
                c = json.loads(lc.read_text())
                L["intervention_vs_native"] = {a: {m: c["diff_vs_reference"][a][m] for m in ("auprc", "auroc")}
                                               for a in c["diff_vs_reference"]}
        if cmp:
            if name in cmp["arms"]:
                L["v4_test_compare"] = cmp["arms"][name]
            if name in cmp.get("diff_vs_reference", {}):
                L["diff_vs_B"] = cmp["diff_vs_reference"][name]
        out["levels"][lv] = L
    sel = [(lv, L["v4_model_val"]["auprc"]) for lv, L in out["levels"].items() if "v4_model_val" in L]
    out["selected_on_model_val"] = max(sel, key=lambda x: x[1])[0] if sel else None
    d = PILOT / "runs/sweep_analysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fold0.json").write_text(json.dumps(out, indent=1))
    (d / "fold0.md").write_text(markdown(out))
    print((d / "fold0.md").read_text())


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def markdown(o):
    s = ["# Quét ngưỡng YOLOv8, fold 0 (tự sinh bởi code/sweep_analysis.py)", "",
         "## 1. Detector riêng (model_val, chọn theo F1 của chính YOLOv8, không dùng Faster R-CNN)", ""]
    for key in ("f1@0.5", "f1@0.3"):
        b, t = o[f"detector_best_{key}_model_val"], o[f"detector_best_{key}_on_test"]
        s.append(f"- Max {key}: ngưỡng {b['threshold']:.4f}, tỉ lệ ảnh bình thường có hộp {b['m_rate_nonfractured']:.3f} "
                 f"(model_val); trên test: precision@0.5 {f(t['box_precision@0.5'])}, recall@0.5 {f(t['box_recall@0.5'])}, "
                 f"ảnh gãy có hộp trúng (IoU≥0.3) {f(t['image_hit_rate@0.3'])}")
    s += ["", "## 2. Theo mức quét (test)", "",
          "| Mức | Ngưỡng outer | Ảnh BT có hộp | Hộp sai / ảnh BT | Precision hộp @0.3 / @0.5 | Recall hộp @0.3 / @0.5 | Ảnh gãy có hộp trúng @0.3 | V4 AUPRC model_val | V4 AUPRC test | AUROC test |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lv, L in o["levels"].items():
        d = L["detector_test"]
        s.append(f"| {lv[:5]} | {L['thresholds']['outer']:.4f} | {f(d['m_rate_nonfractured'])} | {f(d['false_boxes_per_normal_image'], 2)} | "
                 f"{f(d['box_precision@0.3'])} / {f(d['box_precision@0.5'])} | {f(d['box_recall@0.3'])} / {f(d['box_recall@0.5'])} | "
                 f"{f(d['image_hit_rate@0.3'])} | {f(L.get('v4_model_val', {}).get('auprc'))} | "
                 f"{f(L.get('v4_test', {}).get('auprc'))} | {f(L.get('v4_test', {}).get('auroc'))} |")
    s += ["", f"Mức chọn trên model_val: **{o['selected_on_model_val']}**", "",
          "## 3. Grounding của câu trả lời (test, ảnh gãy)", "",
          "| Mức | Hộp trúng / trượt / không hộp | Độ nhạy: trúng / trượt / không | TP có hộp trúng | Đúng có căn cứ / ảnh gãy | Placebo ΔAUPRC [CI] | Oracle vị trí ΔAUPRC [CI] |",
          "|---|---|---|---|---|---|---|"]
    for lv, L in o["levels"].items():
        g = L.get("grounding_test")
        if not g:
            continue
        iv = L.get("intervention_vs_native", {})

        def dci(key):
            a = next((v for k, v in iv.items() if k.endswith(key)), None)
            return "—" if not a else f"{a['auprc']['diff']:+.3f} [{a['auprc']['ci95'][0]:+.3f}; {a['auprc']['ci95'][1]:+.3f}]"
        s.append(f"| {lv[:5]} | {g['share_hit']:.2f} / {g['share_miss']:.2f} / {g['share_none']:.2f} | "
                 f"{f(g['sens@0.5_hit'], 2)} / {f(g['sens@0.5_miss'], 2)} / {f(g['sens@0.5_none'], 2)} | "
                 f"{f(g['tp_with_box_on_fracture'], 2)} | {f(g['grounded_tp_over_fractured'], 2)} | {dci('_placebo')} | {dci('_oracleloc')} |")
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        report()
