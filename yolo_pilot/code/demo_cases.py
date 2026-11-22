"""Demo figures of the final configuration (JF) on one fold (user 21/11: "demo lại ảnh gãy và ảnh u lành, u ác và bbox do
detector phát hiện, demo cả một số ca nhận diện chính xác tên bệnh").

    venv/bin/python -I code/demo_cases.py [fold]

Boxes: green = ground truth, red = YOLOv8 joint detector of the fold at ITS OWN operating point (max box F1@0.5 on
model_val, per class; runs/summary_final/summary.json "detector"), not the V4 proposal level.  Text = JF answers
(FracAtlas fracture probability; BTXRD tumour / malignancy probabilities and diagnosis).
Selection is a rule, not a hand pick: among the cases that meet the condition of the panel (detector box IoU >= 0.3 with
a ground-truth box, and the JF answer correct), take k cases evenly spaced in image-id order.  The counts of qualifying
cases are written next to the figures (runs/demo/fold<f>/cases.json) so the demo is not read as typical accuracy.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

PILOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
VG = Path("/Users/uyen/Downloads/BONEVQA/v4_grounded_next")
JF = "JF_v4cf_yolojoint_btxrd_r0.02"
VI = {"osteochondroma": "u xương sụn", "osteosarcoma": "sarcoma xương", "simple bone cyst": "nang xương đơn độc",
      "multiple osteochondromas": "đa u xương sụn", "giant cell tumor": "u tế bào khổng lồ", "osteofibroma": "u xơ xương",
      "synovial osteochondroma": "u sụn màng hoạt dịch", "other benign bone tumor": "u lành khác",
      "other malignant bone tumor": "u ác khác"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def spaced(xs, k):
    xs = sorted(xs, key=lambda c: c["image_id"])
    if len(xs) <= k:
        return xs
    return [xs[round(i * (len(xs) - 1) / (k - 1))] for i in range(k)] if k > 1 else [xs[len(xs) // 2]]


def main(fold=0):
    out = PILOT / f"runs/demo/fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    pre = "" if fold == 0 else f"f{fold}_"
    det = json.load(open(PILOT / "runs/summary_final/summary.json"))["detector"][str(fold)]
    thr = {0: det["fracatlas"]["own_best_threshold"], 1: det["btxrd"]["own_best_threshold"]}
    pred = {c: json.load(open(PILOT / f"runs/pred/{pre}fa_btxrd_cls{c}.json"))["boxes"] for c in (0, 1)}
    fa_gt = json.load(open(PILOT / f"datasets/fa_fold{fold}/gt.json"))
    bt_gt = json.load(open(PILOT / "datasets/btxrd_pilot/gt.json"))

    def boxes(c, iid):
        return [b for b in pred[c].get(iid, []) if b[4] >= thr[c]]

    def hit(c, iid, gtb):
        return any(iou(p, g) >= 0.3 for p in boxes(c, iid) for g in gtb)

    # ---------------------------------------------------------------- FracAtlas (test)
    fa_rows = json.load(open(VG / f"runs/yolo_ablation/fold_{fold}/{JF}_test.json"))["rows"]
    frac_pos = [r for r in fa_rows if r["y"] == 1 and not fa_gt[r["image_id"]]["hardware"]]
    frac_ok = []
    for r in frac_pos:
        g = fa_gt[r["image_id"]]
        if r["p_final"] >= 0.5 and hit(0, r["image_id"], g["boxes"]):
            frac_ok.append(dict(r, src="fa", role="test"))

    # ---------------------------------------------------------------- BTXRD (test + external)
    bt = {}
    for role in ("test", "external"):
        rows = json.load(open(VG / f"runs/joint/fold_{fold}/{JF}_btxrd_{role}.json"))["rows"]
        for k in ("tumor", "malignancy", "diagnosis"):
            for r in rows[k]:
                bt.setdefault((role, r["image_id"]), {"image_id": r["image_id"], "role": role, "src": "bt"})[k] = r
    cases = [c for c in bt.values() if "diagnosis" in c]
    for c in cases:
        g = bt_gt[c["image_id"]]
        c["hit"] = hit(1, c["image_id"], [b[:4] for b in g["boxes"]])
        c["tumor_ok"] = c["tumor"]["p_yes"] >= 0.5
        c["mal_ok"] = (c["malignancy"]["p_yes"] >= 0.5) == bool(c["malignancy"]["y"])
    test = [c for c in cases if c["role"] == "test"]
    benign_ok = [c for c in test if not c["malignancy"]["y"] and c["hit"] and c["tumor_ok"] and c["mal_ok"]]
    malig_ok = [c for c in test if c["malignancy"]["y"] and c["hit"] and c["tumor_ok"] and c["mal_ok"]]
    diag_ok = [c for c in cases if c["diagnosis"]["correct"] and c["hit"]]
    diag_pick = []
    for cls, k in (("osteochondroma", 2), ("osteosarcoma", 2), ("simple bone cyst", 2), ("multiple osteochondromas", 2)):
        diag_pick += spaced([c for c in diag_ok if c["diagnosis"]["gt"] == cls and c["role"] == "test"]
                            or [c for c in diag_ok if c["diagnosis"]["gt"] == cls], k)
    diag_miss = [c for c in test if not c["diagnosis"]["correct"] and c["hit"]
                 and c["diagnosis"]["gt"] in ("giant cell tumor", "osteofibroma", "other benign bone tumor", "simple bone cyst")]
    miss_pick = []
    for cls in ("giant cell tumor", "osteofibroma", "other benign bone tumor", "simple bone cyst"):
        miss_pick += spaced([c for c in diag_miss if c["diagnosis"]["gt"] == cls], 1)

    def draw(ax, c):
        iid = c["image_id"]
        if c["src"] == "fa":
            path = PILOT / f"datasets/fa_fold{fold}/images/test/{iid}.jpg"
            gtb, cl = fa_gt[iid]["boxes"], 0
        else:
            path = next((PILOT / f"datasets/btxrd_pilot/images/{c['role']}").glob(f"{iid}.*"))
            gtb, cl = [b[:4] for b in bt_gt[iid]["boxes"]], 1
        im = Image.open(path).convert("L")
        ax.imshow(im, cmap="gray")
        for g in gtb:
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="#22c55e", lw=1.6, ls="--"))
        for b in boxes(cl, iid):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#ef4444", lw=1.8))
            ax.text(b[0], b[1], f"{b[4]:.2f}", color="white", fontsize=7, va="bottom",
                    bbox=dict(fc="#ef4444", ec="none", pad=1))
        ax.set_xticks([]); ax.set_yticks([])
        if c["src"] == "fa":
            t = f"{iid} (FracAtlas test)\nThực tế: gãy\nJF: gãy, p = {c['p_final']:.2f}"
        else:
            g = bt_gt[iid]
            d = c["diagnosis"]
            ok = "đúng" if d["correct"] else "sai"
            t = (f"{iid} (BTXRD {c['role']})\nThực tế: {'u ác' if g['malignant'] else 'u lành'}, {VI.get(d['gt'], d['gt'])}\n"
                 f"JF: có u p = {c['tumor']['p_yes']:.2f}; ác tính p = {c['malignancy']['p_yes']:.2f}\n"
                 f"JF chẩn đoán: {VI.get(d['pred'], d['pred'])} ({ok})")
        ax.set_title(t, fontsize=8, loc="left")

    def panel(name, title, picks, ncol=4):
        if not picks:
            return None
        n = len(picks); nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.0 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, c in zip(axes.flat, picks):
            ax.axis("on"); draw(ax, c)
        fig.suptitle(title + "\nXanh lá nét đứt = vùng tổn thương thật; đỏ = hộp YOLOv8 ở ngưỡng riêng của detector "
                     f"(gãy {thr[0]:.3f}, u {thr[1]:.3f}) kèm điểm tin cậy", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = out / f"{name}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        return str(p)

    files = [
        panel("1_gay_xuong", f"Fold {fold}: ảnh gãy xương (FracAtlas test, không có dụng cụ kim loại), JF trả lời đúng và "
              f"YOLOv8 khoanh trúng: {len(frac_ok)}/{len(frac_pos)} ca", spaced(frac_ok, 4)),
        panel("2_u_lanh", f"Fold {fold}: u lành (BTXRD test), JF đúng có u + lành tính và YOLOv8 khoanh trúng: "
              f"{len(benign_ok)}/{sum(1 for c in test if not c['malignancy']['y'])} ca", spaced(benign_ok, 4)),
        panel("3_u_ac", f"Fold {fold}: u ác (BTXRD test), JF đúng có u + ác tính và YOLOv8 khoanh trúng: "
              f"{len(malig_ok)}/{sum(1 for c in test if c['malignancy']['y'])} ca", spaced(malig_ok, 4)),
        panel("4_chan_doan_dung", f"Fold {fold}: JF gọi đúng tên bệnh (BTXRD; chỉ đúng được ở 4/9 loại, xem cases.json)",
              diag_pick),
        panel("5_chan_doan_sai", f"Fold {fold}: ví dụ JF gọi sai tên bệnh dù YOLOv8 khoanh trúng (các loại hiếm)", miss_pick),
    ]
    from collections import Counter
    summ = {"fold": fold, "detector_threshold_own_best": thr,
            "fracture": {"qualifying": len(frac_ok), "fractured_no_hardware_test": len(frac_pos)},
            "benign_test": {"qualifying": len(benign_ok)}, "malignant_test": {"qualifying": len(malig_ok)},
            "diagnosis_correct_by_class": {r: dict(Counter(c["diagnosis"]["gt"] for c in cases if c["role"] == r and c["diagnosis"]["correct"]))
                                           for r in ("test", "external")},
            "diagnosis_n_by_class": {r: dict(Counter(c["diagnosis"]["gt"] for c in cases if c["role"] == r)) for r in ("test", "external")},
            "picked": {"fracture": [c["image_id"] for c in spaced(frac_ok, 4)], "benign": [c["image_id"] for c in spaced(benign_ok, 4)],
                       "malignant": [c["image_id"] for c in spaced(malig_ok, 4)],
                       "diagnosis_correct": [(c["role"], c["image_id"]) for c in diag_pick],
                       "diagnosis_wrong": [(c["role"], c["image_id"]) for c in miss_pick]},
            "files": [f for f in files if f]}
    json.dump(summ, open(out / "cases.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
