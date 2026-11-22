"""Figures of the IEEE Access manuscript (English labels). Every number is read from stored result files.

Run: yolo_pilot/venv/bin/python -I analysis/figs.py
Writes sources/img/fig1_framework.png, fig3_regime.png, fig4_ladder.png, fig5_forest.png, fig6_cases.png.
Palette: validated categorical slots 1-5 (dataviz reference palette, light mode); markers differ by shape too.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "sources/img"; OUT.mkdir(parents=True, exist_ok=True)
RES = PAPER / "results"
F0 = ROOT / "v4_grounded_next/runs/yolo_ablation/fold_0"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTE, GRID = "#1f2933", "#6b7280", "#e5e7eb"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
OFF, ON, ACC, BAD = "#e8eef5", "#eef6ee", "#1d5c96", "#b23b3b"
GAP = 0.004
SCALE = 0.86


def box(ax, x, y, w, h, text, fc, ec=INK, fs=6.4, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=0.8, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs * SCALE, color=INK, weight="bold" if bold else "normal",
            linespacing=1.25, zorder=3)


def path(ax, pts, color=INK, ls="-", lw=0.85, head=True):
    pts = [np.array(p, float) for p in pts]
    d = pts[-1] - pts[-2]; u = d / np.linalg.norm(d); end = pts[-1] - u * GAP
    line = pts[:-1] + [end - u * (0.012 if head else 0)]
    ax.plot([p[0] for p in line], [p[1] for p in line], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    if head:
        ax.annotate("", xy=end, xytext=end - u * 0.013,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=4)


def dot(ax, x, y):
    ax.plot([x], [y], "o", ms=2.2, color=INK, zorder=4)


# ------------------------------------------------------------------ Fig. 1 framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.16, 4.3)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.535), 0.99, 0.455, boxstyle="round,pad=0,rounding_size=0.015", fc="#f7f9fc", ec=MUTE, lw=0.6, ls="--", zorder=0))
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.495, boxstyle="round,pad=0,rounding_size=0.015", fc="#f8fbf7", ec=MUTE, lw=0.6, ls="--", zorder=0))
    ax.text(0.015, 0.963, "OFFLINE STAGE (per outer fold; lesion annotations allowed)", fontsize=7.6, weight="bold", color=ACC, va="center")
    ax.text(0.015, 0.478, "ONLINE STAGE (deployment; inputs = radiograph + question only)", fontsize=7.6, weight="bold", color="#2f6b2f", va="center")
    # offline
    box(ax, 0.02, 0.80, 0.135, 0.12, "FracAtlas-VQA\n4,013 images, 5 folds\n(component-grouped)", OFF, fs=5.4)
    box(ax, 0.02, 0.64, 0.135, 0.12, "BTXRD-VQA\ncenter 1: train/val/test\ncenters 2–3: external", OFF, fs=5.4)
    box(ax, 0.19, 0.72, 0.13, 0.12, "Two-class YOLOv8\n(fracture, lesion)\non TRAIN", OFF, fs=5.6)
    box(ax, 0.355, 0.80, 0.15, 0.12, "Cross-fitting:\n3 inner detectors\n→ TRAIN proposals", OFF, fs=5.5)
    box(ax, 0.355, 0.64, 0.15, 0.12, "Levels on model-val:\nτ_det (display),\nτ(r): r = 0.02 / 0.10", OFF, fs=5.5)
    box(ax, 0.54, 0.72, 0.15, 0.12, "Joint VQA training\nV4 module, FracAtlas\n+ BTXRD, 2 epochs", OFF, fs=5.5)
    box(ax, 0.725, 0.72, 0.13, 0.12, "Confidence layer\nfit on calibration\n/ BTXRD val", OFF, fs=5.5)
    box(ax, 0.875, 0.72, 0.105, 0.12, "Locked\nartifacts +\nSHA-256", OFF, fs=5.5)
    box(ax, 0.19, 0.565, 0.665, 0.05, "lesion boxes used only offline: detector training · leakage audit · oracle / placebo boxes · box scoring", "#fbeeee", ec=BAD, fs=5.3)
    path(ax, [(0.155, 0.86), (0.1725, 0.86), (0.1725, 0.80), (0.19, 0.80)])
    path(ax, [(0.155, 0.70), (0.1725, 0.70), (0.1725, 0.76), (0.19, 0.76)])
    path(ax, [(0.32, 0.78), (0.3375, 0.78)], head=False); dot(ax, 0.3375, 0.78)
    path(ax, [(0.3375, 0.78), (0.3375, 0.86), (0.355, 0.86)])
    path(ax, [(0.3375, 0.78), (0.3375, 0.70), (0.355, 0.70)])
    path(ax, [(0.505, 0.86), (0.5225, 0.86), (0.5225, 0.80), (0.54, 0.80)])
    path(ax, [(0.505, 0.70), (0.5225, 0.70), (0.5225, 0.76), (0.54, 0.76)])
    path(ax, [(0.69, 0.78), (0.725, 0.78)]); path(ax, [(0.855, 0.78), (0.875, 0.78)])
    path(ax, [(0.9275, 0.72), (0.9275, 0.41)], color=MUTE, ls=(0, (2, 2)), lw=0.8)
    ax.text(0.934, 0.57, "deploy", fontsize=6, color=MUTE, rotation=90, va="center")
    # online
    box(ax, 0.02, 0.17, 0.10, 0.20, "Radiograph I\n+\nquestion q", ON, bold=True, fs=5.6)
    box(ax, 0.15, 0.31, 0.12, 0.09, "YOLOv8:\nboxes, scores", ON, fs=5.5)
    box(ax, 0.15, 0.06, 0.12, 0.09, "BiomedCLIP ViT\n+ PubMedBERT", ON, fs=5.5)
    box(ax, 0.30, 0.39, 0.15, 0.065, "display boxes ≥ τ_det", "#ffffff", ec=MUTE, fs=5.3, ls="--")
    box(ax, 0.30, 0.24, 0.15, 0.10, "router: class by question;\ntop-3 boxes ≥ τ_c(r)", ON, fs=5.2)
    box(ax, 0.48, 0.24, 0.13, 0.10, "crop, shared ViT,\n2×2 pool → C (≤ 12)", ON, fs=5.3)
    box(ax, 0.30, 0.06, 0.15, 0.09, "global tokens G,\nquestion tokens Q", ON, fs=5.4)
    box(ax, 0.64, 0.10, 0.11, 0.24, "V4 module\nquestion-\nconditioned\nMHA + gated\nresidual m·γ·Δ", "#e3effa", ec=ACC, fs=5.3)
    box(ax, 0.775, 0.27, 0.10, 0.07, "fracture head", ON, fs=5.3)
    box(ax, 0.775, 0.18, 0.10, 0.07, "closed head", ON, fs=5.3)
    box(ax, 0.775, 0.06, 0.10, 0.10, "Qwen2.5-0.5B:\nopen answer,\ndiagnosis score", ON, fs=4.9)
    box(ax, 0.895, 0.06, 0.09, 0.28, "Confidence\nP_ok from\n|p − 0.5|\nand s_max:\nanswer or\ndefer", "#fff7e6", ec="#b7791f", fs=5.2)
    path(ax, [(0.12, 0.27), (0.135, 0.27)], head=False); dot(ax, 0.135, 0.27)
    path(ax, [(0.135, 0.27), (0.135, 0.355), (0.15, 0.355)])
    path(ax, [(0.135, 0.27), (0.135, 0.105), (0.15, 0.105)])
    path(ax, [(0.27, 0.355), (0.285, 0.355)], head=False); dot(ax, 0.285, 0.355)
    path(ax, [(0.285, 0.355), (0.285, 0.4225), (0.30, 0.4225)])
    path(ax, [(0.285, 0.355), (0.285, 0.29), (0.30, 0.29)])
    path(ax, [(0.45, 0.29), (0.48, 0.29)]); path(ax, [(0.61, 0.29), (0.64, 0.29)])
    path(ax, [(0.27, 0.105), (0.30, 0.105)]); path(ax, [(0.45, 0.105), (0.64, 0.105)])
    path(ax, [(0.75, 0.22), (0.7625, 0.22)], head=False); dot(ax, 0.7625, 0.22)
    path(ax, [(0.7625, 0.11), (0.7625, 0.305)], head=False)
    for y in (0.305, 0.215, 0.11):
        path(ax, [(0.7625, y), (0.775, y)])
    for y in (0.305, 0.215, 0.11):
        path(ax, [(0.875, y), (0.895, y)])
    path(ax, [(0.45, 0.26), (0.465, 0.26), (0.465, 0.03), (0.94, 0.03), (0.94, 0.06)], color=MUTE, lw=0.6)
    ax.text(0.70, 0.036, "highest box score s_max of the routed class", fontsize=4.8, color=MUTE, ha="center", va="bottom", style="italic")
    fig.savefig(OUT / "fig1_framework.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig. 3 regime
def fig_regime():
    fac = json.load(open(RES / "regime_factorial_fold0.json"))["arms"]
    def auprc(name):
        return json.load(open(F0 / f"{name}_test.json"))["metrics"]["auprc"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.32))
    dets = ["Faster R-CNN", "YOLOv8 Y1", "YOLOv8 Y2"]
    cells = {"No retraining (swap)": ["A", "E", "E2"], "Retrained, in-sample": ["A", "C", "C2"], "Retrained, cross-fitted": ["B", "D", "D2"]}
    x = np.arange(3); w = 0.27
    for k, (lab, keys) in enumerate(cells.items()):
        v = [fac[c]["auprc"] for c in keys]
        a.plot(x + (k - 1) * w, v, ls="none", marker=["o", "s", "^"][k], ms=6, color=[S1, S2, S3][k], label=lab, zorder=3)
        for xi, vi in zip(x + (k - 1) * w, v):
            a.text(xi + (-0.06, 0.06, 0.0)[k], vi + 0.0035, f"{vi:.3f}", ha=("right", "left", "center")[k], va="bottom", fontsize=5.2, color=INK)
    for xi in x:
        a.axvspan(xi - 0.42, xi + 0.42, color="#f3f4f6", zorder=0, lw=0)
    a.set_xticks(x, dets); a.set_xlim(-0.5, 2.6); a.set_ylim(0.72, 0.83); a.set_ylabel("Fracture AUPRC (fold 0 test)")
    a.grid(axis="y", color=GRID, lw=0.5, zorder=0); a.legend(frameon=False, fontsize=6, loc="upper left")
    a.set_title("(a) Detector × training regime (r = 0.284)", fontsize=7.6, loc="left")
    levels = [("0.02", "D2_sweep_r0.02"), ("0.15", "D2_sweep_r0.15"), ("0.284", "D2_v4cf_yolojoint"), ("none", "D2_sweep_rmax")]
    xs = np.arange(len(levels))
    for lab, suf, col, mk in (("As trained", "", S1, "o"), ("Placebo boxes", "_placebo", S2, "s"), ("Oracle boxes", "_oracleloc", S3, "^")):
        vals = [auprc(n + suf) for _, n in levels]
        b.plot(xs, vals, color=col, marker=mk, ms=4.5, lw=1.4, label=lab, zorder=3)
    b.set_xticks(xs, [f"r = {l}" if l != "none" else "no threshold" for l, _ in levels]); b.set_ylim(0.70, 0.88)
    b.set_ylabel("Fracture AUPRC (fold 0 test)"); b.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    b.legend(frameon=False, fontsize=6, loc="upper right")
    b.set_title("(b) Y2 cross-fitted, retrained at each proposal level", fontsize=7.6, loc="left")
    fig.savefig(OUT / "fig3_regime.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig. 4 ladder
def fig_ladder():
    cm = json.load(open(RES / "compare_methods.json"))["arms"]
    arms = [("V4", "L_A", S1, "o"), ("V4-XF", "L_B", S2, "s"), ("Y-XF", "L_D2", S3, "^"), ("Y-XF-J", "L_J", S4, "D"), ("JF", "JF", S5, "*")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for lab, key, col, mk in arms:
        v = cm[key]["fold_ap"]
        ax.plot(range(5), v, color=col, marker=mk, ms=5 if mk != "*" else 8, lw=1.3 if lab != "JF" else 2.0, label=lab, zorder=3 if lab != "JF" else 4)
    ax.text(4.12, cm["JF"]["fold_ap"][4], "JF", fontsize=6.5, va="center", color=INK)
    ax.set_xticks(range(5), [f"Fold {f}" for f in range(5)]); ax.set_xlim(-0.2, 4.45); ax.set_ylim(0.68, 0.90)
    ax.set_ylabel("Fracture AUPRC (test)"); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper left")
    fig.savefig(OUT / "fig4_ladder.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig. 5 forest
def fig_forest():
    cm = json.load(open(RES / "compare_methods.json"))["JF_minus"]
    rows = [("BiomedCLIP + logistic regression", "BiomedCLIP-LR"), ("MobileNetV3-Large", "MobileNetV3"), ("Base (no region prompt)", "Base"),
            ("FAVP-style visual prompt", "FAVP-style"), ("LaPA-style latent prompt", "LaPA-style"), ("Lens-style training view", "Lens-style"),
            ("V4", "V4"), ("V4-XF", "L_B"), ("Y-XF", "L_D2"), ("Y-XF-J", "L_J")]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, (lab, k) in enumerate(rows[::-1]):
        d = cm[k]["pooled_ap"]
        ax.plot(d["ci95"], [i, i], color=S1, lw=1.4, solid_capstyle="round")
        ax.plot([d["diff"]], [i], "o", color=S1, ms=4)
        ax.text(0.245, i, f"+{d['diff']:.3f}", va="center", fontsize=5.8, color=INK)
    ax.axvline(0, color=MUTE, lw=0.7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=6.2)
    ax.set_xlim(-0.01, 0.275); ax.set_xlabel("JF − method, pooled fracture AP (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    fig.savefig(OUT / "fig5_forest.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig. 6 cases
def fig_cases():
    import importlib.util
    pilot = ROOT / "yolo_pilot"
    cases = json.load(open(pilot / "runs/demo/fold0/cases.json"))
    det = json.load(open(pilot / "runs/summary_final/summary.json"))["detector"]["0"]
    thr = {0: det["fracatlas"]["own_best_threshold"], 1: det["btxrd"]["own_best_threshold"]}
    pred = {c: json.load(open(pilot / f"runs/pred/fa_btxrd_cls{c}.json"))["boxes"] for c in (0, 1)}
    fa_gt = json.load(open(pilot / "datasets/fa_fold0/gt.json")); bt_gt = json.load(open(pilot / "datasets/btxrd_pilot/gt.json"))
    JF = "JF_v4cf_yolojoint_btxrd_r0.02"
    fa_rows = {r["image_id"]: r for r in json.load(open(F0 / f"{JF}_test.json"))["rows"]}
    bt = {}
    for role in ("test", "external"):
        rows = json.load(open(ROOT / f"v4_grounded_next/runs/joint/fold_0/{JF}_btxrd_{role}.json"))["rows"]
        for k in ("tumor", "malignancy", "diagnosis"):
            for r in rows[k]:
                bt.setdefault((role, r["image_id"]), {})[k] = r
    picks = [("fa", "test", i) for i in cases["picked"]["fracture"][:2]] + [("bt", "test", cases["picked"]["benign"][0]),
             ("bt", "test", cases["picked"]["malignant"][0])]
    used = {i for _, _, i in picks}
    dx = [(r, i) for r, i in cases["picked"]["diagnosis_correct"] if i not in used]
    picks += [("bt", r, i) for r, i in (dx[0], dx[2], dx[3])]  # osteochondroma, osteosarcoma, external case
    picks += [("bt", r, i) for r, i in cases["picked"]["diagnosis_wrong"][:1]]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.6))
    for n, (ax, (src, role, iid)) in enumerate(zip(axes.flat, picks)):
        if src == "fa":
            img = pilot / f"datasets/fa_fold0/images/test/{iid}.jpg"; gtb, cl = fa_gt[iid]["boxes"], 0
        else:
            img = next((pilot / f"datasets/btxrd_pilot/images/{role}").glob(f"{iid}.*")); gtb, cl = [b[:4] for b in bt_gt[iid]["boxes"]], 1
        im = Image.open(img).convert("L"); ax.imshow(im, cmap="gray")
        lw = max(im.size) / 600
        for g in gtb:
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="#22c55e", lw=1.0, ls="--"))
        for bb in [b for b in pred[cl].get(iid, []) if b[4] >= thr[cl]]:
            ax.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="#ef4444", lw=1.1))
            ax.text(bb[0], bb[1], f"{bb[4]:.2f}", color="white", fontsize=5, va="bottom", bbox=dict(fc="#ef4444", ec="none", pad=0.6))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if src == "fa":
            t = f"({'abcdefgh'[n]}) FracAtlas {iid}\nTruth: fracture\nJF: fracture, p = {fa_rows[iid]['p_final']:.2f}"
        else:
            c = bt[(role, iid)]; g = bt_gt[iid]; d = c["diagnosis"]
            t = (f"({'abcdefgh'[n]}) BTXRD {role} {iid}\nTruth: {'malignant' if g['malignant'] else 'benign'}, {d['gt']}\n"
                 f"JF: tumor {c['tumor']['p_yes']:.2f}, malig. {c['malignancy']['p_yes']:.2f}\n"
                 f"Dx: {d['pred']} ({'correct' if d['correct'] else 'wrong'})")
        ax.set_title(t, fontsize=6.0, loc="left", color=INK)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.6)
    fig.savefig(OUT / "fig6_cases.png", dpi=250, bbox_inches="tight"); plt.close(fig)
    return picks


if __name__ == "__main__":
    fig_framework(); fig_regime(); fig_ladder(); fig_forest(); print(fig_cases())
