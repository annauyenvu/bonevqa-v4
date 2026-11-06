"""Metrics, paired component bootstrap and pre-declared gates for v5 suites.

Screening decisions use MODEL-VALIDATION predictions only.  Outer-test
predictions are summarised for confirmation suites and never used to pick an
arm.  Gate (declared before any v5 run):
  pass  <=>  AP(arm) - AP(reference) >= 0.005  and  AP(arm) - AP(control) >= 0.005
where reference/control are listed in GATES; arms without a control use the
reference only.  Bootstrap: 2000 draws of components within fold, seed 20261030.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
OUT = ROOT / "refactor_v2/reports/v5_studies"
DATA = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3"
CAN = "Is there a fracture in this X-ray?"
MARGIN = 0.005

# arm: (reference, own control or None).  Factorial screen: every prompt combination is
# compared with B; the paper-faithful V is additionally compared with the v4 local route.
import itertools as _it
FACTORIAL = ["".join(c) for n in range(1, 5) for c in _it.combinations("VLPT", n)]
GATES = {a: ("B", None) for a in FACTORIAL}
GATES["V4"] = ("B", None)
FACTORS = "VLPT"

def read(p):
    return json.loads(Path(p).read_text())


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < sum(y) < len(y) else float("nan")


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < sum(y) < len(y) else float("nan")


def open_recall(pred, gt):
    g = str(gt).lower().replace(".", " ").split()
    p = set(str(pred).lower().replace(".", " ").split())
    return sum(t in p for t in g) / max(1, len(g))


def metrics(records):
    from src.utils import normalize_answer

    can = [r for r in records if r["question"] == CAN]
    y = [int(str(r["gt"]).lower() == "yes") for r in can]
    s = [r["p_yes"] for r in can]
    pred = [int(v >= 0.5) for v in s]
    tp = sum(a and b for a, b in zip(y, pred)); fn = sum(a and not b for a, b in zip(y, pred))
    tn = sum((not a) and (not b) for a, b in zip(y, pred)); fp = sum((not a) and b for a, b in zip(y, pred))
    closed = [r for r in records if r["answer_type"] == "CLOSED"]
    opened = [r for r in records if r["answer_type"] != "CLOSED"]
    eq = lambda r: normalize_answer(r["pred"]) == normalize_answer(r["gt"])
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return {
        "AP": _ap(y, s), "AUROC": _auc(y, s), "sensitivity": sens, "specificity": spec, "BA": (sens + spec) / 2,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "QA_exact": float(np.mean([eq(r) for r in records])),
        "CLOSED_exact": float(np.mean([eq(r) for r in closed])) if closed else float("nan"),
        "OPEN_exact": float(np.mean([eq(r) for r in opened])) if opened else float("nan"),
        # FAVP / LLaVA-Med convention: token recall of ground truth in the generated answer
        "OPEN_token_recall": float(np.mean([open_recall(r["pred"], r["gt"]) for r in opened])) if opened else float("nan"),
        "n_images": len(can), "n_qa": len(records),
    }


def components(fold):
    man = read(DATA / "manifest.json")
    comp = {}
    for i, c in enumerate(man["components"]):
        for x in c:
            comp[x] = f"{fold}_{i}"
    return comp


def paired_bootstrap(a_rows, b_rows, fold, n=2000, seed=20261030):
    """AP(a) - AP(b) on the same canonical images, resampling components."""
    comp = components(fold)
    ca = {r["image_id"]: r for r in a_rows if r["question"] == CAN}
    cb = {r["image_id"]: r for r in b_rows if r["question"] == CAN}
    ids = sorted(set(ca) & set(cb))
    groups = {}
    for i in ids:
        groups.setdefault(comp.get(i, i), []).append(i)
    keys = sorted(groups)
    y = {i: int(str(ca[i]["gt"]).lower() == "yes") for i in ids}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = [i for k in rng.choice(len(keys), len(keys)) for i in groups[keys[k]]]
        yy = [y[i] for i in pick]
        deltas.append(_ap(yy, [ca[i]["p_yes"] for i in pick]) - _ap(yy, [cb[i]["p_yes"] for i in pick]))
    d = np.array(deltas)
    point = _ap([y[i] for i in ids], [ca[i]["p_yes"] for i in ids]) - _ap([y[i] for i in ids], [cb[i]["p_yes"] for i in ids])
    return {"delta_AP": point, "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))], "n_images": len(ids)}


def _cell_ap(rows_by_arm, ids, y, idx):
    out = {}
    for arm, recs in rows_by_arm.items():
        out[arm] = _ap([y[i] for i in idx], [recs[ids[i]] for i in idx])
    return out


def factorial_effects(rows, fold, seed, n=2000, boot_seed=20261030):
    """Main effects and two-way interactions on AP over the 2^4 cells (complete cells only)."""
    cells = {("" if a == "B" else a): rows[(fold, a, seed)]["records"] for (f, a, s_) in rows
             if f == fold and s_ == seed and (a == "B" or a in FACTORIAL)}
    if len(cells) < 16:
        return {"status": f"incomplete factorial ({len(cells)}/16 cells)"}
    can = {a: {r["image_id"]: r["p_yes"] for r in recs if r["question"] == CAN} for a, recs in cells.items()}
    ids = sorted(set.intersection(*[set(v) for v in can.values()]))
    gt = {r["image_id"]: int(str(r["gt"]).lower() == "yes") for r in cells[""] if r["question"] == CAN}
    y = [gt[i] for i in ids]
    comp = components(fold)
    groups = {}
    for k, i in enumerate(ids):
        groups.setdefault(comp.get(i, i), []).append(k)
    keys = sorted(groups)

    def contrasts(idx):
        ap = {a: _ap([y[k] for k in idx], [can[a][ids[k]] for k in idx]) for a in can}
        res = {}
        for f in FACTORS:
            on = [ap[a] for a in ap if f in a]
            off = [ap[a] for a in ap if f not in a]
            res[f] = float(np.mean(on) - np.mean(off))
        for f, g in _it.combinations(FACTORS, 2):
            both = np.mean([ap[a] for a in ap if f in a and g in a])
            fo = np.mean([ap[a] for a in ap if f in a and g not in a])
            go = np.mean([ap[a] for a in ap if g in a and f not in a])
            none = np.mean([ap[a] for a in ap if f not in a and g not in a])
            res[f + "x" + g] = float((both - go) - (fo - none))
        return res

    point = contrasts(list(range(len(ids))))
    rng = np.random.default_rng(boot_seed)
    draws = []
    for _ in range(n):
        idx = [k for g in rng.choice(len(keys), len(keys)) for k in groups[keys[g]]]
        draws.append(contrasts(idx))
    return {name: {"effect_AP": v, "ci95": [float(np.nanpercentile([d[name] for d in draws], 2.5)),
                                            float(np.nanpercentile([d[name] for d in draws], 97.5))]}
            for name, v in point.items()}


def analyze(suite):
    base = OUT / suite
    rows = {}
    for job in sorted(base.glob("fold_*/*_seed*")):
        if not (job / "completed.json").exists():
            continue
        fold = job.parent.name.split("_")[1]
        arm, seed = job.name.rsplit("_seed", 1)
        role = "model_val" if suite.startswith(("screen", "factorial")) else "test"
        recs = read(job / f"{role}_predictions.json")["records"]
        rows[(fold, arm, int(seed))] = {"role": role, "records": recs, "metrics": metrics(recs)}
    table = {f"{f}/{a}/{s}": v["metrics"] for (f, a, s), v in rows.items()}
    gates = {}
    for (f, a, s), v in rows.items():
        if a not in GATES:
            continue
        ref, ctrl = GATES[a]
        res = {"reference": ref, "control": ctrl}
        ok = True
        for name, other in (("vs_reference", ref), ("vs_control", ctrl)):
            if other is None:
                continue
            if (f, other, s) not in rows:
                res[name] = "pending"
                ok = False
                continue
            bt = paired_bootstrap(v["records"], rows[(f, other, s)]["records"], f)
            res[name] = bt
            ok = ok and bt["delta_AP"] >= MARGIN
        res["passed"] = ok
        gates[f"{f}/{a}/{s}"] = res
    effects = {}
    for (f, a, sd) in rows:
        if a == "B" and f"{f}/{sd}" not in effects:
            effects[f"{f}/{sd}"] = factorial_effects(rows, f, sd)
    report = {"suite": suite, "factorial_effects": effects, "selection_role": "model_val" if suite.startswith(("screen", "factorial")) else "test (confirmation only)",
              "margin_AP": MARGIN, "metrics": table, "gates": gates}
    (base / "analysis.json").write_text(json.dumps(report, indent=2))
    lines = [f"# {suite}", "", "| job | AP | AUROC | Sens | Spec | BA | QA | OPEN exact | OPEN recall |", "|---|---|---|---|---|---|---|---|---|"]
    for k, m in sorted(table.items()):
        lines.append(f"| {k} | {m['AP']:.4f} | {m['AUROC']:.4f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['BA']:.3f} | "
                     f"{m['QA_exact']:.4f} | {m['OPEN_exact']:.4f} | {m['OPEN_token_recall']:.4f} |")
    lines += ["", "| arm | ref | ΔAP vs ref [CI] | control | ΔAP vs control [CI] | gate |", "|---|---|---|---|---|---|"]
    for k, g in sorted(gates.items()):
        def fmt(x):
            return x if isinstance(x, str) else f"{x['delta_AP']:+.4f} [{x['ci95'][0]:+.4f}; {x['ci95'][1]:+.4f}]"
        lines.append(f"| {k} | {g['reference']} | {fmt(g.get('vs_reference', '-'))} | {g['control'] or '-'} | "
                     f"{fmt(g['vs_control']) if 'vs_control' in g else '-'} | {'PASS' if g['passed'] else 'no'} |")
    for key, eff in effects.items():
        lines += ["", f"Factorial effects on AP, fold/seed {key} (mean over 2^4 cells; interaction = difference-in-differences)", ""]
        if "status" in eff:
            lines.append(eff["status"])
            continue
        lines += ["| term | effect | 95% CI |", "|---|---|---|"]
        for name, e in eff.items():
            lines.append(f"| {name} | {e['effect_AP']:+.4f} | [{e['ci95'][0]:+.4f}; {e['ci95'][1]:+.4f}] |")
    (base / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report
