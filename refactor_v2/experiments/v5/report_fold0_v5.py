"""Exploratory fold-0 tables for the manuscript: python report_fold0_v5.py model_val|test.

Reduced scope (2026-10-31): B, T, P, PT, V4, V, L (MedSAM only), VLPT, seed 42.  The arm decision is made on
model_val and written to SELECTION_fold0.json; the test role refuses to run before that file exists.
"""
import json
import sys

from experiments.v5.analyze_v5 import MARGIN, OUT, metrics, paired_bootstrap, read

ARMS = ["B", "T", "P", "PT", "V4", "V", "L", "VLPT"]
BASE = OUT / "factorial_fold0"
role = sys.argv[1]
selection = BASE / "SELECTION_fold0.json"
assert role in ("model_val", "test")
assert role == "model_val" or selection.exists(), "freeze the model_val decision first"
rows = {}
for a in ARMS:
    job = BASE / f"fold_0/{a}_seed42"
    assert (job / "completed.json").exists() and read(job / "replay.json")["passed"], a
    rows[a] = read(job / f"{role}_predictions.json")["records"]
    assert [(r["image_id"], r["question"]) for r in rows[a]] == [(r["image_id"], r["question"]) for r in rows["B"]], a
res = {a: metrics(r) for a, r in rows.items()}
vs_b = {a: paired_bootstrap(rows[a], rows["B"], "0") for a in ARMS[1:]}
vs_v4 = {a: paired_bootstrap(rows[a], rows["V4"], "0") for a in ("V", "L", "VLPT")}
ci = lambda x: f"{x['delta_AP']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}]"
lines = [f"# Fold 0, seed 42, role {role} (exploratory)", "",
         f"n_images {res['B']['n_images']}, n_qa {res['B']['n_qa']}; threshold 0.5; gate: dAP vs B >= {MARGIN}; no multiplicity correction.", "",
         "| Arm | AP | AUROC | Sens | Spec | BA | QA | CLOSED | OPEN | dAP vs B [95% CI] | dAP vs V4 [95% CI] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for a, m in res.items():
    lines.append(f"| {a} | {m['AP']:.3f} | {m['AUROC']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['BA']:.3f} | {m['QA_exact']:.3f} | "
                 f"{m['CLOSED_exact']:.3f} | {m['OPEN_exact']:.3f} | {ci(vs_b[a]) if a in vs_b else '-'} | {ci(vs_v4[a]) if a in vs_v4 else '-'} |")
(BASE / f"FOLD0_{role}.json").write_text(json.dumps({"metrics": res, "vs_B": vs_b, "vs_V4": vs_v4}, indent=2))
(BASE / f"FOLD0_{role}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
