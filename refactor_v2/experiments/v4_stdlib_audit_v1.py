"""Independent audit: stdlib only, reads the exported answers, recounts QA, contradictions and AP with its own code."""
import json, re, string
d = json.load(open("reports/v4_detector_ext_v1/final_test_answers.json"))
def norm(t): return " ".join(t.lower().translate(str.maketrans("", "", string.punctuation)).split())
rows = [r for v in d.values() for r in v]; can = sorted((r for r in rows if r["s"] is not None), key=lambda r: -r["s"])
tp = 0; prec = []
for i, r in enumerate(can, 1):
    if r["gt"] == "yes": tp += 1; prec.append(tp / i)
by = {}
for r in rows: by.setdefault(r["image_id"], []).append(r)
bad = 0
for rs in by.values():
    yes = next((r["pred"] for r in rs if r["s"] is not None), None)
    if yes is None: continue
    for r in rs:
        q, p = r["question"].lower(), r["pred"].lower()
        if q == "describe the finding.": bad += p.startswith("no fracture") == (yes == "yes")
        if q.startswith("what abnormality"): bad += (p == "no abnormality") == (yes == "yes")
        if q.startswith("how many fracture"): bad += (p == "zero") == (yes == "yes")
        if yes == "no" and re.search(r"fractured\?$|multiple fractures\?$", q): bad += p == "yes"
print("rows", len(rows), "images", len(by), "canonical", len(can), "positives", tp, "pooled AP", round(sum(prec) / tp, 6), "QA", round(sum(norm(r["pred"]) == norm(r["gt"]) for r in rows) / len(rows), 6), "contradictions", bad)
