"""Always-negative/positive fracture controls; never fit a validation-dependent rule."""
import copy
import hashlib
import json
from pathlib import Path
from src.utils import compute_metrics


def main():
    folder = Path("refactor_v2/reports/fracatlas_baseline_pilot_v3")
    output = folder / "degenerate_fracture_controls.json"
    if output.exists():
        raise RuntimeError("Do not overwrite existing controls")
    source = folder / "question_frequency_control.json"
    original = json.loads(source.read_text())
    result = {"scope": "Deterministic degeneracy controls on already-inspected142-QA development; no fitted rule or threshold",
              "source_control_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "other_questions": "same original TRAIN-only per-question majority", "conditions": {},
              "warning": "An always-no model may score higher overall on imbalanced validation while missing every fracture; report sensitivity/specificity and discrimination, never pick a control as a learned solution"}
    for label, score in (("no", 0.0), ("yes", 1.0)):
        records = copy.deepcopy(original["records"])
        for row in records:
            if row["question"] == "Is there a fracture in this X-ray?":
                row.update(pred=label, p_yes=score)
        result["conditions"]["always_" + label + "_fracture"] = {"metrics": compute_metrics(records, fracture_threshold=0.5), "records": records}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["metrics"] for key, value in result["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
