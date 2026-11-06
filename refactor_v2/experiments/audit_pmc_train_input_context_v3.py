"""TRAIN-only structural flags for raw-question PMC input; never rewrites GT."""
import hashlib
import json
import re
from pathlib import Path


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_alignment_execution_v3"
    output = folder / "train_input_context_audit.json"
    if output.exists():
        raise RuntimeError("Preserve previous input audit")
    cfg = json.loads((folder / "released_config.json").read_text())
    path = Path(cfg["data"]["loader_kwargs"]["sealed_records_by_split"]["train"])
    rows = json.loads(path.read_text())
    flags, choice_count = [], 0
    for index, row in enumerate(rows):
        question, answer = str(row["question"]), str(row["answer"])
        reasons = []
        if row.get("choices"):
            choice_count += 1
        if re.search(r"\b(?:both\s+[a-d]\s+and\s+[a-d]|(?:all|none)\s+of\s+(?:the\s+)?above|neither\s+[a-d]\s+nor\s+[a-d])\b", answer, re.I):
            reasons.append("answer_explicitly_references_option_or_panel_set")
        if re.search(r"\b(?:which|what)\s+of\s+the\s+following\b|\b(?:above\s+options|following\s+options)\b", question, re.I):
            reasons.append("question_explicitly_references_options")
        if reasons:
            flags.append({"index": index, "image": row.get("image"), "question": question, "original_GT": answer,
                          "source_choices_present": bool(row.get("choices")), "reasons": reasons})
    counts = {reason: sum(reason in row["reasons"] for row in flags) for reason in
              ("answer_explicitly_references_option_or_panel_set", "question_explicitly_references_options")}
    report = {"scope": "PMC sealed TRAIN only; structural input-context audit, no medical relabeling",
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "train_rows": len(rows),
              "source_rows_with_choices": choice_count, "runtime_input": "original image + raw question; choices not supplied",
              "flagged_rows": len(flags), "flag_counts_nonexclusive": counts, "records": flags,
              "flags_are_not_adjudicated_invalid_labels": True, "gt_modified": False, "test_used": False,
              "next": "Design TRAIN-derived explicit option-context eligibility policy and versioned adapter/seal; distinguish figure-panel references from missing multiple-choice options. Do not append choices ad hoc, silently drop data, or overwrite published scores.",
              "long_training_released": False}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
