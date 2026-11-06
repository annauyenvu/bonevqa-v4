"""Versioned TRAIN-derived raw-question eligibility candidate; no source edits."""
import hashlib
import json
import re
from pathlib import Path

POLICY = {
    "version": "pmc_raw_question_eligibility_v3_2_candidate",
    "production_input_unchanged": "original image + raw question; no choices/caption/annotation/GT input",
    "scope": "conservative structural eligibility, NOT clinical label adjudication or proof of visual answerability",
    "patterns": {
        "option_dependent_answer": r"\b(?:both\s+[a-d]\s+and\s+[a-d]|(?:all|none)\s+of\s+(?:the\s+)?above|neither\s+[a-d]\s+nor\s+[a-d])\b",
        "option_dependent_question": r"\b(?:which|what)\s+of\s+the\s+following\b|\b(?:above\s+options|following\s+options)\b",
        "caption_or_text_dependent_question": r"\bcaption\b|\baccording\s+to\s+(?:the\s+)?(?:text|article|report)\b"
    },
    "action": "quarantine flagged QA from raw-question candidate, preserve original row and GT unchanged",
    "retain": "single panel labels A/(a), direct image-panel references and all other existing GT unless explicitly flagged",
    "limitations": ["heuristics can over-exclude or miss context-dependent QA", "figure A/B combinations may be ambiguous: quarantine is not a claim of label invalidity",
                    "no automatic clinical contradiction relabeling", "cannot conclude that all retained QA are clinically valid"],
    "scientific_release": False
}


def strip_prefix(value):
    return re.sub(r"^[A-D]:", "", str(value or "").strip(), flags=re.I).strip()


def effective_GT(row):
    label = str(row.get("answer_label", "")).upper().strip(" .:()")
    choice = (row.get("choices") or {}).get(label)
    return strip_prefix(choice) if choice and strip_prefix(choice) else strip_prefix(row.get("answer"))


def classify(row):
    reasons = [name for name, pattern in POLICY["patterns"].items()
               if re.search(pattern, effective_GT(row) if name == "option_dependent_answer" else str(row["question"]), re.I)]
    for key, text in (row.get("choices") or {}).items():
        match = re.match(r"^([A-D]):", str(text).strip(), re.I)
        if match and match.group(1).upper() != key.upper():
            reasons.append("choice_key_vs_embedded_prefix_conflict")
            break
    direct = strip_prefix(row.get("answer"))
    if direct and " ".join(direct.casefold().split()) != " ".join(effective_GT(row).casefold().split()):
        reasons.append("nonempty_source_answer_vs_label_choice_conflict")
    return reasons


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Structural tests defined before processing any evaluation partition.
    tests = [({"question": "What is shown?", "answer": "Both A and B"}, ["option_dependent_answer"]),
             ({"question": "Which of the following is shown?", "answer": "MRI"}, ["option_dependent_question"]),
             ({"question": "What is mentioned in the caption?", "answer": "Elbow"}, ["caption_or_text_dependent_question"]),
             ({"question": "Which image is CT?", "answer": "(a)"}, []),
             ({"question": "What does panel A show?", "answer": "Elbow"}, []),
             ({"question": "What is shown?", "answer": "Both kidneys"}, []),
             ({"question": "What is shown?", "answer": "B1", "answer_label": "D", "choices": {"D": "C:B2"}},
              ["choice_key_vs_embedded_prefix_conflict", "nonempty_source_answer_vs_label_choice_conflict"]),
             ({"question": "What is shown?", "answer": "", "answer_label": "A", "choices": {"A": "A:None"}}, [])]
    for row, expected in tests:
        assert classify(row) == expected, row
    root = Path.cwd()
    output = root / "refactor_v2/reports/pmc_raw_question_policy_v3_2"
    if output.exists():
        raise RuntimeError("Preserve existing policy candidate")
    cfg = json.loads((root / "refactor_v2/reports/pmc_alignment_execution_v3/released_config.json").read_text())
    source = Path(cfg["data"]["loader_kwargs"]["sealed_records_by_split"]["train"])
    parent_seal = Path(cfg["data"]["loader_kwargs"]["seal_path"])
    seal = json.loads(parent_seal.read_text())
    assert seal["artifacts"][source.relative_to(root).as_posix()] == sha(source)
    rows = json.loads(source.read_text())
    keep, quarantine, retained_indices = [], [], []
    for index, row in enumerate(rows):
        reasons = classify(row)
        if reasons:
            quarantine.append({"source_index": index, "reasons": reasons, "original_record": row})
        else:
            keep.append(row)
            retained_indices.append(index)
    assert len(keep) + len(quarantine) == len(rows)
    assert keep == [rows[index] for index in retained_indices]
    output.mkdir()
    for name, data in (("policy.json", POLICY), ("train_candidate.json", keep), ("train_quarantine.json", quarantine)):
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    summary = {"status": "TRAIN_candidate_only_not_scientific_release", "source_sha256": sha(source),
               "parent_seal_sha256": sha(parent_seal), "source_rows": len(rows), "retained": len(keep), "quarantined": len(quarantine),
               "counts_nonexclusive": {name: sum(name in row["reasons"] for row in quarantine) for name in
                                       (*POLICY["patterns"], "choice_key_vs_embedded_prefix_conflict", "nonempty_source_answer_vs_label_choice_conflict")},
               "structural_tests_passed": len(tests), "retained_GT_and_rows_exactly_unchanged": True,
               "original_manifests_modified": False, "validation_or_test_read": False,
               "artifacts": {name: sha(output / name) for name in ("policy.json", "train_candidate.json", "train_quarantine.json")},
               "next": "Independent loader/provenance verification; apply frozen eligibility only to development validation for a separately sealed bounded matched pilot. No test tuning or long training release."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
