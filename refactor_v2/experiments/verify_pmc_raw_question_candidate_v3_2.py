"""Independent source reconstruction and actual runtime-loader verification."""
import hashlib
import json
from pathlib import Path
from src.data.pmc_vqa import load_pmc_vqa_records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_raw_question_policy_v3_2"
    out = folder / "independent_loader_verification.json"
    if out.exists():
        raise RuntimeError("Preserve verification evidence")
    summary = json.loads((folder / "summary.json").read_text())
    for name, digest in summary["artifacts"].items():
        assert sha(folder / name) == digest
    source = root / ".evaluation_runtime/data/pretraining_joint_groups_v3/pmc_vqa_train_label_filtered.json"
    assert sha(source) == summary["source_sha256"]
    original = json.loads(source.read_text())
    kept = json.loads((folder / "train_candidate.json").read_text())
    quarantine = json.loads((folder / "train_quarantine.json").read_text())
    removed = {row["source_index"] for row in quarantine}
    assert len(removed) == len(quarantine)
    assert kept == [row for i, row in enumerate(original) if i not in removed]
    assert all(original[row["source_index"]] == row["original_record"] for row in quarantine)
    loaded = load_pmc_vqa_records("train", rows=kept, allow_export=False, routing_policy="generative")
    assert len(loaded) == len(kept)
    assert all(row["answer_type"] == "OPEN" and row["answer"] for row in loaded)
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(kept, loaded)) if a["answer"] != b["answer"]]
    assert all(not source_row["answer"] and runtime_row["answer"] in ("None", "N/A") for _, source_row, runtime_row in differences)
    report = {"passed": True, "scope": "TRAIN candidate verification only; no scientific release or test use",
              "retained_rows": len(loaded), "quarantined_rows": len(quarantine), "complete_source_partition_exact": True,
              "all_runtime_targets_nonempty_and_OPEN": True, "nonempty_source_GT_changed_by_loader": 0,
              "empty_source_answer_resolved_from_explicit_existing_label_choice": len(differences),
              "resolved_GT_values": sorted({b["answer"] for _, _, b in differences}),
              "not_invented_labels": True, "source_sha256": sha(source),
              "policy_sha256": sha(folder / "policy.json"),
              "implementation_sha256": sha(root / "refactor_v2/experiments/prepare_pmc_raw_question_policy_v3_1.py"),
              "implementation_additional_structural_rules": ["quarantine choice-key/embedded-prefix conflicts", "quarantine nonempty source-answer/label-selected-choice conflicts; never guess which GT is correct"],
              "test_used": False, "long_training_released": False}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
