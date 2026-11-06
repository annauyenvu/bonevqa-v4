"""Independent full model-validation predictions/error audit; never uses PMC test."""
import hashlib
import json
import os
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data import build_dataset, collate_fn
from src.evaluate import evaluate_model
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest
from src.train import build_split
from src.utils import normalize_answer, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_alignment_execution_v3"
    progress = json.loads((folder / "progress.json").read_text())
    if progress["status"] != "training_complete_pending_independent_checkpoint_evaluation":
        raise RuntimeError("PMC training still active/failed; preserve it, do not run competing evaluation")
    output = folder / "independent_model_validation_predictions.json"
    errors_path = folder / "model_validation_error_audit.json"
    progress_path = folder / "independent_evaluation_progress.json"
    if output.exists() or errors_path.exists() or progress_path.exists():
        raise RuntimeError("Existing evaluation must not be overwritten")
    checkpoint = Path(progress["checkpoint_dir"]) / "best.pt"
    if sha(checkpoint) != progress["best_checkpoint_sha256"]:
        raise RuntimeError("Best checkpoint hash mismatch")
    status = {"scope": "PMC model-validation independent reload/generation audit; no test",
        "status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha(checkpoint), "planned_qa": 1500}
    progress_path.write_text(json.dumps(status, indent=2) + "\n")
    cfg = json.loads((checkpoint.parent / "config.json").read_text())
    data = build_split(build_dataset, "pmc_vqa", "model_val", None, True, cfg["data"]["loader_kwargs"])
    if len(data) != 1500:
        raise RuntimeError("Full model-validation cohort must contain1500 QA")
    saved_manifest = json.loads((checkpoint.parent / "data_manifest.json").read_text())["val"]
    if manifest(data) != saved_manifest:
        raise RuntimeError("Model-validation loader manifest differs from training")
    set_seed(42)
    model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not all(bool(torch.isfinite(value).all()) for value in payload["state"].values()):
        raise RuntimeError("Nonfinite saved weights")
    loader = DataLoader(data, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    metrics, rows = evaluate_model(model, loader, len(data), torch.device("cpu"))
    if len(rows) != len(data):
        raise RuntimeError("Incomplete independent validation")
    result = {"scope": "PMC development/model-validation, inspected for model selection; NOT independent test/clinical evidence",
        "checkpoint_sha256": sha(checkpoint), "manifest": saved_manifest,
        "metrics": metrics, "records": rows, "test_used": False}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    pred = [normalize_answer(row["pred"]) for row in rows]
    gt = [normalize_answer(row["gt"]) for row in rows]
    incorrect = [(row, p, target) for row, p, target in zip(rows, pred, gt) if p != target]
    audit = {"scope": result["scope"], "prediction_file_sha256": sha(output), "qa": len(rows),
        "exact_correct": sum(p == target for p, target in zip(pred, gt)),
        "incorrect_with_gt_as_whole_word_phrase": sum(bool(target) and (" " + target + " ") in (" " + p + " ") for _, p, target in incorrect),
        "empty_predictions": sum(not p for p in pred),
        "predictions_with_non_ascii": sum(any(ord(c) > 127 for c in p) for p in pred),
        "ground_truth_with_non_ascii": sum(any(ord(c) > 127 for c in target) for target in gt),
        "most_common_predictions": Counter(pred).most_common(10),
        "ground_truth_word_lengths": Counter(str(len(target.split())) for target in gt),
        "first10_incorrect_existing_GT_examples": [row for row, _, _ in incorrect[:10]],
        "interpretation": "Descriptive format/content diagnostics only; no automatic clinical correctness relabel or test-tuned repair",
        "next_stage_released": False}
    errors_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    status.update(status="generation_complete_pending_error_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(),
        evaluated_qa=len(rows), predictions_sha256=sha(output), error_audit_sha256=sha(errors_path))
    progress_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "qa": len(rows), "prediction_output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
