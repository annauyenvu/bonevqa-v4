#!/usr/bin/env python3
"""Run the sealed None validation/test calibration sequence without duplicates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "refactor_v2/reports/eval/none_calibration_v2_status.json"
PYTHON = Path("/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python")
CHECKPOINT = ROOT / "refactor_v2/checkpoints/fracatlas_clean_none_seed42_ce_v3/best.pt"
REPORT_DIR = ROOT / "refactor_v2/reports/eval"
FIGURE_DIR = ROOT / "refactor_v2/reports/figures"
VAL_TAG = "none_seed42_ce_v4_softmax_fullval"
TEST_TAG = "none_seed42_ce_v4_softmax_fulltest"


def write_status(state: str, **extra) -> None:
    payload = {
        "state": state,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate(split: str, tag: str) -> Path:
    command = [
        str(PYTHON), "-m", "src.evaluate",
        "--checkpoint", str(CHECKPOINT),
        "--dataset", "fracatlas_component_clean",
        "--split", split,
        "--batch_size", "4",
        "--report_dir", str(REPORT_DIR),
        "--figure_dir", str(FIGURE_DIR),
        "--tag", tag,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return REPORT_DIR / f"results_fracatlas_component_clean_{tag}.json"


def main() -> None:
    try:
        write_status("running_validation")
        validation = evaluate("val", VAL_TAG)
        write_status("running_test", validation=str(validation))
        test = evaluate("test", TEST_TAG)
        output = REPORT_DIR / "fracture_calibration_none_seed42_ce_v4.json"
        write_status("calibrating", validation=str(validation), test=str(test))
        subprocess.run([
            str(PYTHON), str(ROOT / "refactor_v2/scripts/calibrate_fracture_threshold_v2.py"),
            "--validation", str(validation),
            "--test", str(test),
            "--output", str(output),
        ], cwd=ROOT, check=True)
        write_status("complete", validation=str(validation), test=str(test), report=str(output))
    except Exception as exc:
        write_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
