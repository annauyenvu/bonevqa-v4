#!/usr/bin/env python3
"""Run sealed full validation/test evaluation and validation-only calibration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python")
REPORT_DIR = ROOT / "refactor_v2/reports/eval"
FIGURE_DIR = ROOT / "refactor_v2/reports/figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--boxes", type=Path)
    args = parser.parse_args()
    status = REPORT_DIR / f"{args.tag}_status.json"

    def write_status(state: str, **extra) -> None:
        status.write_text(json.dumps({
            "state": state,
            "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }, indent=2), encoding="utf-8")

    def evaluate(split: str) -> Path:
        tag = f"{args.tag}_{split}"
        command = [
            str(PYTHON), "-m", "src.evaluate",
            "--checkpoint", str(args.checkpoint),
            "--dataset", "fracatlas_component_clean",
            "--split", split,
            "--batch_size", "4",
            "--report_dir", str(REPORT_DIR),
            "--figure_dir", str(FIGURE_DIR),
            "--tag", tag,
        ]
        if args.boxes:
            command.extend(["--box_bo_phat_hien", str(args.boxes)])
        subprocess.run(command, cwd=ROOT, check=True)
        return REPORT_DIR / f"results_fracatlas_component_clean_{tag}.json"

    try:
        write_status("running_validation")
        validation = evaluate("val")
        write_status("running_test", validation=str(validation))
        test = evaluate("test")
        output = REPORT_DIR / f"fracture_calibration_{args.tag}.json"
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
    main()
