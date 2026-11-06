#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import traceback
from pathlib import Path

from runtime import ROOT, configure


SEEDS = [42, 123, 456, 789, 2026]
VARIANTS = ["none", "v", "p", "vp"]


def main() -> None:
    common, engine, specs = configure()
    common.setup(42)
    lock_handle = (ROOT / "pipeline.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    common.verify_lock()
    assert common.read(ROOT / "roi_bank/COMPLETE.json")["roi_failed"] == 0
    completed = 0
    total = len(SEEDS) * len(VARIANTS)
    for seed in SEEDS:
        for variant in VARIANTS:
            destination = ROOT / "detector" / variant / f"seed_{seed}"
            qc_path = destination / "QC.json"
            if qc_path.exists():
                qc = common.read(qc_path)
                if not qc.get("valid"):
                    raise RuntimeError(f"failed run requires manual review: {destination}")
            else:
                common.status("primary_matrix", completed=completed, total=total, seed=seed, variant=variant)
                command = [sys.executable, str(ROOT / "code/worker.py"), "train", "--variant", variant,
                           "--seed", str(seed), "--dest", str(destination)]
                with (destination / "stdout.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONHASHSEED": str(seed)}, check=True)
            completed += 1
            common.write(ROOT / "progress.json", {"completed_primary_runs": completed,
                                                    "total_primary_runs": total,
                                                    "last_completed": str(destination.relative_to(ROOT))})
            print("COMPLETED", completed, "/", total, destination.relative_to(ROOT), flush=True)
    common.status("analysis", completed_primary_runs=completed)
    with (ROOT / "analysis.log").open("a") as log:
        subprocess.run([sys.executable, str(ROOT / "code/analyze.py")], stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONHASHSEED": "42"}, check=True)
    common.status("mandatory_training", "complete", completed_primary_runs=completed)
    print("MANDATORY TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        common, _, _ = configure()
        common.write(ROOT / "TECHNICAL_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                                       "policy": "no automatic retry; inspect and resume explicitly"})
        common.status("mandatory_training", "technical_failure", error=repr(exc))
        raise
