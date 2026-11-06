"""Bounded TRAIN forward/backward benchmark; never modifies weights or starts amid training."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_threads", type=int)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    folder = root / "refactor_v2/reports/cpu_thread_benchmark_v3"
    status_path = root / "refactor_v2/reports/roco_alignment_execution_v3/progress.json"
    status = json.loads(status_path.read_text())
    if args.worker_threads is None:
        processes = subprocess.check_output(["ps", "-axo", "pid,command"], text=True)
        active = [line for line in processes.splitlines() if " -m src.train " in line]
        if active or status["status"] == "running":
            print(json.dumps({"benchmark_started": False, "reason": "existing training has priority; do not contend",
                              "planned_threads": [8, 10, 12], "training_status": status["status"]}))
            return
        if args.check_only:
            print(json.dumps({"benchmark_started": False, "ready_for_bounded_benchmark": True}))
            return
        if status["status"] != "training_complete_pending_independent_checkpoint_evaluation":
            raise RuntimeError("Need successfully completed ROCO stage before benchmarking")
        if folder.exists():
            raise RuntimeError("Existing benchmark must not be overwritten")
        folder.mkdir()
        outputs = []
        for threads in (8, 10, 12):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads))
            with (folder / (str(threads) + ".log")).open("w") as log:
                subprocess.run([sys.executable, __file__, "--worker_threads", str(threads)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            outputs.append(json.loads((folder / (str(threads) + ".json")).read_text()))
        reference = outputs[0]
        for row in outputs:
            row["loss_delta_vs_8_threads"] = max(abs(a - b) for a, b in zip(reference["losses"], row["losses"]))
            row["numerical_screen_passed"] = row["finite_selected_gradients"] and row["loss_delta_vs_8_threads"] <= 1e-5
        passed = [row for row in outputs if row["numerical_screen_passed"]]
        result = {"scope": "four TRAIN captions, same weights/seed; forward+backward only; no optimizer or held-out evidence",
                  "results": outputs, "fastest_screened_threads": min(passed, key=lambda row: row["median_seconds"])["threads"] if passed else None,
                  "runtime_released": False,
                  "remaining": "independent update/resume numerical gate and end-to-end pilot required before changing sealed runtime"}
        (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    # Separate processes isolate OpenMP pools and caches between thread arms.
    import resource
    import torch
    from src.data import build_dataset, collate_fn
    from src.models.bonevqa import BoneVQAModel
    from src.train import build_split
    from src.utils import set_seed
    threads = args.worker_threads
    if threads not in (8, 10, 12):
        raise ValueError("Only the declared bounded thread arms are allowed")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    checkpoint = root / "refactor_v2/checkpoints/revised_roco_alignment_v3/best.pt"
    cfg = json.loads((checkpoint.parent / "config.json").read_text())
    dataset = build_split(build_dataset, "roco", "train", None, True, cfg["data"]["loader_kwargs"])
    batch = collate_fn([dataset[i] for i in (0, 1, 2, 3)])
    model = BoneVQAModel.load(str(checkpoint), device="cpu").train()
    selected = dict(model.named_parameters())
    names = ("qformer.query", "prefix_proj.0.weight")
    times, losses, finite = [], [], True
    for repeat in range(3):
        set_seed(42)
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out = model(batch, stage=1)
        out["loss"].backward()
        elapsed = time.perf_counter() - start
        finite = finite and all(selected[name].grad is not None and bool(torch.isfinite(selected[name].grad).all())
                                for name in names)
        if repeat:
            times.append(elapsed)
            losses.append(float(out["loss"].detach()))
    result = {"threads": threads, "seconds": times, "median_seconds": statistics.median(times),
              "losses": losses, "finite_selected_gradients": finite,
              "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "optimizer_steps": 0, "checkpoint_modified": False}
    (folder / (str(threads) + ".json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
