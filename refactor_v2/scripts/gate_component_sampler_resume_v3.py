"""Bounded real training CLI sampler/resume gate, canonical CLOSED TRAIN QA only."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from src.data.sealed_manifest import load_sealed_records
from src.endpoints import is_canonical_fracture_presence_question
from src.utils import load_config, normalize_answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["B", "P"], default="B")
    parser.add_argument("--context", choices=["conditioned", "unconditioned", "shuffled"], default="conditioned")
    args = parser.parse_args()
    if args.variant == "B" and args.context != "conditioned":
        raise ValueError("Baseline does not use LF context")
    start = time.perf_counter()
    root = Path.cwd()
    suffix = "" if args.variant == "B" else "_p_" + args.context
    folder = root / ("refactor_v2/reports/component_sampler_resume_train_only_v3" + suffix)
    checkpoint_dir = root / ("refactor_v2/checkpoints/component_sampler_resume_train_only_v3" + suffix)
    if folder.exists() or checkpoint_dir.exists():
        raise RuntimeError("Existing sampler audit must not be overwritten")
    folder.mkdir()
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    loader = config["data"]["loader_kwargs"]
    source = Path(loader["sealed_records_by_split"]["train"])
    parent_seal = Path(loader["seal_path"])
    rows = load_sealed_records(source, parent_seal, root)
    component_path = root / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"
    singleton = {group[0] for group in json.loads(component_path.read_text())["components"] if len(group) == 1}
    labels = {label: [row for row in rows if row["image_id"] in singleton and
        is_canonical_fracture_presence_question(row["question"]) and normalize_answer(row["answer"]) == label]
        for label in ("yes", "no")}
    paths, artifacts = {}, {}
    for role, begin, end in (("train", 0, 4), ("model_val", 4, 6)):
        selected = labels["yes"][begin:end] + labels["no"][begin:end]
        path = folder / (role + ".json")
        path.write_text(json.dumps(selected, indent=2) + "\n")
        paths[role] = str(path)
        artifacts[path.relative_to(root).as_posix()] = sha(path)
    seal = folder / "data_seal.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "scope": "technical_smoke_train_only",
        "parent_manifest_sha256": sha(source), "parent_seal_sha256": sha(parent_seal), "artifacts": artifacts}, indent=2))
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["figure_dir"] = str(folder / "figures")
    config["init_from"] = str(root / "refactor_v2/checkpoints/common_initialization_seed42_fold0_v3/initial.pt")
    config["model"].update(prompt_variant=args.variant, use_latent_prompt=args.variant == "P",
                            prior_labels=[], latent_context_control=args.context)
    config["data"].update(sealed_answer_vocab=None, max_samples=None, val_max_samples=None, num_workers=0)
    loader.update(seal_path=str(seal), sealed_records_by_split=paths)
    config["train"].update(epochs=2, batch_size=4, grad_accum=1, epoch_seeded_rng=True,
        save_optimizer=True, eval_max_samples=None, select_metric="fracture_auprc",
        component_batching={"manifest": str(component_path), "seal": str(parent_seal)})
    configs = {}
    for name in ("continuous", "resumed"):
        cfg = copy.deepcopy(config)
        cfg["run_name"] = name
        path = folder / (name + ".json")
        path.write_text(json.dumps(cfg, indent=2))
        configs[name] = path
    def run(name, tag, extra=()):
        with (folder / (tag + ".log")).open("w") as log:
            subprocess.run([sys.executable, "-m", "src.train", "--config", str(configs[name]),
                            "--device", "cpu", "--technical_component_smoke", *extra],
                            env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    run("continuous", "continuous")
    run("resumed", "first_epoch", ["--stop_after_epoch", "1"])
    run("resumed", "resume_epoch2", ["--resume", str(checkpoint_dir / "resumed/last.pt")])
    left, right = [torch.load(checkpoint_dir / name / "last.pt", map_location="cpu", weights_only=False)
                   for name in ("continuous", "resumed")]
    lo, ro = [torch.load(checkpoint_dir / name / "optimizer.pt", map_location="cpu", weights_only=False)
              for name in ("continuous", "resumed")]
    report = {"scope": "8 TRAIN closed fracture QA; 4 separate TRAIN QA for internal validation; not efficacy/open-generation evidence",
        "variant": args.variant, "context_control": args.context,
        "parameters_exact": equal_tree(left["state"], right["state"]),
        "optimizer_exact": equal_tree(lo["optimizer"], ro["optimizer"]),
        "scheduler_exact": equal_tree(lo["scheduler"], ro["scheduler"]),
        "batch_binding_exact": left["extra"]["component_batch_binding"] == right["extra"]["component_batch_binding"],
        "tensor_count": len(left["state"]), "steps": left["extra"]["global_step"],
        "seconds": time.perf_counter() - start, "shuffled_scientific_training_enabled": False}
    report["passed"] = all(report[key] for key in ("parameters_exact", "optimizer_exact", "scheduler_exact", "batch_binding_exact"))
    (folder / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
