"""Bounded 16-QA TRAIN memorization diagnostic, never generalization evidence."""
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import manifest, records_of
from src.train import build_split, seed_training_epoch
from src.utils import compute_metrics, normalize_answer, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_train_capacity_v3_2"
    if folder.exists():
        raise RuntimeError("Existing capacity diagnostic must be preserved, not duplicated")
    commands = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in commands.splitlines()):
        raise RuntimeError("Existing training alive")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("OMP_NUM_THREADS") != "8":
        raise RuntimeError("Offline CPU8 reference required")
    pilot = root / "refactor_v2/reports/pmc_objective_pilot_v3_2"
    evaluated = json.loads((pilot / "independent_evaluation_progress.json").read_text())
    if evaluated["status"] != "complete_pending_interpretation" or sha(pilot / "independent_paired_answer_evaluation.json") != evaluated["report_sha256"]:
        raise RuntimeError("Pilot evaluation incomplete/unverified")
    cfg = json.loads((pilot / "pmc_objective_current_v3_2.json").read_text())
    source = Path(cfg["data"]["loader_kwargs"]["sealed_records_by_split"]["train"])
    rows = json.loads(source.read_text())
    selected = [row for row in rows if 1 <= len(str(row["answer"]).split()) <= 4 and normalize_answer(row["answer"])][:16]
    if len(selected) != 16:
        raise RuntimeError("Need16 deterministic short existing TRAIN targets")
    folder.mkdir()
    progress = folder / "progress.json"
    status = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "scope": "16 existing TRAIN QA; fixed100 optimization steps; memorization capacity ONLY; no validation/test"}
    def save():
        progress.write_text(json.dumps(status, indent=2) + "\n")
    save()
    started = time.perf_counter()
    try:
        path = folder / "train.json"
        path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
        seal_path = folder / "data_seal.json"
        seal = {"status": "sealed_development_data", "scope": "technical_smoke_train_only",
                "parent_manifest_sha256": sha(source), "parent_seal_sha256": sha(cfg["data"]["loader_kwargs"]["seal_path"]),
                "selection": "first16 TRAIN article-distinct rows with1to4word nonempty normalized targets; not a semantic quality subset",
                "artifacts": {path.relative_to(root).as_posix(): sha(path)}, "validation_used": False, "test_used": False}
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
        loader = dict(cfg["data"]["loader_kwargs"])
        loader.update(seal_path=str(seal_path), sealed_records_by_split={"train": str(path), "model_val": None, "calibration": None, "test": None})
        dataset = build_split(build_dataset, "pmc_vqa", "train", None, True, loader)
        assert len(dataset) == 16 and len({row["image_id"] for row in records_of(dataset)}) == 16
        checkpoint = root / "refactor_v2/checkpoints/pmc_objective_current_v3_2/best.pt"
        completion = json.loads((pilot / "pmc_objective_current_v3_2_completion.json").read_text())
        assert sha(checkpoint) == completion["best_sha256"]
        set_seed(42)
        model = BoneVQAModel.load(str(checkpoint), device="cpu")
        # No recipe search: retain current loss weights and optimizer family.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01, betas=(0.9, 0.98))
        def schedule(step):
            if step < 5:
                return (step + 1) / 5
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 5) / 95)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        parameters = dict(model.named_parameters())
        selected_names = ["qformer.query", "prefix_proj.0.weight"]
        initial = {name: parameters[name].detach().clone() for name in selected_names}
        @torch.no_grad()
        def measure(step):
            model.eval()
            records, losses = [], []
            for start in range(0, 16, 4):
                batch = collate_fn([dataset[index] for index in range(start, start + 4)])
                nb = normalize_batch(batch)
                losses.append(float(model(batch, stage=2)["loss_lm"]))
                predicted = model.predict_batch(batch)
                for i, pred in enumerate(predicted):
                    records.append({"image_id": nb["image_ids"][i], "question": nb["questions"][i], "gt": nb["answers"][i],
                                    "pred": pred["answer"], "answer_type": "OPEN"})
            return {"step": step, "mean_microbatch_answer_ce": sum(losses) / 4, "TRAIN_metrics": compute_metrics(records), "records": records}
        snapshots = [measure(0)]
        release = {"scope": status["scope"], "max_optimizer_steps": 100, "batch_size": 4, "grad_accum": 4,
                   "init_checkpoint_sha256": sha(checkpoint), "actual_train_manifest": manifest(dataset), "loss_weights": {k: model.cfg[k] for k in ("lam", "eta", "type_loss_weight")},
                   "optimizer": "AdamW1e-4,weight_decay0.01,betas0.9/0.98;5step warmup+cosine100steps;clip1.0",
                   "capacity_criterion_predeclared": "at_least14of16 TRAIN exact and at_least80percent drop in mean_microbatch_answer_CE at step100",
                   "criterion_is_not_a_semantic_or_clinical_gate": True, "scientific_training_released": False}
        (folder / "execution_release.json").write_text(json.dumps(release, indent=2) + "\n")
        gradient_qc = {}
        for step in range(100):
            seed_training_epoch(42, step)
            order = list(range(16))
            random.Random(42 + step).shuffle(order)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for start in range(0, 16, 4):
                out = model(collate_fn([dataset[index] for index in order[start:start + 4]]), stage=2)
                if not bool(torch.isfinite(out["loss"])):
                    raise RuntimeError("Nonfinite TRAIN capacity loss")
                if step == 0 and start == 0:
                    gradients = torch.autograd.grad(out["loss_answer"], [parameters[name] for name in selected_names], retain_graph=True, allow_unused=True)
                    gradient_qc = {name: 0.0 if grad is None else float(grad.abs().max()) for name, grad in zip(selected_names, gradients)}
                    assert all(value > 0 and math.isfinite(value) for value in gradient_qc.values())
                losses.append(float(out["loss_lm"]))
                (out["loss"] / 4).backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite capacity gradient norm")
            optimizer.step()
            scheduler.step()
            status.update(status="running", completed_steps=step + 1, max_steps=100, latest_train_answer_ce=sum(losses) / 4)
            if step + 1 in (25, 50, 100):
                snapshots.append(measure(step + 1))
                status["latest_TRAIN_exact"] = snapshots[-1]["TRAIN_metrics"]["open_acc"]
            save()
        del out
        updates = {name: float((parameters[name] - initial[name]).abs().max()) for name in selected_names}
        assert all(value > 0 and math.isfinite(value) for value in updates.values())
        final = folder / "technical_capacity_only.pt"
        model.save(str(final), {"scope": "technical_smoke_train_only_memorization_NOT_production", "train_manifest": manifest(dataset), "optimizer_steps": 100})
        ratio = snapshots[-1]["mean_microbatch_answer_ce"] / snapshots[0]["mean_microbatch_answer_ce"]
        report = {"scope": status["scope"], "snapshots": snapshots, "answer_only_gradient_max": gradient_qc,
                  "selected_parameter_max_update": updates, "TRAIN_capacity_criterion_passed": snapshots[-1]["TRAIN_metrics"]["open_acc"] >= 14 / 16 and ratio <= 0.2,
                  "answer_ce_final_over_initial": ratio, "checkpoint_sha256": sha(final), "elapsed_seconds": time.perf_counter() - started,
                  "clinical_or_generalization_evidence": False, "long_training_released": False, "validation_used": False, "test_used": False}
        report_path = folder / "capacity_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        status.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(report_path))
        save()
    except Exception as error:
        status.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
