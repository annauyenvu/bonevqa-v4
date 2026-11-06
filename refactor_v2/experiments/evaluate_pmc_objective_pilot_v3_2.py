"""Verify paired pilot artifacts, then independently reload dev predictions/answer CE."""
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import manifest
from src.train import build_split
from src.utils import compute_metrics, normalize_answer, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


@torch.no_grad()
def main():
    root = Path.cwd()
    folder = root / "refactor_v2/reports/pmc_objective_pilot_v3_2"
    status = json.loads((folder / "progress.json").read_text())
    progress = folder / "independent_evaluation_progress.json"
    output = folder / "independent_paired_answer_evaluation.json"
    if status["status"] != "both_arms_complete_pending_independent_evaluation" or progress.exists() or output.exists():
        raise RuntimeError("Pilot incomplete or evaluation already exists; never duplicate/overwrite")
    state = {"status": "verifying", "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "scope": "independent reload of128 inspected development QA; NOT independent test/clinical evidence"}
    def save():
        progress.write_text(json.dumps(state, indent=2) + "\n")
    save()
    started = time.perf_counter()
    try:
        gate = json.loads((folder / "execution_release.json").read_text())
        assert gate["passed"] and gate["initialization_byte_identical"]
        names = ("pmc_objective_current_v3_2", "pmc_objective_answer_control_v3_2")
        results, checks, checkpoints = {}, {}, {}
        selected = []
        for name in names:
            run = root / "refactor_v2/checkpoints" / name
            completion = json.loads((folder / (name + "_completion.json")).read_text())
            assert sha(folder / (name + ".json")) == gate["config_sha256"][name]
            assert sha(run / "best.pt") == completion["best_sha256"]
            assert sha(run / "last.pt") == completion["last_sha256"]
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            optimizer = torch.load(run / "optimizer.pt", map_location="cpu", weights_only=False)
            history = json.loads((run / "history.json").read_text())
            cfg = json.loads((run / "config.json").read_text())
            assert finite_tree(best["state"]) and finite_tree(last["state"]) and finite_tree(optimizer)
            assert last["extra"]["global_step"] == 64 and last["extra"]["epoch"] == 4
            assert len(history) == 4 and all(row["val_n_open"] == 128 and row["skipped_no_supervision_batches"] == 0 for row in history)
            assert best["extra"]["best_metric"] == min(row["val_loss"] for row in history)
            selected.append((best["extra"]["epoch"], best["extra"]["global_step"]))
            data = {role: build_split(build_dataset, "pmc_vqa", role, None, True, cfg["data"]["loader_kwargs"]) for role in ("train", "model_val")}
            actual = {role: manifest(ds) for role, ds in data.items()}
            assert actual == gate["actual_data_manifests"]
            saved = json.loads((run / "data_manifest.json").read_text())
            assert saved["train"] == actual["train"] and saved["val"] == actual["model_val"]
            qc = json.loads((run / "prompt_module_qc.json").read_text())
            checks[name] = {"finite_weights_and_optimizer": True, "full64_step_budget": True,
                            "exact_selected_manifests": True, "best_epoch": selected[-1][0], "best_step": selected[-1][1],
                            "prompt_module_qc": qc, "wall_seconds": completion["wall_seconds"]}
            checkpoints[name] = run / "best.pt"
            del best, last, optimizer
            gc.collect()
        assert len(set(selected)) == 1, "Best checkpoints must have equal exposure for this paired diagnostic"
        cfg = json.loads((root / "refactor_v2/checkpoints" / names[0] / "config.json").read_text())
        dataset = build_split(build_dataset, "pmc_vqa", "model_val", None, True, cfg["data"]["loader_kwargs"])
        assert len(dataset) == 128
        checkpoints = {"roco_initialization": Path(cfg["init_from"]), **checkpoints}
        assert sha(checkpoints["roco_initialization"]) == gate["fresh_roco_sha256"]
        (folder / "independent_artifact_verification.json").write_text(json.dumps({"passed": True, "arms": checks,
            "best_checkpoint_exposure_matched": True, "test_used": False, "quality_gate_passed": False}, indent=2) + "\n")
        for name, checkpoint in checkpoints.items():
            set_seed(42)
            model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
            for condition in ("correct_image", "cyclic_wrong_image"):
                records, nll, token_count = [], 0.0, 0
                for offset in range(0, len(dataset), 4):
                    nb = normalize_batch(collate_fn([dataset[i] for i in range(offset, offset + 4)]))
                    images = nb["images"]
                    if condition == "cyclic_wrong_image":
                        images = images[1:] + images[:1]
                    core = model._encode_core(images, nb["questions"], nb["regions"], [None] * 4, model.cfg["prompt_kind"], False)
                    inputs, attention, labels = model._build_llm_inputs(model._llm_prefix(core), core["q_texts"], nb["answers"])
                    logits = model.llm(inputs_embeds=inputs, attention_mask=attention).logits
                    targets = labels[:, 1:]
                    nll += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum"))
                    token_count += int((targets != -100).sum())
                    del logits, inputs
                    generated = model.generate_open(core, core["q_texts"])
                    for i, pred in enumerate(generated):
                        records.append({"image_id": nb["image_ids"][i], "question": nb["questions"][i], "gt": nb["answers"][i],
                                        "pred": pred["text"], "answer_type": "OPEN"})
                results[name + "/" + condition] = {"checkpoint_sha256": sha(checkpoint), "metrics": compute_metrics(records),
                    "answer_token_ce": nll / token_count, "supervised_tokens": token_count, "records": records}
                state.update(status="running", checkpoint=name, condition_completed=condition)
                save()
            del model
            gc.collect()
        for name in checkpoints:
            correct = results[name + "/correct_image"]
            wrong = results[name + "/cyclic_wrong_image"]
            correct["wrong_minus_correct_answer_ce"] = wrong["answer_token_ce"] - correct["answer_token_ce"]
            correct["answer_changes_on_image_swap"] = sum(normalize_answer(a["pred"]) != normalize_answer(b["pred"]) for a, b in zip(correct["records"], wrong["records"]))
        report = {"scope": state["scope"], "conditions": results, "elapsed_seconds": time.perf_counter() - started,
                  "matched_seed": 42, "best_checkpoint_exposure": selected[0], "artifact_verification_passed": True,
                  "limitations": ["one-seed bounded development pilot, not independent evidence", "raw val_loss totals are not comparable across different contrastive weights",
                                  "no clinical answer relabeling; original exact normalizer retained", "changed answer under image swap does not prove useful grounding"],
                  "test_used": False, "long_training_released": False, "quality_gate_passed": False}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        state.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(output))
        save()
        print(json.dumps({key: {k: v for k, v in value.items() if k != "records"} for key, value in results.items()}, indent=2))
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry", failure=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
