"""Bounded V/L wiring on automatic TRAIN-only smoke detector proposals."""
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.checkpointing import compatible_state
from src.data import build_dataset, collate_fn
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import records_of
from src.provenance import validate_detector_proposals
from src.train import build_split, seed_training_epoch
from src.utils import load_config, set_seed
from experiments.validate_detector640_TRAIN_fixture_v3 import validate as validate_TRAIN_fixture
import hashlib


def main():
    config = load_config("refactor_v2/configs/revised_sealed_candidates_v3/b.json")
    report_path = Path("refactor_v2/reports/vl_detector640_target_routes_v3.json")
    if report_path.exists():
        raise RuntimeError("Existing target gate report; preserve")
    proposals = Path("refactor_v2/reports/detector640_top3_binding_v3/TRAIN_technical_boxes.json")
    validate_TRAIN_fixture(proposals)
    boxes = json.loads(proposals.read_text())
    roi_image = next(image for image, value in boxes.items() if value)
    no_roi_image = next(image for image, value in boxes.items() if not value)
    loader = copy.deepcopy(config["data"]["loader_kwargs"])
    loader.update(with_masks=False, with_boxes=False, boxes_from=str(proposals), boxes_as_masks=True)
    dataset = build_split(build_dataset, config["data"]["dataset"], "train", None, True, loader)
    rows = records_of(dataset)
    chosen = []
    for image in (roi_image, no_roi_image):
        chosen.append(next(i for i, row in enumerate(rows) if row["image_id"] == image
                           and is_canonical_fracture_presence_question(row["question"])))
        chosen.append(next(i for i, row in enumerate(rows) if row["image_id"] == image
                           and row.get("answer_type", "").upper() == "OPEN"))
    batch = collate_fn([dataset[index] for index in chosen])
    nb = normalize_batch(batch)
    if any(nb["boxes"][i] for i in (2, 3)):
        raise RuntimeError("Detector-negative rows unexpectedly acquired proposals")
    assert hashlib.sha256(Path("refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt").read_bytes()).hexdigest() == "bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b"
    initial = torch.load("refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt",
                         map_location="cpu", weights_only=False)
    results = []
    for variant in ("V", "L"):
        start = time.perf_counter()
        folder = Path("refactor_v2/checkpoints") / ("gate_" + variant.lower() + "_detector640_target_route_v3")
        if folder.exists():
            raise RuntimeError("Existing gate artifacts must be inspected, not overwritten: " + str(folder))
        cfg = copy.deepcopy(initial["config"])
        cfg.update(prompt_variant=variant, use_visual_prompt=variant == "V", use_lens=variant == "L", use_latent_prompt=False, use_region_in_prompt=False, use_textual_prompt=False, prior_labels=[])
        set_seed(42)
        model = BoneVQAModel(cfg, initial["answer_vocab"], device="cpu").to("cpu")
        model.load_state_dict(compatible_state(model.trainable_state_dict(), initial["state"],
                                              config["checkpoint_allow_mismatch_prefixes"]), strict=False)
        shared = model.trainable_state_dict()
        assert len(initial["state"]) == 263 and len(initial["answer_vocab"]) == 68
        assert all(torch.equal(shared[k], value) for k, value in initial["state"].items())
        del shared
        parameters = dict(model.named_parameters())
        if variant == "V":
            names = ["prompt_mixer.gamma_visual", "prompt_mixer.visual_proj.0.weight"]
        else:
            names = [next(name for name in parameters if name.startswith("lens_fusion.")),
                     next(name for name in parameters if name.startswith("visual.") and "lora_B" in name)]
        before = {key: parameters[key].detach().clone() for key in names}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01)
        steps = []
        for step in range(2):
            seed_training_epoch(42, step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, stage=3)
            objective = out["loss_answer"] if variant == "V" else out["loss_lens_alignment"]
            grads = torch.autograd.grad(objective, [parameters[key] for key in names],
                                        retain_graph=True, allow_unused=True)
            gradient = {key: 0.0 if value is None else float(value.abs().max()) for key, value in zip(names, grads)}
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps.append({"step": step + 1, "loss": float(out["loss"].detach()),
                          "lens_alignment_loss": float(out["loss_lens_align"]), "route_gradients": gradient})

        @torch.no_grad()
        def endpoints(data):
            model.eval()
            core = model._encode_core(data["images"], data["questions"], data["regions"],
                                      model._prepare_masks(data["images"], data["masks"], data["boxes"], data["image_ids"]),
                                      cfg["prompt_kind"], False)
            idx = [1, 3]
            prefix = model._llm_prefix(core)[idx]
            embeds, attention, labels = model._build_llm_inputs(prefix, [core["q_texts"][i] for i in idx],
                                                               [data["answers"][i] for i in idx])
            logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
            return {"closed": core["logits_closed"].detach(), "fracture": core["fracture_logit"].detach(),
                    "open": logits[:, :-1][labels[:, 1:] != -100].detach().float()}

        real = endpoints(nb)
        empty = dict(nb)
        empty.update(boxes=[None] * 4, masks=[np.zeros((0, im.height, im.width), dtype=np.uint8) for im in nb["images"]])
        no_roi = endpoints(empty)
        key = "visual_force_zero" if variant == "V" else "lens_force_zero"
        model.cfg[key] = True
        neutral = endpoints(nb)
        model.cfg[key] = False
        no_roi_row_gate_changes = {name: float((real[name][2:] - neutral[name][2:]).abs().max()) for name in ("closed", "fracture")}
        if variant == "V":
            assert all(value == 0 for value in no_roi_row_gate_changes.values())
        updates = {name: float((parameters[name].detach() - before[name]).abs().max()) for name in names}
        difference = {name: float((real[name] - neutral[name]).abs().max()) for name in real}
        original_only = {name: float((real[name] - no_roi[name]).abs().max()) for name in real}
        folder.mkdir(parents=True)
        model.save(str(folder / "smoke.pt"), {"scope": "two-step TRAIN-only detector-proposal wiring audit"})
        model.load_trainable(str(folder / "smoke.pt"))
        restored = endpoints(nb)
        reload_diff = {name: float((real[name] - restored[name]).abs().max()) for name in real}
        passed = (all(value > 0 for value in steps[-1]["route_gradients"].values())
                  and all(value > 0 for value in updates.values())
                  and all(value == 0 for value in reload_diff.values()))
        passed = passed and (all(value > 0 for value in difference.values()) if variant == "V"
                             else all(value == 0 for value in original_only.values()))
        results.append({"variant": variant, "steps": steps, "selected_updates": updates,
                        "shared_initial_tensors_exact": 263, "vocab_size": 68, "no_roi_row_gate_differences": no_roi_row_gate_changes,
                        "gate_zero_endpoint_differences": difference, "no_roi_inference_differences": original_only,
                        "reload_differences": reload_diff, "seconds": time.perf_counter() - start, "passed": passed})
        print(json.dumps(results[-1]), flush=True)
        del optimizer, model, parameters, out, grads, real, neutral, restored
        gc.collect()
    report = {"scope": "four fullfoldTRAIN QA rows,two updates perV/L with vocab68;technical only,no quality or scientific superiority",
              "device": "cpu", "proposals": str(proposals), "source": "effective640 detector automatic proposals only,annotation ROI disabled",
              "proposal_sha256": hashlib.sha256(proposals.read_bytes()).hexdigest(),
              "PREtask_initialization_sha256": hashlib.sha256(Path("refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt").read_bytes()).hexdigest(),
              "calibration_or_test_used": False, "scientific_training_released": False,
              "train_indices": chosen, "roi_image": roi_image, "no_roi_image": no_roi_image,
              "limitations": ["selected for branch coverage, not unbiased efficacy;fullfoldTRAIN rows not necessarilyBpilot261images",
                              "same-instance strict reload only;independent fresh-process reload still pending",
                              "L is training-only alignment; zero direct inference contribution is intentional",
                              "full fold proposals, matched controls and held-out outcomes remain pending"],
              "results": results, "all_passed": all(row["passed"] for row in results)}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = Path("refactor_v2/reports/vl_detector640_target_routes_v3_failure.json")
        if not failure.exists():
            failure.write_text(json.dumps({"status": "failed_preserved_no_automatic_retry", "failure": repr(error), "scope": "TRAIN-only2updatewiringgate,noqualitytraining"}, indent=2) + "\n")
        raise
