"""Audit resolved candidate data/loss/lineage without launching or fitting models."""
import hashlib
import json
import math
from pathlib import Path
from src.context_controls import load_component_lookup
from src.data.sealed_manifest import load_sealed_records
from src.protocol import validate_prompt_protocol_config
from src.utils import normalize_answer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    index_path = root / "refactor_v2/reports/revised_recipe_candidates_v3.json"
    index = json.loads(index_path.read_text())
    output = root / "refactor_v2/reports/revised_recipe_audit_v3.json"
    if output.exists():
        raise RuntimeError("Existing recipe audit must not be overwritten")
    cache, configs, audits = {}, {}, []
    for relative, expected in index["artifacts"].items():
        path = root / relative
        if digest(path) != expected:
            raise RuntimeError("Candidate config hash changed: " + relative)
        cfg = json.loads(path.read_text())
        validate_prompt_protocol_config(cfg)
        configs[path.stem] = cfg
        model, train, data = cfg["model"], cfg["train"], cfg["data"]
        loader = data["loader_kwargs"]
        roles, rows_by_role = {}, {}
        for role, source in loader["sealed_records_by_split"].items():
            if not source:
                continue
            key = (source, loader["seal_path"])
            if key not in cache:
                cache[key] = load_sealed_records(source, loader["seal_path"], loader["project_root"])
            rows_by_role[role] = cache[key]
            roles[role] = {"rows": len(cache[key]), "sha256": digest(source)}
        assert train["epoch_seeded_rng"] and not model["grad_checkpoint"] and not model["llm_grad_checkpoint"]
        assert not model["prior_labels"] and not loader["with_masks"] and not loader["with_boxes"]
        assert not model["use_region_in_prompt"] and not model["sam"]["enabled"]
        for coefficient in ("eta", "lam", "type_loss_weight", "fracture_loss_weight"):
            assert math.isfinite(model[coefficient]) and model[coefficient] >= 0
        missing = []
        if cfg.get("init_from") and not (root / cfg["init_from"]).exists():
            missing.append("upstream_checkpoint")
        if data["dataset"] == "fracatlas_component_clean":
            batching = train["component_batching"]
            lookup, component_hash = load_component_lookup(batching["manifest"], batching["seal"], root)
            groups = [{lookup[row["image_id"]] for row in rows} for rows in rows_by_role.values()]
            assert all(not left & right for i, left in enumerate(groups) for right in groups[i + 1:])
            vocab = load_sealed_records(data["sealed_answer_vocab"], loader["seal_path"], root)
            assert set(vocab) == {normalize_answer(row["answer"]) for row in rows_by_role["train"]}
            assert set(roles) == {"train", "model_val", "calibration", "test"}
            if train["stage"] != 1:
                assert train["select_metric"] == "fracture_auprc"
            if model["use_visual_prompt"] or model["use_lens"]:
                assert Path(loader["boxes_from"]) == root / "refactor_v2/detector_fold0_v3/boxes.json"
                assert loader["boxes_as_masks"]
                if not Path(loader["boxes_from"]).exists():
                    missing.append("new_fold_complete_detector_proposals")
        else:
            assert train["component_batching"] is None and data["sealed_answer_vocab"] is None
            assert not model["use_fracture_head"] and train["select_metric"] == "val_loss"
        audits.append({"recipe": path.stem, "sha256": expected, "roles": roles,
            "stage": train["stage"], "epochs": train["epochs"], "reference_device": cfg["verified_reference_device"],
            "execution_inputs_not_yet_available": missing,
            "effective_contrastive": "dcl" if train["stage"] == 1 else model["contrastive_mode"],
            "effective_open_ce_weight": model.get("stage1_caption_loss_weight", 0) if train["stage"] == 1 else 1,
            "passed_recipe_structure": True})
    for phase in ("align", "vqa"):
        reference = configs["l_real_" + phase]
        for arm in ("original", "matched", "shuffled"):
            control = configs["l_" + arm + "_" + phase]
            assert reference["train"] == control["train"]
            assert reference["data"]["sealed_answer_vocab"] == control["data"]["sealed_answer_vocab"]
            for field in ("eta", "lam", "type_loss_weight", "fracture_loss_weight", "lens_alignment_weight"):
                assert reference["model"][field] == control["model"][field]
            if phase == "vqa":
                assert ("l_" + arm + "_align") in control["init_from"]
    for name in ("p_unconditioned", "p_shuffled_train_only", "p_random_bank", "p_empty_bank"):
        assert configs["p"]["train"] == configs[name]["train"]
    assert configs["p_shuffled_train_only"]["model"]["latent_context_control_eval"] == "conditioned"
    assert configs["p_unconditioned"]["model"]["latent_context_control_eval"] == "unconditioned"
    report = {"status": "passed_structural_recipe_audit", "candidate_index_sha256": digest(index_path),
        "recipes": audits, "count": len(audits), "L_phase_train_settings_equal": True,
        "P_train_budget_settings_equal": True, "passed": True,
        "long_training_authorized": False,
        "limitations": ["not measured wall-clock/FLOP equality", "upstream checkpoints and full-fold proposals remain unavailable",
                        "new upstream-trained initialization needs byte-parity QC before downstream ablations",
                        "no clinical, causal superiority or final module selection claim"]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "recipes"}, indent=2))


if __name__ == "__main__":
    main()
