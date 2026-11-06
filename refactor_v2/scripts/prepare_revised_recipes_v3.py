"""Versioned, fully resolved candidates; deliberately NOT a scientific launch seal."""
import copy
import hashlib
import json
from pathlib import Path
from src.protocol import validate_prompt_protocol_config
from src.utils import load_config


def main():
    root = Path.cwd()
    destination = root / "refactor_v2/configs/revised_sealed_candidates_v3"
    if destination.exists():
        raise RuntimeError("Existing revised recipes must not be overwritten")
    base = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    base["train"].update(epoch_seeded_rng=True, component_batching={
        "manifest": str(root / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"),
        "seal": str(root / "refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json")})
    base["model"].update(eta=0.01, lam=0.1, type_loss_weight=0.1, fracture_loss_weight=1.0,
        lens_alignment_weight=0.1, stage1_caption_loss_weight=0.0,
        latent_context_control="conditioned", latent_context_control_eval="conditioned")
    proposal = str(root / "refactor_v2/detector_fold0_v3/boxes.json")
    recipes = {}
    def add(name, cfg):
        cfg["run_name"] = "revised_fold0_" + name + "_seed42_v3"
        cfg["recipe_status"] = "candidate_not_launch_sealed"
        cfg["verified_reference_device"] = "cpu"
        validate_prompt_protocol_config(cfg)
        recipes[name] = cfg
    for variant in ("B", "V", "P", "T", "VP"):
        cfg = copy.deepcopy(base)
        cfg["model"].update(prompt_variant=variant,
            use_visual_prompt="V" in variant, use_lens=False,
            use_latent_prompt="P" in variant, use_textual_prompt="T" in variant)
        if "V" in variant:
            cfg["data"]["loader_kwargs"].update(boxes_from=proposal, boxes_as_masks=True)
        add(variant.lower(), cfg)
    for name, mode in (("p_unconditioned", "unconditioned"), ("p_shuffled_train_only", "shuffled")):
        cfg = copy.deepcopy(recipes["p"])
        cfg["model"].update(latent_context_control=mode,
            latent_context_control_eval="conditioned" if mode == "shuffled" else mode)
        add(name, cfg)
    for name, mode in (("p_random_bank", "random"), ("p_empty_bank", "empty")):
        cfg = copy.deepcopy(recipes["p"])
        cfg["model"]["latent_answer_bank_mode"] = mode
        add(name, cfg)
    for name, style in (("t_irrelevant_matched", "irrelevant_matched"), ("t_paraphrase", "clinical_paraphrase")):
        cfg = copy.deepcopy(recipes["t"])
        cfg["model"]["textual_prompt_style"] = style
        add(name, cfg)
    for name, control in (("v_matched", "matched"), ("v_shuffled", "shuffled")):
        cfg = copy.deepcopy(recipes["v"])
        cfg["data"]["loader_kwargs"]["roi_control"] = control
        add(name, cfg)
    # Every L arm gets the SAME two phases, loss coefficients and exposure
    # horizon. L-vs-B alone is not an equal-compute semantic comparison.
    for arm, control, original in (("real", "real", False), ("original", "real", True),
                                    ("matched", "matched", False), ("shuffled", "shuffled", False)):
        cfg = copy.deepcopy(base)
        cfg["model"].update(prompt_variant="L", use_lens=True, lens_training_only=True,
                            lens_dropout=0.0, lens_force_zero=original, lens_alignment_weight=1.0)
        cfg["data"]["loader_kwargs"].update(boxes_from=proposal, boxes_as_masks=True, roi_control=control)
        cfg["train"].update(stage=1, select_metric="val_loss")
        add("l_" + arm + "_align", cfg)
        fine = copy.deepcopy(cfg)
        fine["train"].update(stage=3, select_metric="fracture_auprc")
        fine["model"]["lens_alignment_weight"] = 0.1
        fine["init_from"] = str(root / base["checkpoint_dir"] / (cfg["run_name"] + "/best.pt"))
        add("l_" + arm + "_vqa", fine)
    # Resolve upstream separately: do not inherit FracAtlas batching/vocab.
    for name, source in (("roco_alignment", "roco_sealed_alignment_v3.yaml"),
                          ("pmc_alignment", "pmc_sealed_generative_v3.yaml")):
        cfg = load_config(str(root / "refactor_v2/configs" / source))
        cfg["train"].update(epoch_seeded_rng=True, component_batching=None)
        add(name, cfg)
        # Keep the established upstream run names expected by the downstream
        # initialization chain, while preserving these new config paths.
        cfg["run_name"] = "revised_roco_alignment_v3" if name.startswith("roco") else "revised_pmcvqa_alignment_v3"
    destination.mkdir()
    hashes = {}
    for name, cfg in recipes.items():
        path = destination / (name + ".json")
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"status": "candidate_recipes_not_launch_sealed", "recipes": len(recipes), "artifacts": hashes,
        "long_training_authorized": False, "reference_device": "cpu",
        "L_primary_semantic_comparison": "real vs original/matched/shuffled, all with same two-phase horizon; B comparison secondary and budget-unmatched",
        "P_shuffled_rule": "corrupt context only during training; evaluate with own conditioned context, no evaluation-cohort donors",
        "remaining": ["real-data eval-context routing/batch-independence check", "resolved candidate loss/data/init/proposal audit", "final source/config cryptographic seal"],
        "execution_prerequisites": ["fresh ROCO/PMC alignment checkpoints", "new-fold detector and verified complete proposals"],
        "matched_budget_caveat": "same params/steps/exposures does not establish exact FLOPs/runtime equality; measure actual costs"}
    output = root / "refactor_v2/reports/revised_recipe_candidates_v3.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
