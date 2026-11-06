"""Resolve inherited configs and record blockers; never launches training."""
import hashlib
import json
from pathlib import Path
import yaml
from src.protocol import validate_prompt_protocol_config
from src.utils import load_config


def dependencies(path):
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text()) or {}
    result = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    if payload.get("_base_"):
        result.update(dependencies(path.parent / payload["_base_"]))
    return result


def main():
    root = Path.cwd()
    output = root / "refactor_v2/reports/resolved_config_audit_v3.json"
    if output.exists():
        raise RuntimeError("Existing resolved config audit must not be overwritten")
    records = []
    for path in sorted((root / "refactor_v2/configs").glob("*.yaml")):
        cfg = load_config(str(path))
        if int(cfg.get("model", {}).get("prompt_protocol_version", 0)) < 3:
            continue
        blockers = []
        validation_error = None
        try:
            validate_prompt_protocol_config(cfg)
        except Exception as error:
            validation_error = str(error)
            blockers.append("semantic_config_validation_failed")
        data, model, train = cfg["data"], cfg["model"], cfg["train"]
        loader = data.get("loader_kwargs") or {}
        sealed = loader.get("sealed_records_by_split") or {}
        if not sealed:
            blockers.append("not_bound_to_revised_sealed_data")
        elif any(value and not Path(value).is_file() for value in sealed.values()):
            blockers.append("sealed_input_file_missing")
        init = cfg.get("init_from")
        if init and not (root / init).is_file():
            blockers.append("upstream_initialization_missing")
        if model.get("grad_checkpoint", True) or model.get("llm_grad_checkpoint", True):
            blockers.append("stochastic_checkpoint_recipe_not_certified")
        if not train.get("epoch_seeded_rng", False):
            blockers.append("not_using_verified_epoch_seeded_resume_policy")
        if data.get("dataset") == "fracatlas_component_clean" and not train.get("component_batching"):
            blockers.append("common_component_batching_not_bound")
        if model.get("use_visual_prompt") or model.get("use_lens"):
            proposal = loader.get("boxes_from")
            # Explicitly exclude the known historical detector artifact even
            # when it exists: existence alone is not new-fold provenance.
            if not proposal or "detector_clean/" in str(proposal) or "detector_clean_v3/" in str(proposal):
                blockers.append("proposal_not_bound_to_revised_outer_fold")
            elif not (root / proposal).is_file():
                blockers.append("full_fold_proposal_missing")
        if loader.get("roi_control") == "matched_all":
            blockers.append("matched_all_changes_roi_presence_and_view_budget")
        if model.get("textual_prompt_style") == "irrelevant":
            blockers.append("irrelevant_template_not_joint_token_matched")
        if "pilot" in path.name:
            blockers.append("bounded_pilot_only")
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        records.append({"config": str(path.relative_to(root)), "resolved_sha256": digest,
            "source_dependency_sha256": dependencies(path), "semantic_config_passed": validation_error is None,
            "validation_error": validation_error, "launch_blockers": blockers,
            "resolved": cfg})
    result = {"scope": "resolved recipe audit, NOT protocol seal or scientific launch authorization",
        "configs": records, "count": len(records),
        "semantic_validation_passed": sum(row["semantic_config_passed"] for row in records),
        "full_training_authorized": False,
        "limitations": "No fresh model run, detector quality, compute equality or efficacy certified by configuration checks"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "configs"}, indent=2))


if __name__ == "__main__":
    main()
