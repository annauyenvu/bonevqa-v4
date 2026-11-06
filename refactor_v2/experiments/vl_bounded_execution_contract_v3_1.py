"""Fail-closed planned pilot caps;validation alone never releases training."""
import copy
import functools
import json
from pathlib import Path

import torch
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_prepared_proposals

ARMS=("V_real","V_location_placebo","L_real","L_original_teacher","L_location_placebo")
INIT="refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt"
INIT_SHA="bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b"
OPTIMIZER={"name":"AdamW","lr":0.0001,"betas":[0.9,0.98],"weight_decay":0.01,"max_grad_norm":1.0,"batch_size":4,"gradient_accumulation":4,"scheduler":"warmup_cosine","warmup_ratio":0.05,"fresh_optimizer_each_phase":True}


@functools.lru_cache(maxsize=1)
def initial_config():
    return torch.load(INIT,map_location="cpu",weights_only=False)["config"]


def phases(arm):
    return [{"stage":3,"epochs":4,"global_epoch_offset":0,"optimizer_steps_cap":296}] if arm.startswith("V_") else [{"stage":1,"epochs":2,"global_epoch_offset":0,"optimizer_steps_cap":148},{"stage":3,"epochs":2,"global_epoch_offset":2,"optimizer_steps_cap":148}]


def model_config(arm):
    result=copy.deepcopy(initial_config())
    result.update(prompt_variant=arm[0],use_visual_prompt=arm.startswith("V_"),use_lens=arm.startswith("L_"),use_latent_prompt=False,use_textual_prompt=False,use_region_in_prompt=False,lens_force_zero=arm=="L_original_teacher")
    return result


def validate_plan(config,*,deep_sources=True):
    root=Path.cwd()
    assert config["scope"]=="bounded_VL_development_plan_v3_1"
    assert config["arm"] in ARMS and config["seed"]==42
    assert config["recipe_status"]=="candidate_not_launch_sealed" and config["quality_training_released"] is False
    assert config["phases"]==phases(config["arm"]) and config["total_optimizer_steps_cap"]==296
    assert config["optimizer"]==OPTIMIZER
    assert config["runtime"]=={"device":"cpu","dtype":"float32","torch_threads":8,"workers":0,"offline":True}
    assert set(config["data_roles"])=={"train","model_val"}
    assert config["data_roles"]=={"train":{"QA_rows":1173,"images":261},"model_val":{"QA_rows":142,"images":32}}
    assert config["classification_threshold"]==0.5 and config["fit_calibration"] is False
    assert config["initialization"]=={"path":str(root/INIT),"sha256":INIT_SHA,"kind":"ROCO_PREtask_NOT_smoke_or_tasktrained"}
    assert config["model"]==model_config(config["arm"])
    assert config["roi_condition"]==("group_translation_location_placebo" if config["arm"].endswith("location_placebo") else "real")
    assert config["primary_checkpoint"]=="LAST296cumulativeupdates"
    assert config["allowed_inference_keys"]==(["images","questions","automatic_boxes"] if config["arm"].startswith("V_") else ["images","questions"])
    if deep_sources:
        assert sha(INIT)==INIT_SHA
        proposals=root/"refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json"
        assert config["proposals"]["path"]==str(proposals) and config["proposals"]["sha256"]==sha(proposals)
        validate_prepared_proposals(proposals)
        controls=root/"refactor_v2/reports/vl_matched_control_audit_v3_1"
        report=controls/"report.json"
        assert config["controls"]["report_sha256"]==sha(report)=="8469fa82341bf13c198a17ae9bc6924d78cc93664823b23cba09bc57de57ea07"
        assert config["controls"]["geometry_path"]==str(controls/"group_translation_location_placebo.json")
        assert config["controls"]["geometry_sha256"]==sha(config["controls"]["geometry_path"])
    return True
