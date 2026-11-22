"""Unified ablation registry.  Every arm shares splits, backbone, budget and seeds;
only the keys below differ.  ``proposals`` names the TRAIN proposal source."""
from __future__ import annotations

import copy

SEEDS = (42, 43, 44)
BUDGET = {"epochs": 2, "batch_size": 4, "grad_accum": 4, "lr": 1e-4, "weight_decay": 0.01, "max_grad_norm": 1.0}

_OFF = {"vnext_enabled": True, "vg_ground": False, "vg_contrast": False, "vg_recovery": False,
        "vg_token_mode": "crop", "vg_recovery_fallback": "global", "vnext_aux_weight": 0.0, "vnext_force_zero": False}

ARMS = {
    "V4":          {"model": {}, "proposals": "insample_v4"},
    "V4-CF":       {"model": {}, "proposals": "crossfit"},
    "V4-G":        {"model": {"vg_ground": True}, "proposals": "crossfit"},
    "V4-GC":       {"model": {"vg_ground": True, "vg_contrast": True}, "proposals": "crossfit"},
    "V4-GCR":      {"model": {"vg_ground": True, "vg_contrast": True, "vg_recovery": True}, "proposals": "crossfit"},
    "V4-GCR-flag": {"model": {"vg_ground": True, "vg_contrast": True, "vg_recovery": True, "vg_token_mode": "flag"},
                    "proposals": "crossfit"},
    # Secondary recovery mode (multiscale fixed-grid crops when m(I)=0).
    "V4-GCR-grid": {"model": {"vg_ground": True, "vg_contrast": True, "vg_recovery": True,
                              "vg_recovery_fallback": "grid"}, "proposals": "crossfit"},
}

# Inference-time crop interventions, evaluated on V4-G / V4-GC / V4-GCR checkpoints.
CROP_INTERVENTIONS = ("correct", "shifted", "masked_lesion")


def arm_model_cfg(base_model_cfg: dict, arm: str) -> dict:
    cfg = copy.deepcopy(base_model_cfg)
    cfg.update(copy.deepcopy(_OFF))
    cfg.update(copy.deepcopy(ARMS[arm]["model"]))
    cfg["vg_variant"] = arm
    return cfg
