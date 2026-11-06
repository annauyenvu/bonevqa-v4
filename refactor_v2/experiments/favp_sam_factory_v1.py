"""Strict native frozen SAMMed2D/source RAD generator, CPU batch16."""
import importlib
from pathlib import Path
import json
import torch
from experiments.favp_visual_bridge_v2 import ROOT, namespace
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.prepare_lapa_pilot_v1 import sha


def make_generator():
    namespace()
    source=json.loads((ROOT/'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    for name,digest in source.items():
        if '/favp/segmed/' in name:assert sha(Path(name))==digest
    build=importlib.import_module('favp.segmed.build_sam')
    generation=importlib.import_module('favp.segmed.automatic_mask_generator')
    model=build.build_sam_vit_b(256,checkpoint=None,encoder_adapter=True).eval()
    archive=load_archive(ROOT/'references/native_assets_favp_v1/sam-med2d_b.pth',map_location='cpu')
    model.load_state_dict(archive['model'],strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in archive['model'].items())
    del archive
    model.requires_grad_(False)
    return generation.SamAutomaticMaskGenerator(model,points_per_side=50,points_per_batch=16,
        pred_iou_thresh=.8,stability_score_thresh=.9,stability_score_offset=.7,
        box_nms_thresh=.5,crop_n_layers=0,crop_nms_thresh=.7,crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=2,min_mask_region_area=400,output_mode='binary_mask')
