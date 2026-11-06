"""Observe native SAM outputs before filtering on fixed original TRAIN case.

No thresholds, outputs, weights, or author source files are changed.
"""
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.favp_visual_bridge_v2 import ROOT, namespace, SOURCE, verify_source
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT = ROOT / 'reports/favp_sam_filter_diagnostic_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    namespace()
    for name in ['build_sam.py','automatic_mask_generator.py','predictor.py','utils/amg.py']:
        verify_source(SOURCE / 'favp/segmed' / name)
    build = importlib.import_module('favp.segmed.build_sam')
    generation = importlib.import_module('favp.segmed.automatic_mask_generator')
    model = build.build_sam_vit_b(256, checkpoint=None, encoder_adapter=True).eval()
    archive = load_archive(ROOT / 'references/native_assets_favp_v1/sam-med2d_b.pth', map_location='cpu')
    model.load_state_dict(archive['model'], strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in archive['model'].items())
    del archive
    model.requires_grad_(False)
    generator = generation.SamAutomaticMaskGenerator(model, points_per_side=50, points_per_batch=16,
        pred_iou_thresh=.8, stability_score_thresh=.9, stability_score_offset=.7,
        box_nms_thresh=.5, crop_n_layers=0, crop_nms_thresh=.7, crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=2, min_mask_region_area=400, output_mode='binary_mask')
    original = generator.predictor.predict_torch
    observed = []
    def observe(*args, **kwargs):
        masks, scores, low = original(*args, **kwargs)
        assert torch.isfinite(scores).all() and torch.isfinite(low).all(), 'Nonfinite native pre-filter output'
        keep = scores.flatten() > generator.pred_iou_thresh
        stability = generation.calculate_stability_score(masks.flatten(0,1)[keep], model.mask_threshold, generator.stability_score_offset)
        observed.append({'scores': scores.detach().flatten().tolist(), 'low_min': float(low.min()),
                         'low_max': float(low.max()), 'iou_pass': int(keep.sum()),
                         'joint_pass': int((stability >= generator.stability_score_thresh).sum()),
                         'stability_after_iou': stability.detach().tolist()})
        return masks, scores, low
    generator.predictor.predict_torch = observe
    batch, _ = batch_data()
    image = np.asarray(batch['images'][0].convert('RGB')).copy()
    proposals = generator.generate(image[:,:,::-1].copy())
    scores = torch.tensor([s for row in observed for s in row['scores']])
    assert scores.numel() == 7500
    assert not proposals, 'Previously accepted fixed-case output changed unexpectedly'
    receipt = {'image_id': batch['image_ids'][0], 'prompt_points': 2500, 'candidate_masks': scores.numel(),
        'native_scores_and_low_logits_finite': True, 'score_min': float(scores.min()), 'score_max': float(scores.max()),
        'score_quantiles': torch.quantile(scores, torch.tensor([0.,.25,.5,.75,.95,1.])).tolist(),
        'passed_native_iou': sum(r['iou_pass'] for r in observed),
        'passed_native_iou_and_stability': sum(r['joint_pass'] for r in observed),
        'final_proposals': len(proposals), 'observations': observed,
        'total_seconds': time.perf_counter()-start,
        'scope': 'read-only pre-filter instrumentation; unchanged native thresholds and original TRAIN image; not threshold tuning or quality evaluation',
        'script_sha256': sha(Path(__file__)), 'quality_released': False}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
