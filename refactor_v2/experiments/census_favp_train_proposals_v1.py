"""Fixed full TRAIN-image native SAM proposal census; no target access."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments.favp_visual_bridge_v2 import ROOT, namespace
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split

OUT = ROOT / 'reports/favp_train_proposal_census_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    cfgpath = ROOT / 'reports/step4_localization_v3_1/B.json'
    cfg = json.loads(cfgpath.read_text())
    kwargs = cfg['data']['loader_kwargs']
    assert not kwargs['with_boxes'] and not kwargs['with_masks']
    manifest = Path(kwargs['sealed_records_by_split']['train'])
    records = json.loads(manifest.read_text())
    assert len(records) == 1164
    # Only image identity/path fields determine census membership and ordering.
    first = {}
    for index, record in enumerate(records):
        if record['image_id'] in first:
            assert record['image'] == first[record['image_id']]['image']
        else:
            first[record['image_id']] = {'image_id': record['image_id'], 'image': record['image'], 'record_index': index}
    cohort = [first[k] for k in sorted(first)]
    assert len(cohort) == 256
    dependencies = [cfgpath, manifest, Path(kwargs['seal_path']), Path(__file__),
        ROOT / 'reports/FAVP_TRAIN_PROPOSAL_CENSUS_PROTOCOL_V1.md',
        ROOT / 'experiments/favp_sam_restricted_loader_v1.py',
        ROOT / 'references/native_assets_favp_v1/sam-med2d_b.pth']
    source = json.loads((ROOT / 'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    for name, digest in source.items():
        if '/favp/segmed/' in name:
            assert sha(Path(name)) == digest
            dependencies.append(Path(name))
    binding = {str(p): sha(p) for p in dependencies}
    (OUT / 'frozen_census.json').write_text(json.dumps({'cohort': cohort, 'dependencies': binding,
        'expected_images': 256, 'wall_budget_seconds': 7200, 'targets_used': False}, indent=2))
    del records
    ds = build_split(build_dataset, cfg['data']['dataset'], 'train', None, True, kwargs)
    assert len(ds) == 1164
    namespace()
    build = importlib.import_module('favp.segmed.build_sam')
    generation = importlib.import_module('favp.segmed.automatic_mask_generator')
    model = build.build_sam_vit_b(256, checkpoint=None, encoder_adapter=True).eval()
    archive = load_archive(ROOT / 'references/native_assets_favp_v1/sam-med2d_b.pth', map_location='cpu')
    model.load_state_dict(archive['model'], strict=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in archive['model'].items())
    del archive
    model.requires_grad_(False)
    generator = generation.SamAutomaticMaskGenerator(model, points_per_side=50, points_per_batch=16,
        pred_iou_thresh=.8, stability_score_thresh=.9, stability_score_offset=.7,
        box_nms_thresh=.5, crop_n_layers=0, crop_nms_thresh=.7, crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=2, min_mask_region_area=400, output_mode='binary_mask')
    results = []
    for item in cohort:
        assert time.perf_counter()-start < 7200, 'Census wall budget exceeded; partial output is not completion'
        assert shutil.disk_usage(OUT).free > 8*1024**3
        sample = ds[item['record_index']]
        assert sample['image_id'] == item['image_id']
        image = np.asarray(sample['image'].convert('RGB')).copy()
        tick = time.perf_counter()
        proposals = generator.generate(image[:,:,::-1].copy())
        row = {'image_id': item['image_id'], 'record_index': item['record_index'], 'shape': list(image.shape),
               'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'count': len(proposals),
               'seconds': time.perf_counter()-tick,
               'proposals': [{k:p[k] for k in ['area','bbox','predicted_iou','stability_score']} for p in proposals]}
        with (OUT / f"{item['image_id']}.json").open('x') as f:
            json.dump(row, f)
        results.append(row)
        (OUT / 'progress.json').write_text(json.dumps({'completed': len(results), 'expected': 256,
            'nonempty': sum(r['count']>0 for r in results), 'elapsed_seconds': time.perf_counter()-start}))
        del sample, image, proposals
    assert len(results) == 256
    assert all(sha(Path(p)) == h for p,h in binding.items())
    (OUT / 'receipt.json').write_text(json.dumps({'complete': True, 'images': 256,
        'nonempty': sum(r['count']>0 for r in results), 'proposal_total': sum(r['count'] for r in results),
        'proposal_max': max(r['count'] for r in results), 'total_seconds': time.perf_counter()-start,
        'frozen_census_sha256': sha(OUT / 'frozen_census.json'), 'dependencies_unchanged': True,
        'quality_released': False, 'scope': 'fixed complete TRAIN image proposal census; no annotation/answer supervision or threshold tuning'}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
