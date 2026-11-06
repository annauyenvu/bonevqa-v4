"""Prespecified four-TRAIN automatic SAM→circle→EVA→QFormer route."""
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.favp_visual_bridge_v2 import VisualBridge, ROOT, SOURCE, namespace, verify_source
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split

OUT = ROOT / 'reports/favp_fixed_train_views_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    start = time.perf_counter()
    indices_path = ROOT / 'reports/vnext_local_tokens_v1/gate_indices.json'
    indices = json.loads(indices_path.read_text())
    assert indices == [8, 5, 16, 26]
    cfg = json.loads((ROOT / 'reports/step4_localization_v3_1/B.json').read_text())
    ds = build_split(build_dataset, cfg['data']['dataset'], 'train', None, True, cfg['data']['loader_kwargs'])
    namespace()
    manifest = json.loads((ROOT / 'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    for name, digest in manifest.items():
        if '/favp/segmed/' in name:
            assert sha(Path(name)) == digest
    build = importlib.import_module('favp.segmed.build_sam')
    generation = importlib.import_module('favp.segmed.automatic_mask_generator')
    sam = build.build_sam_vit_b(256, checkpoint=None, encoder_adapter=True).eval()
    archive = load_archive(ROOT / 'references/native_assets_favp_v1/sam-med2d_b.pth', map_location='cpu')
    sam.load_state_dict(archive['model'], strict=True)
    assert all(torch.equal(v, sam.state_dict()[k]) for k, v in archive['model'].items())
    del archive
    sam.requires_grad_(False)
    generator = generation.SamAutomaticMaskGenerator(sam, points_per_side=50, points_per_batch=16,
        pred_iou_thresh=.8, stability_score_thresh=.9, stability_score_offset=.7,
        box_nms_thresh=.5, crop_n_layers=0, crop_nms_thresh=.7, crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=2, min_mask_region_area=400, output_mode='binary_mask')
    visual = VisualBridge().eval().requires_grad_(False)
    for relative in ['favp/models/fine_grained_visual_prompt.py', 'favp/models/transform.py', 'favp/segmed/utils/transforms.py']:
        verify_source(SOURCE / relative)
    renderer_cls = importlib.import_module('favp.models.fine_grained_visual_prompt').FGVP_ENSEMBLE
    resize_cls = importlib.import_module('favp.segmed.utils.transforms').ResizeLongestSide
    renderer = renderer_cls('red', 2, 'green', .5, 'resize', 224, resize_cls(224),
        torch.tensor([.485,.456,.406]).view(3,1,1)*255,
        torch.tensor([.229,.224,.225]).view(3,1,1)*255, 100, device='cpu')
    records = []
    for index in indices:
        sample = ds[index]
        image = np.asarray(sample['image'].convert('RGB')).copy()
        tick = time.perf_counter()
        proposals = generator.generate(image[:,:,::-1].copy())
        entry = {'train_index': index, 'image_id': sample['image_id'], 'image_rgb_sha256': __import__('hashlib').sha256(image.tobytes()).hexdigest(),
                 'proposal_count': len(proposals), 'proposal_seconds': time.perf_counter()-tick}
        if len(proposals)>32:
            entry['route_accepted'] = False
            entry['reason'] = 'prespecified >32-proposal resource guard; no truncation or threshold adjustment'
        else:
            if proposals:
                masks = torch.from_numpy(np.stack([p['segmentation'] for p in proposals])).unsqueeze(1)
                boxes = torch.from_numpy(np.stack([p['bbox'] for p in proposals])).float()
                boxes[:,2:] += boxes[:,:2]
                centers = torch.stack([boxes[:,0::2].mean(1), boxes[:,1::2].mean(1)],1)
            else:
                masks, boxes, centers = [], [], []
            views = renderer('circle', image, centers, boxes, masks)
            assert len(views) == 1+len(proposals)
            tick = time.perf_counter()
            with torch.no_grad():
                queries, memory = visual(views)
                repeat, memory_repeat = visual(views)
                original, _ = visual(views[:1])
                # Independently verify native local-before-global memory construction.
                encoded = [visual.ln_vision(visual.visual(v.unsqueeze(0))) for v in views]
                expected = torch.cat(encoded[1:]+encoded[:1],1) if len(encoded)>1 else encoded[0]
            assert memory.shape == (1, 257*len(views), 1408)
            assert torch.equal(memory, expected) and torch.equal(memory, memory_repeat)
            assert torch.equal(queries, repeat) and torch.isfinite(queries).all()
            torch.save({'queries': queries, 'original_queries': original, 'image_id': sample['image_id'],
                        'proposal_count': len(proposals)}, OUT / f'index_{index}_queries.pt')
            entry.update(route_accepted=True, views=list(views.shape), memory_shape=list(memory.shape),
                         query_shape=list(queries.shape), repeat_exact=True, local_before_global_exact=True,
                         query_max_abs_vs_original=float((queries-original).abs().max()),
                         visual_seconds=time.perf_counter()-tick)
            del memory, memory_repeat, expected, encoded, views, queries, repeat, original, masks, boxes, centers
        records.append(entry)
        (OUT / f'index_{index}.json').write_text(json.dumps(entry, indent=2))
        del proposals, sample, image
    receipt = {'indices': indices, 'indices_sha256': sha(indices_path), 'records': records,
        'natural_local_branch_cases': sum(r['proposal_count']>0 and r['route_accepted'] for r in records),
        'total_seconds': time.perf_counter()-start, 'quality_released': False,
        'scope': 'four fixed TRAIN interface cases, original images only; native proposals/source RAD thresholds; no targets or annotation ROI used',
        'protocol_sha256': sha(ROOT / 'reports/FAVP_FIXED_TRAIN_ROUTE_CHECK_V1.md'), 'script_sha256': sha(Path(__file__))}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
