"""Real automatic proposal route from pinned FAVP SAMMed2D on one TRAIN image."""
import importlib
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
SOURCE=ROOT/'references/favp_author_b43941ba'
OUT=ROOT/'reports/favp_sam_native_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    manifest=json.loads((ROOT/'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    for name,h in manifest.items():
        if '/favp/segmed/' in name:assert sha(Path(name))==h,name
    # Namespace isolation avoids importing unrelated GPU/LLM registry modules.
    for name,path in [('favp',SOURCE/'favp'),('favp.segmed',SOURCE/'favp/segmed')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    build=importlib.import_module('favp.segmed.build_sam')
    generation=importlib.import_module('favp.segmed.automatic_mask_generator')
    model=build.build_sam_vit_b(256,checkpoint=None,encoder_adapter=True).eval()
    checkpoint=ROOT/'references/native_assets_favp_v1/sam-med2d_b.pth'
    archive=load_archive(checkpoint,map_location='cpu')
    model.load_state_dict(archive['model'],strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in archive['model'].items())
    del archive
    model.requires_grad_(False)
    generator=generation.SamAutomaticMaskGenerator(model,points_per_side=50,points_per_batch=16,
        pred_iou_thresh=.8,stability_score_thresh=.9,stability_score_offset=.7,
        box_nms_thresh=.5,crop_n_layers=0,crop_nms_thresh=.7,crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=2,min_mask_region_area=400,output_mode='binary_mask')
    batch,_=batch_data();image=np.asarray(batch['images'][0].convert('RGB'))
    # Preserve author's cv2.imread->generator BGR boundary, disclose explicitly.
    image=image[:,:,::-1].copy()
    tick=time.perf_counter();proposals=generator.generate(image);seconds=time.perf_counter()-tick
    records=[]
    for p in proposals:
        assert p['segmentation'].shape==image.shape[:2] and np.isfinite(p['predicted_iou'])
        records.append({k:p[k] for k in ['area','bbox','predicted_iou','stability_score']})
    if proposals:np.savez_compressed(OUT/'automatic_masks.npz',masks=np.stack([p['segmentation'] for p in proposals]))
    receipt={'image_id':batch['image_ids'][0],'original_shape':list(image.shape),'strict_model_load':True,
             'proposals':records,'count':len(proposals),'generation_seconds':seconds,'total_seconds':time.perf_counter()-start,
             'grid_points':2500,'points_per_batch':16,'author_points_per_batch':256,
             'adaptations':['CPU execution','batch16 instead of256 to bound proposal memory','isolated package loading; source unchanged'],
             'no_annotation_ROI_or_targets':True,'scope':'Actual SAMMed2D automatic proposal route only; no EVA/QFormer/Vicuna yet',
             'source_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(checkpoint),'quality_released':False}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
