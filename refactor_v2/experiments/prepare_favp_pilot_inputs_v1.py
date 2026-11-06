"""Freeze subset/schedules before any new validation proposal inference."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.favp_sam_factory_v1 import make_generator
from experiments.favp_circle_cache_v1 import render_circle
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split

OUT=ROOT/'reports/favp_pilot_inputs_v1'
CANONICAL='Is there a fracture in this X-ray?'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    start=time.perf_counter()
    generation=json.loads((ROOT/'reports/favp_public_generation_v2/receipt.json').read_text())
    assert generation['no_target_argument'] and generation['greedy_repeat_exact']
    cfgpath=ROOT/'reports/step4_localization_v3_1/B.json'
    cfg=json.loads(cfgpath.read_text());kw=cfg['data']['loader_kwargs']
    assert not kw['with_boxes'] and not kw['with_masks']
    subset={};dependencies=[cfgpath,Path(kw['seal_path']),ROOT/'reports/FAVP_BOUNDED_ADAPTED_PILOT_PROTOCOL_V2.md',
        Path(__file__),ROOT/'experiments/favp_sam_factory_v1.py',ROOT/'experiments/favp_circle_cache_v1.py']
    for role,count,total in [('train',32,256),('model_val',8,321)]:
        path=Path(kw['sealed_records_by_split'][role]);dependencies.append(path)
        records=json.loads(path.read_text())
        canonical=[{'image_id':r['image_id'],'image':r['image'],'record_index':i,'question':r['question']}
                   for i,r in enumerate(records) if r['question']==CANONICAL]
        assert len(canonical)==total and len({r['image_id'] for r in canonical})==total
        subset[role]=sorted(canonical,key=lambda r:r['image_id'])[:count]
    assert not ({r['image_id'] for r in subset['train']} & {r['image_id'] for r in subset['model_val']})
    schedules=[]
    for epoch in [1,2]:
        gen=torch.Generator().manual_seed(42+epoch)
        schedules.append([subset['train'][i]['record_index'] for i in torch.randperm(32,generator=gen).tolist()])
    bindings={str(p):sha(p) for p in dependencies}
    frozen={'subset':subset,'schedules':schedules,'dependencies':bindings,'seed':42,'updates':64,'quality_released':False}
    # Written before the SAM model or any selected validation raster is opened.
    (OUT/'selection_and_schedules.json').write_text(json.dumps(frozen,indent=2))
    generator=make_generator()
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,kw)
    caches=[]
    for selected in subset['model_val']:
        sample=ds[selected['record_index']]
        assert sample['image_id']==selected['image_id']
        image=sample['image'].convert('RGB');raster=np.asarray(image).copy()
        tick=time.perf_counter();proposals=generator.generate(raster[:,:,::-1].copy())
        assert len(proposals)<=32,'Resource guard: no proposal truncation or threshold adjustment'
        boxes=[p['bbox'] for p in proposals]
        views=render_circle(image,boxes)
        mask_independence=None
        if proposals:
            masks=torch.from_numpy(np.stack([p['segmentation'] for p in proposals])).unsqueeze(1)
            actual=render_circle(image,boxes,masks)
            assert torch.equal(views,actual),'Circle compact-cache rendering differs from actual SAM masks'
            mask_independence=True
        row={'image_id':selected['image_id'],'record_index':selected['record_index'],
            'rgb_sha256':hashlib.sha256(raster.tobytes()).hexdigest(),'boxes_xywh':boxes,'count':len(proposals),
            'renderer_shape':list(views.shape),'circle_mask_pixels_unused_verified':mask_independence,
            'seconds':time.perf_counter()-tick,'scope':'automatic original-image-only cache; no target or annotation ROI'}
        (OUT/f"{selected['image_id']}.json").write_text(json.dumps(row,indent=2))
        caches.append(row)
    assert all(sha(Path(p))==h for p,h in bindings.items())
    (OUT/'public_validation_cache.json').write_text(json.dumps(caches,indent=2))
    (OUT/'receipt.json').write_text(json.dumps({'complete':True,'train':32,'model_val':8,
        'validation_nonempty':sum(r['count']>0 for r in caches),'targets_used_for_selection_or_cache':False,
        'selection_sha256':sha(OUT/'selection_and_schedules.json'),
        'public_cache_sha256':sha(OUT/'public_validation_cache.json'),'dependencies_unchanged':True,
        'seconds':time.perf_counter()-start,'quality_released':False},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
