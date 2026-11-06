"""Three additional fixed TRAIN SAM-Huge cases; reuse accepted first views."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from transformers import SamModel,SamImageProcessor,pipeline
from experiments.lens_native_common_v1 import ROOT,SOURCE,sha,verify_source,verify_package,fixed_train_image

OUT=ROOT/'reports/lens_fixed_native_views_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    protocol=ROOT/'reports/LENS_FIXED_NATIVE_VIEW_GATE_PROTOCOL_V1.md'
    (OUT/'frozen_protocol.json').write_text(json.dumps({'indices':[8,5,16,26],'protocol_sha256':sha(protocol),'seed':42,'quality_released':False}))
    first=json.loads((ROOT/'reports/lens_sam_native_recovery_v2/receipt.json').read_text())
    assert first['recovered_view_artifacts']
    for p,h in first['view_artifacts'].items():assert sha(Path(p))==h
    rows=[{'index':8,'image':first['image'],'eligible':True,'view_directory':str(ROOT/'reports/lens_sam_native_v1/views'),
           'retained_masks':first['retained_masks_after_source_filters'],'reuse_receipt_sha256':sha(ROOT/'reports/lens_sam_native_recovery_v2/receipt.json'),
           'alpha':.5,'scope':'reused prior fixed mean-alpha native views'}]
    for p in ['src/localization_lens/sam_augment.py','src/localization_lens/views.py','src/localization_lens/io.py']:verify_source(p)
    sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.sam_augment import generate_masks
    from localization_lens.views import render_views
    folder=verify_package('sam-vit-huge')
    model,loading=SamModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,dtype=torch.float32,output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    generator=pipeline('mask-generation',model=model.eval().requires_grad_(False),
        image_processor=SamImageProcessor.from_pretrained(folder,local_files_only=True),device=-1)
    args=SimpleNamespace(crops_n_layers=0,points_per_batch=8,points_per_crop=32,pred_iou_thresh=.8,stability_score_thresh=.95)
    rng=np.random.default_rng(42)
    for index in [5,16,26]:
        assert time.perf_counter()-start<900,'Fixed native preparation time budget exceeded'
        image,binding=fixed_train_image(index);tick=time.perf_counter()
        masks=generate_masks(generator,image,args);alpha=float(np.clip(rng.normal(.5,.1),0.,1.))
        directory=OUT/binding['image_id'];render_error=None;artifacts={}
        try:artifacts=render_views(image,masks,directory,'sam',alpha)
        except ValueError as exc:render_error=str(exc)
        count=int(np.load(directory/'sam_masks.npz')['masks'].shape[0]) if artifacts else 0
        row={'index':index,'image':binding,'eligible':bool(artifacts),'view_directory':str(directory),
             'raw_selected_masks':len(masks),'retained_masks':count,'alpha':alpha,'render_error':render_error,
             'artifacts':{p:sha(Path(p)) for p in artifacts.values()},'seconds':time.perf_counter()-tick}
        rows.append(row);(OUT/f'index_{index}.json').write_text(json.dumps(row,indent=2))
    eligible=[r for r in rows if r['eligible']]
    (OUT/'receipt.json').write_text(json.dumps({'indices':[8,5,16,26],'cases':rows,'eligible_cases':len(eligible),
        'technical_batch_indices':[r['index'] for r in eligible[:2]],'contrastive_batch_available':len(eligible)>=2,
        'arguments':vars(args),'total_seconds':time.perf_counter()-start,'quality_released':False,
        'scope':'four fixed real TRAIN source-SAM views; no targets/ROI/threshold tuning; eligibility is technical coverage only'},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
