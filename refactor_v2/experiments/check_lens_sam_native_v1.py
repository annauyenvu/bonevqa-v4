"""Actual pinned SAM-Huge/source Lens augmentation on one fixed TRAIN image."""
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch
from transformers import SamModel,SamImageProcessor,pipeline
from experiments.lens_native_common_v1 import ROOT,SOURCE,sha,verify_source,verify_package,fixed_train_image

OUT=ROOT/'reports/lens_sam_native_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8)
    start=time.perf_counter()
    accepted=json.loads((ROOT/'reports/lens_runtime_v2/receipt.json').read_text())
    assert accepted['runtime_imports_accepted']
    for path in ['src/localization_lens/sam_augment.py','src/localization_lens/views.py','src/localization_lens/io.py']:
        verify_source(path)
    sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.sam_augment import generate_masks
    from localization_lens.views import render_views
    folder=verify_package('sam-vit-huge')
    model,loading=SamModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,
        dtype=torch.float32,output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    model.eval().requires_grad_(False)
    image_processor=SamImageProcessor.from_pretrained(folder,local_files_only=True)
    generator=pipeline('mask-generation',model=model,image_processor=image_processor,device=-1)
    image,binding=fixed_train_image(8)
    args=SimpleNamespace(crops_n_layers=0,points_per_batch=8,points_per_crop=32,
        pred_iou_thresh=.8,stability_score_thresh=.95)
    (OUT/'started.json').write_text(json.dumps({'image':binding,'arguments':vars(args),
        'scope':'automatic original image only; no annotation ROI or targets','source_points_per_batch':64},indent=2))
    tick=time.perf_counter()
    masks=generate_masks(generator,image,args)
    seconds=time.perf_counter()-tick
    outputs={};render_error=None
    try:
        outputs=render_views(image,masks,OUT/'views','sam',.5)
    except ValueError as exc:
        render_error=str(exc)
    receipt={'image':binding,'size':list(image.size),'native_loading_info':loading,'raw_selected_masks':len(masks),
        'source_rendered_views':outputs,'render_error':render_error,'generation_seconds':seconds,
        'total_seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'source_thresholds':vars(args),'adaptations':['CPU FP32','points_per_batch8 instead of source64 to bound RAM'],
        'scope':'one fixed real TRAIN image; native SAM-Huge/source masks and view renderer; not SmolVLM/model training or representative proposal coverage',
        'no_targets_or_annotation_ROI':True,'quality_released':False,'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
