"""Recover completed SAM/view artifacts after JSON-set serialization failure.

No automatic masks are regenerated; native load metadata is reverified.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import SamModel
from experiments.lens_native_common_v1 import ROOT,sha,verify_package,fixed_train_image

OUT=ROOT/'reports/lens_sam_native_recovery_v2'
PRIOR=ROOT/'reports/lens_sam_native_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8)
    failure=json.loads((PRIOR/'failure.json').read_text())
    assert 'set is not JSON serializable' in failure['error']
    started=json.loads((PRIOR/'started.json').read_text())
    image,binding=fixed_train_image(8);assert binding==started['image']
    folder=verify_package('sam-vit-huge')
    model,loading=SamModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,
        dtype=torch.float32,output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    # Convert all collection-only load diagnostics, not the model state.
    serializable={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
    views=PRIOR/'views';masks=np.load(views/'sam_masks.npz')['masks'].astype(bool)
    assert masks.ndim==3 and tuple(masks.shape[1:])==(image.height,image.width) and masks.shape[0]>0
    union=np.asarray(Image.open(views/'sam_union.png'))>0
    assert np.array_equal(union,np.logical_or.reduce(masks))
    paths=[views/n for n in ['sam_unicolor.png','sam_multicolor.png','sam_masked.png','sam_union.png','sam_masks.npz']]
    for path in paths[:3]:assert Image.open(path).size==image.size
    receipt={'recovered_view_artifacts':True,'automatic_generation_rerun':False,
        'native_SAM_load_reverified':True,'state_tensors':len(model.state_dict()),'loading_info':serializable,
        'image':binding,'retained_masks_after_source_filters':int(masks.shape[0]),
        'view_artifacts':{str(p):sha(p) for p in paths},'arguments':started['arguments'],
        'generation_seconds':None,'timing_limit':'Original generation timing/raw pre-render count were not saved because receipt serialization failed; do not invent them.',
        'scope':'actual completed source SAM-Huge masks/rendered views recovered; not SmolVLM training or representative cohort coverage',
        'quality_released':False,'failure_preserved':str(PRIOR/'failure.json'),'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
