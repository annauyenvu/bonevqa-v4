"""Two real TRAIN-image inference check of pretrained native image/text/MM."""
import json
import time
from pathlib import Path
import torch
from experiments.lapa_native_backbone_v1 import NativeBackbone,ROOT
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/lapa_native_backbone_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    model=NativeBackbone().eval();batch,_=batch_data()
    with torch.no_grad():
        result=model(batch['images'],batch['questions'])
        repeat=model(batch['images'],batch['questions'])
    assert torch.isfinite(result['features']).all() and torch.equal(result['features'],repeat['features'])
    assert result['image_tokens_shape']==[2,577,768]
    assert result['trace']==[x for i in range(1,7) for x in (f'MM{i}',f'LF{i}')]
    report={'strict_backbone_MM_load':True,'audit':model.audit,'image_ids':batch['image_ids'],
            'image_tokens_shape':result['image_tokens_shape'],'text_tokens_shape':result['text_tokens_shape'],
            'output_shape':list(result['features'].shape),'finite_exact_repeat':True,
            'image_token_577_is_NOT_graph_node577':True,'seconds':time.perf_counter()-start,
            'scope':'Actual M3AE image/text/MM weights, random new GM/LF, PF not yet attached, no optimizer or quality',
            'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('lapa_native_backbone_v1.py')]},'quality_released':False}
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(error)},f)
        raise
