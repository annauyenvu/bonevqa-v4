"""Actual pretrained EVA→QFormer inference on accepted zero-proposal TRAIN case."""
import json
import time
from pathlib import Path
import torch
from experiments.favp_visual_bridge_v2 import VisualBridge,render_original,ROOT
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/favp_visual_bridge_v2'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    sam=json.loads((ROOT/'reports/favp_sam_native_v2/receipt.json').read_text());assert sam['count']==0
    model=VisualBridge().eval();batch,_=batch_data();assert batch['image_ids'][0]==sam['image_id']
    views=render_original(batch['images'][0]);assert views.shape==(1,3,224,224)
    tick=time.perf_counter()
    with torch.no_grad():
        queries,memory=model(views);repeat,other=model(views)
    assert queries.shape==(1,32,768) and memory.shape==(1,257,1408)
    assert torch.isfinite(queries).all() and torch.equal(queries,repeat) and torch.equal(memory,other)
    torch.save({'queries':queries,'memory':memory,'image_id':sam['image_id']},OUT/'features.pt')
    receipt={'image_id':sam['image_id'],'accepted_proposals':0,'renderer_shape':list(views.shape),
             'memory_shape':list(memory.shape),'query_shape':list(queries.shape),'finite_exact_repeat':True,
             'two_forwards_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start,
             'audit':model.audit,'scope':'actual pretrained EVA/QFormer original-only fallback; Vicuna/projection/loss not yet executed',
             'quality_released':False,'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('favp_visual_bridge_v2.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise

