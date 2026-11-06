"""Full1432-record native-backbone public evaluator at fresh initialization."""
import json
import math
import time
from pathlib import Path
import torch
from experiments.gate_lapa_native_scheduled_v2 import initialize
from experiments.lapa_native_quality_worker_v1 import evaluate,metrics
from experiments.lapa_integrated_gate_v1 import REPORTS,digest
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split

OUT=REPORTS/'lapa_native_evaluator_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    prep=REPORTS/'lapa_native_preparation_v1'
    seal=json.loads((prep/'preparation_seal.json').read_text())
    for p,h in seal['dependencies'].items():assert sha(Path(p))==h,p
    model=initialize('pf_author').eval()
    shared=torch.load(prep/'shared_PREupdate.pt',map_location='cpu',weights_only=True)
    assert shared['updates']==0 and digest(model.state_dict().items())==shared['initial_digest']
    full=model.state_dict();full.update(shared['mutable']);model.load_state_dict(full,strict=True)
    assert digest(model.state_dict().items())==shared['initial_digest'];del full,shared
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    val=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs']);assert len(val)==1432
    rows=evaluate(model,val)
    assert len(rows)==1432 and all(len(r['logits'])==41 and all(math.isfinite(x) for x in r['logits']) for r in rows)
    calculated=metrics(rows);assert calculated['fracture_records']==321
    (OUT/'rows.json').write_text(json.dumps(rows))
    receipt={'scope':'PRE-update native-backbone real-data evaluator, no tuning','metrics':calculated,
             'seconds':time.perf_counter()-start,'hashes':{str(p):sha(p) for p in [OUT/'rows.json',Path(__file__)]},'quality_released':False}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
