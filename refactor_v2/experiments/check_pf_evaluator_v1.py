"""Full saved-role evaluator check at PRE-update initialization, no training."""
import json
import math
import time
from pathlib import Path
import torch
from experiments.gate_pf_comparison_v2 import initialize
from experiments.lapa_integrated_gate_v1 import REPORTS,digest
from experiments.prepare_lapa_pilot_v1 import sha
from experiments.lapa_quality_worker_v1 import evaluate,metrics
from src.data import build_dataset
from src.train import build_split

OUT=REPORTS/'pf_evaluator_check_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    prep=REPORTS/'pf_comparison_preparation_v1'
    seal=json.loads((prep/'preparation_seal.json').read_text())
    for path,h in seal['dependencies'].items():assert sha(Path(path))==h,path
    model=initialize('residual_real').eval()
    saved=torch.load(prep/'shared_PREupdate.pt',map_location='cpu')
    assert saved['updates']==0 and digest(model.state_dict().items())==saved['initial_digest']
    state=model.state_dict();state.update(saved['trainable']);model.load_state_dict(state,strict=True)
    assert digest(model.state_dict().items())==saved['initial_digest']
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    val=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    assert len(val)==1432
    rows=evaluate(model,val)
    assert all(len(r['logits'])==41 and all(math.isfinite(x) for x in r['logits']) for r in rows)
    result=metrics(rows);assert result['records']==1432 and result['fracture_records']==321
    (OUT/'rows.json').write_text(json.dumps(rows))
    receipt={'scope':'PRE-update residual_real; full real model_val pipeline check, no tuning',
             'metrics':result,'seconds':time.perf_counter()-start,'quality_released':False,
             'hashes':{str(p):sha(p) for p in [Path(__file__),OUT/'rows.json']}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
