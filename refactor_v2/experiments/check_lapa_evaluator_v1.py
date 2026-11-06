"""Full model-validation endpoint check at PRE-update initialization."""
import json
import time
import torch
from experiments.lapa_quality_worker_v1 import evaluate, metrics, PREP, REPORTS
from experiments.lapa_pilot_runner_v1 import PilotRunner
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split
from pathlib import Path

OUT=REPORTS/'lapa_evaluator_check_v1_retry2'


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(8);torch.manual_seed(42)
    seal=json.loads((PREP/'preparation_seal.json').read_text())
    for p,h in seal['hashes'].items():assert sha(Path(p))==h
    model=PilotRunner('carrier').eval()
    state=torch.load(PREP/'shared_PREupdate.pt',map_location='cpu')
    assert state['updates']==0
    model.load_state_dict(state['state'],strict=True);del state
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    val=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    assert len(val)==1432
    tick=time.perf_counter();rows=evaluate(model,val);seconds=time.perf_counter()-tick
    result=metrics(rows)
    assert result['fracture_records']==321 and len(rows)==1432
    assert all(torch.isfinite(torch.tensor(r['logits'])).all() for r in rows)
    with (OUT/'outputs.json').open('x') as f:json.dump(rows,f)
    with (OUT/'receipt.json').open('x') as f:json.dump({'scope':'PRE-update carrier evaluator check; no optimizer, selection or tuning.',
        'records':1432,'canonical_images':321,'seconds':seconds,'finite_logits':True,'metrics':result,
        'quality_released':False,'outputs_sha256':sha(OUT/'outputs.json'),'script_sha256':sha(Path(__file__))},f,indent=2)
    print(json.dumps({'records':1432,'seconds':seconds,'canonical_images':321,'finite_logits':True}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
