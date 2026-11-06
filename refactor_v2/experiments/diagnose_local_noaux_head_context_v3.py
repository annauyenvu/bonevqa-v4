"""Read-only canonical-versus-mixed batching diagnosis; no fitting or updates."""
import json,time,traceback
from pathlib import Path
import torch
from experiments import local_noaux_head_analysis_v2 as h
from experiments.vnext_model_v1 import VNextModel

torch.set_num_threads(8)
dest=h.O/'head_context_diagnostic_v3'
dest.mkdir(exist_ok=False)
h.dump(dest/'started.json',{'wall':time.time(),'no_fitting':True},True)
try:
    model=VNextModel.load(str(h.O/'fold_0/local_real_noaux/LAST.pt'),device='cpu').eval()
    results={}
    for role in ('model_val','calibration','test'):
        x,actual=h.extract(model,'0','local_real_noaux',role)
        expected=[r for r in h.read(h.O/f'fold_0/local_real_noaux/{role}_predictions.json')['records'] if r['question']==h.CAN]
        assert [(r['image_id'],r['gt']) for r in actual]==[(r['image_id'],r['gt']) for r in expected]
        delta=[abs(a['p_yes']-b['p_yes']) for a,b in zip(actual,expected)]
        results[role]={'n':len(actual),'exact':actual==expected,'max_probability_delta':max(delta),'different_probabilities':sum(d!=0 for d in delta),'different_answers':sum(a['pred']!=b['pred'] for a,b in zip(actual,expected))}
        h.dump(dest/'progress.json',results)
    h.dump(dest/'receipt.json',{'roles':results,'no_fitting':True,'no_optimizer_updates':True},True)
except Exception as e:
    h.dump(dest/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()},True)
    raise
