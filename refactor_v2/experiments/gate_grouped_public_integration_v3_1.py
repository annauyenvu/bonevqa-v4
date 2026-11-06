"""Real LAST model integration on inspected modelval prefixes, zero updates."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_public_integration_gate_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def measure(seed):
    import torch
    from src.data import build_dataset
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    from experiments.grouped_public_evaluation_core_v3_1 import predict_records
    check_environment();plan=read(OUT/'plan.json')
    for p,h in plan['dependencies'].items():assert sha(p)==h,p
    bindings=read(REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json');result={}
    for fold,b in bindings['folds'].items():
        cfg=read(b['config']);set_seed(seed)
        model=BoneVQAModel.load(b['checkpoint'],device='cpu').eval()
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        hashes=lambda:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
        before=hashes();expected=read(b['roles']['model_val']['path'])[:32]
        ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
        prefix=[ds[i] for i in range(len(expected))]
        records=predict_records(model,prefix,expected)
        old=REPORTS/'fullfold0_fixed_LAST_evaluation_v3_1/token_mean.json' if fold=='0' else REPORTS/f'grouped_modelval_evaluation_v3_1/fold_{fold}.json'
        assert records==read(old)['records'][:32],fold
        assert before==hashes() and sha(b['checkpoint'])==b['checkpoint_sha256']
        result[fold]={'records':records,'weights_unchanged':True,'trainable_tensors_checked':len(before)}
        del model,ds,prefix;gc.collect()
    return result

def run():
    assert not OUT.exists();OUT.mkdir();start=time.perf_counter()
    paths=[Path(__file__).resolve(),REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json',REPORTS/'grouped_evaluation_bindings_v3_1/independent_audit.json',ROOT/'refactor_v2/experiments/grouped_public_evaluation_core_v3_1.py',ROOT/'refactor_v2/experiments/grouped_calibration_v3_1.py']
    paths+=list((ROOT/'refactor_v2/src').rglob('*.py'))
    for name in ('grouped_public_core_gate_v3_1.json','grouped_calibration_technical_gate_v3_1.json','grouped_bootstrap_technical_gate_v3_1.json'):
        p=REPORTS/name;gate=read(p);assert gate['passed'];paths.append(p)
        for source,h in gate['dependencies'].items():assert sha(source)==h;paths.append(Path(source))
    binding=read(REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json')
    for p,h in binding['dependencies'].items():assert sha(p)==h;paths.append(Path(p))
    for fold in binding['folds']:
        paths.append(REPORTS/'fullfold0_fixed_LAST_evaluation_v3_1/token_mean.json' if fold=='0' else REPORTS/f'grouped_modelval_evaluation_v3_1/fold_{fold}.json')
    dump(OUT/'plan.json',{'dependencies':{str(p):sha(p) for p in paths},'scope':'Inspected modelval first32 per fold, existing batches4; no calibration/test data accessed by model','optimizer_updates':0})
    try:
        dump(OUT/'first.json',measure(90210))
        with (OUT/'replay.log').open('x') as log:subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        dump(OUT/'report.json',{'passed':True,'elapsed_seconds':time.perf_counter()-start,'folds':list(binding['folds']),'all160_records_exact_to_verified_modelval':True,'fresh_process_exact_replay':True,'optimizer_updates':0,'reserved_role_inference':False,'reserved_role_execution_authorized':False,'remaining_gate':'Scoped reserved-role driver and release'})
    except Exception as e:dump(OUT/'failure.json',{'error':repr(e),'preserved_no_automatic_retry':True});raise

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='replay':assert measure(90211)==read(OUT/'first.json')
    else:run()
