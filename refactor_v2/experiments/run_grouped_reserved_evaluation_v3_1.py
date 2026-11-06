"""Scoped fixed five-fold calibration/outer execution, zero optimizer updates.

Requires real-model integration and complete dependency/role bindings. Failures
are retained without automatic retry. Each fixed fold is replayed in a new process.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_reserved_evaluation_v3_1'
BOUND=REPORTS/'grouped_evaluation_bindings_v3_1'
INTEGRATION=REPORTS/'grouped_public_integration_gate_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def prepare():
    assert not OUT.exists(),'Preserve previous run'
    driver_gate_path=REPORTS/'grouped_reserved_driver_gate_v3_1.json'
    driver_gate=read(driver_gate_path);assert driver_gate['passed']
    for p,h in driver_gate['dependencies'].items():assert sha(p)==h,p
    gate=read(INTEGRATION/'report.json');assert gate['passed'] and gate['fresh_process_exact_replay']
    audit=read(BOUND/'independent_audit.json');assert audit['passed'] and audit['bindings_sha256']==sha(BOUND/'bindings.json')
    binding=read(BOUND/'bindings.json')
    deps={**binding['dependencies'],**read(INTEGRATION/'plan.json')['dependencies']}
    paths=[Path(__file__).resolve(),BOUND/'bindings.json',BOUND/'independent_audit.json',INTEGRATION/'report.json',INTEGRATION/'plan.json',ROOT/'refactor_v2/experiments/grouped_public_evaluation_core_v3_1.py',ROOT/'refactor_v2/experiments/grouped_calibration_v3_1.py',ROOT/'refactor_v2/experiments/grouped_bootstrap_v3_1.py']
    paths.append(driver_gate_path)
    for p in paths:deps[str(p)]=sha(p)
    for p,h in deps.items():assert sha(p)==h,p
    OUT.mkdir();dump(OUT/'execution_release.json',{'dependencies':deps,'folds':['0','1','2','3','4'],'constructor_seeds':[90210,90211],'training_seed':42,'raw_threshold':.5,'scope':'Only frozen per-fold canonical calibration and full outer public prediction with exact fresh-process replay; no model/epoch/threshold search','failure_policy':'Preserve failure; no automatic retry or hyperparameter change','optimizer_updates':0,'independent_confirmation':False,'bootstrap_and_independent_audits_pending':True,'production_released':False})

def ready():
    release=read(OUT/'execution_release.json')
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    return release,read(BOUND/'bindings.json')

def measure(fold,seed,track):
    import torch
    from src.data import build_dataset
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed,compute_metrics
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    from experiments.grouped_public_evaluation_core_v3_1 import predict_records,fit_from_canonical,calibrated_records,CANONICAL
    check_environment();release,binding=ready();assert fold in release['folds'];b=binding['folds'][fold]
    cfg=read(b['config']);loader=dict(cfg['data']['loader_kwargs'])
    loader['sealed_records_by_split']={role:info['path'] for role,info in b['roles'].items()}
    set_seed(seed);model=BoneVQAModel.load(b['checkpoint'],device='cpu').eval()
    assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
    assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
    hashes=lambda:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    before=hashes()
    def infer(role,canonical):
        info=b['roles'][role];assert sha(info['path'])==info['sha256']
        expected=read(info['path'])
        ds=build_split(build_dataset,cfg['data']['dataset'],role,None,True,loader)
        def progress(done,total):
            if track:dump(OUT/'inference_progress.json',{'fold':fold,'role':role,'completed_QA':done,'total_QA':total})
        rows=predict_records(model,ds,expected,canonical_only=canonical,progress=progress)
        canonical_rows=[r for r in rows if r['question']==CANONICAL]
        assert len(canonical_rows)==info['images']==len({r['image_id'] for r in canonical_rows})
        return rows
    calibration_rows=infer('calibration',True)
    calibration=fit_from_canonical(calibration_rows)
    raw=infer('test',False);calibrated=calibrated_records(raw,calibration)
    assert before==hashes() and sha(b['checkpoint'])==b['checkpoint_sha256']
    def metrics(rows):return {'VQA':compute_metrics(rows,fracture_threshold=.5),'canonical':compute_metrics([r for r in rows if r['question']==CANONICAL],fracture_threshold=.5)}
    return {'fold':fold,'checkpoint_sha256':b['checkpoint_sha256'],'calibration_records':calibration_rows,'calibration':calibration,'raw_records':raw,'calibrated_records':calibrated,'raw_metrics':metrics(raw),'calibrated_metrics':metrics(calibrated) if calibrated is not None else None,'weights_unchanged':True,'optimizer_updates':0}

def run():
    release,_=ready();assert not (OUT/'progress.json').exists()
    os.close(os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
    started=time.perf_counter();completed=[];dump(OUT/'progress.json',{'status':'running','pid':os.getpid(),'completed':completed})
    try:
        for fold in release['folds']:
            for mode in ('worker','replay'):
                dump(OUT/'progress.json',{'status':'running','pid':os.getpid(),'completed':completed,'active_fold':fold,'phase':mode})
                with (OUT/f'fold_{fold}_{mode}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),mode,fold],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
            completed.append(fold)
        dump(OUT/'report.json',{'passed_exact_replay':True,'elapsed_seconds':time.perf_counter()-started,'folds':completed,'bootstrap_independent_audits_reports_pending':True,'production_released':False})
        dump(OUT/'progress.json',{'status':'replayed_pending_bootstrap_independent_audits','completed':completed})
    except Exception as e:
        dump(OUT/'failure.json',{'error':repr(e),'preserved_no_automatic_retry':True})
        dump(OUT/'progress.json',{'status':'failed_preserved_no_retry','completed':completed});raise

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='run':run()
    elif mode=='worker':
        started=time.perf_counter();fold=sys.argv[2]
        dump(OUT/f'fold_{fold}.json',measure(fold,90210,True))
        dump(OUT/f'fold_{fold}_timing.json',{'elapsed_seconds':time.perf_counter()-started})
    elif mode=='replay':
        fold=sys.argv[2];assert measure(fold,90211,False)==read(OUT/f'fold_{fold}.json')
        dump(OUT/f'fold_{fold}_replay.json',{'passed':True,'all_records_scores_calibration_metrics_exact':True})
    else:raise ValueError(mode)
