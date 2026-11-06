"""Fixed LAST modelval only, public inputs, independent-process exact replay."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
REPORTS = ROOT/'refactor_v2/reports'
RUN = REPORTS/'grouped_native_training_v3_1'
OUT = REPORTS/'grouped_modelval_evaluation_v3_1'
CANONICAL = 'Is there a fracture in this X-ray?'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def prepare():
    from experiments.run_grouped_native_v3_1 import validate
    release=validate()
    assert read(RUN/'progress.json')['status']=='trained_pending_independent_evaluation'
    assert not OUT.exists()
    paths=[Path(__file__).resolve(),RUN/'release.json']+list((ROOT/'refactor_v2/src').rglob('*.py'))
    targets={}
    for fold,b in release['bindings'].items():
        audit=REPORTS/f'grouped_fold{fold}_execution_artifact_audit_v3_1.json'
        assert read(audit)['passed'] and read(audit)['independent_quality_audit_pending']
        cfg=read(b['config']); splits=cfg['data']['loader_kwargs']['sealed_records_by_split']
        assert splits['calibration'] is None and splits['test'] is None
        ck=Path(b['checkpoint_dir'])/'last.pt'
        assert sha(ck)==read(RUN/f'fold_{fold}_completion.json')['last_sha256']
        paths += [Path(b['config']),ck,audit,Path(splits['model_val']),Path(cfg['data']['sealed_answer_vocab']),ck.parent/'history.json']
        targets[fold]={'config':b['config'],'checkpoint':str(ck)}
    OUT.mkdir()
    dump(OUT/'plan.json',{'dependencies':{str(p):sha(p) for p in paths},'targets':targets,'public_inputs':['images','questions'],'batch_size':4,'threshold':0.5,'constructor_seeds':[90210,90211],'calibration_test_allowed':False,'optimizer_updates':0,'scope':'Development-informed modelval only; no selection or independent confirmation','independent_metric_audit_pending':True})

def measure(fold,seed,track):
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.protocol import records_of
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed,compute_metrics
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();plan=read(OUT/'plan.json')
    for p,h in plan['dependencies'].items(): assert sha(p)==h,p
    target=plan['targets'][fold];cfg=read(target['config']);path=Path(target['checkpoint'])
    set_seed(seed);model=BoneVQAModel.load(str(path),device='cpu').eval()
    assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
    assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
    hashes=lambda:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    before=hashes()
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    expected=records_of(ds);rows=[]
    for offset in range(0,len(ds),4):
        b=collate_fn([ds[i] for i in range(offset,min(offset+4,len(ds)))])
        with torch.no_grad(): predictions=model.predict_batch({'images':b['images'],'questions':b['questions']})
        assert len(predictions)==len(b['questions'])
        for j,p in enumerate(predictions):
            r={'image_id':b['image_ids'][j],'question':b['questions'][j],'gt':b['answers'][j],'answer_type':b['answer_types'][j],'pred':p['answer']}
            if 'p_yes' in p:r['p_yes']=p['p_yes']
            if r['question']==CANONICAL:assert 'p_yes' in r
            rows.append(r)
        if track and offset%100==0:dump(OUT/'inference_progress.json',{'fold':fold,'completed_QA':len(rows),'total_QA':len(ds)})
    assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
    assert before==hashes()
    metrics=compute_metrics(rows,fracture_threshold=.5)
    history=read(path.parent/'history.json')[-1]
    differences={k[4:]:metrics[k[4:]]-v for k,v in history.items() if k.startswith('val_') and k!='val_loss'}
    assert all(abs(v)<1e-12 for v in differences.values()),differences
    clinical=[r for r in rows if r['question']==CANONICAL]
    assert len(clinical)==len({r['image_id'] for r in clinical})==321
    return {'records':rows,'metrics':metrics,'canonical_metrics':compute_metrics(clinical,fracture_threshold=.5),'trainer_metric_differences':differences,'weights_unchanged':True,'checkpoint_sha256':sha(path)}

def run():
    assert not (OUT/'progress.json').exists()
    start=time.perf_counter();dump(OUT/'progress.json',{'status':'running','completed':[]})
    try:
        completed=[]
        for fold in read(OUT/'plan.json')['targets']:
            for mode in ('worker','replay'):
                with (OUT/f'fold_{fold}_{mode}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),mode,fold],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            completed.append(fold);dump(OUT/'progress.json',{'status':'running','completed':completed})
        dump(OUT/'report.json',{'passed_exact_replay':True,'elapsed_seconds':time.perf_counter()-start,'independent_metric_audit_pending':True,'reserved_role_inference':False})
        dump(OUT/'progress.json',{'status':'replayed_pending_independent_metric_audit','completed':completed})
    except Exception as e:
        dump(OUT/'failure.json',{'error':repr(e),'preserved_no_automatic_retry':True});raise

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='run':run()
    elif mode=='worker':dump(OUT/f'fold_{sys.argv[2]}.json',measure(sys.argv[2],90210,True))
    elif mode=='replay':
        fold=sys.argv[2];assert measure(fold,90211,False)==read(OUT/f'fold_{fold}.json')
        dump(OUT/f'fold_{fold}_replay.json',{'passed':True,'all_records_scores_metrics_exact':True})
    else:raise ValueError(mode)
