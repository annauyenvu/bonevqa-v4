"""Fixed pooled paired component bootstrap; requires all five outer audits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from src.utils import normalize_answer
from experiments.grouped_bootstrap_v3_1 import bootstrap

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
RUN=REPORTS/'grouped_reserved_evaluation_v3_1';OUT=REPORTS/'grouped_outer_bootstrap_v3_1'
CANONICAL='Is there a fracture in this X-ray?'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def clinical(y,p):
    d=p>=.5;positive=y==1;negative=~positive
    sensitivity=float(d[positive].mean()) if positive.any() else None
    specificity=float((~d[negative]).mean()) if negative.any() else None
    edges=np.linspace(0,1,11);ece=0.
    for i in range(10):
        mask=(p>=edges[i])&((p<edges[i+1]) if i<9 else (p<=edges[i+1]))
        if mask.any():ece+=float(mask.mean())*abs(float(p[mask].mean())-float(y[mask].mean()))
    return {'fracture_sensitivity':sensitivity,'fracture_specificity':specificity,'fracture_balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'fracture_auroc':float(roc_auc_score(y,p)) if positive.any() and negative.any() else None,
            'fracture_auprc':float(average_precision_score(y,p)) if positive.any() else None,
            'fracture_brier':float(((p-y)**2).mean()),'fracture_ece10':ece}

def arm(values,calibrated=False):
    mask=values[:,0]==1;correct=values[:,5 if calibrated else 4]
    result=clinical(values[mask,1],values[mask,3 if calibrated else 2])
    for name,selected in [('overall',np.ones(len(values),dtype=bool)),('closed',values[:,6]==1),('open',values[:,7]==1)]:
        result[name+'_acc']=float(correct[selected].mean()) if selected.any() else None
    return result

def score(rows):
    values=np.asarray([r['v'] for r in rows],dtype=np.float64);raw=arm(values)
    calibrated=arm(values,True) if np.isfinite(values[:,5]).all() else {k:None for k in raw}
    return {**{'raw.'+k:v for k,v in raw.items()},**{'calibrated.'+k:v for k,v in calibrated.items()},
            **{'calibrated_minus_raw.'+k:calibrated[k]-v if calibrated[k] is not None and v is not None else None for k,v in raw.items()}}

def prepare():
    assert not OUT.exists();assert read(RUN/'report.json')['passed_exact_replay']
    binding_path=REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json'
    paths=[Path(__file__).resolve(),binding_path,RUN/'report.json',RUN/'execution_release.json',ROOT/'refactor_v2/experiments/grouped_bootstrap_v3_1.py',REPORTS/'grouped_bootstrap_technical_gate_v3_1.json']
    gate=read(paths[-1]);assert gate['passed']
    for p,h in gate['dependencies'].items():assert sha(p)==h;paths.append(Path(p))
    for fold in ('0','1','2','3','4'):
        audit=RUN/f'fold_{fold}_independent_audit.json';assert read(audit)['passed'] and read(audit)['prediction_sha256']==sha(RUN/f'fold_{fold}.json')
        paths += [RUN/f'fold_{fold}.json',audit,RUN/f'fold_{fold}_replay.json']
    OUT.mkdir();dump(OUT/'plan.json',{'dependencies':{str(p):sha(p) for p in paths},'resamples':2000,'seed':20261021,'primary':'Pooled raw canonical fracture BA at0.5; per-fold point results','comparison':'Calibrated minus raw on identical component draws','fit_during_bootstrap':False,'scope':'Development-informed known-component grouped outer data; one fixed training seed, historical fold0; no untouched/patient-independent confirmation','fresh_process_exact_replay_required':True})

def analyze():
    for p,h in read(OUT/'plan.json')['dependencies'].items():assert sha(p)==h,p
    bindings=read(REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json');owner=bindings['image_to_known_component']
    rows=[];per_fold={};images=set();failed=[]
    for fold in ('0','1','2','3','4'):
        data=read(RUN/f'fold_{fold}.json');raw=data['raw_records'];cal=data['calibrated_records']
        if cal is None:failed.append(fold)
        descriptor=[]
        for i,r in enumerate(raw):
            c=cal[i] if cal is not None else None;canonical=r['question']==CANONICAL
            v=[int(canonical),int(normalize_answer(r['gt'])=='yes'),r.get('p_yes',0.),c.get('p_yes',0.) if c is not None else float('nan'),int(normalize_answer(r['gt'])==normalize_answer(r['pred'])),int(normalize_answer(c['gt'])==normalize_answer(c['pred'])) if c is not None else float('nan'),int(r['answer_type']=='CLOSED'),int(r['answer_type']=='OPEN')]
            descriptor.append({'fold':fold,'component':owner[r['image_id']],'v':v})
            if canonical:assert r['image_id'] not in images;images.add(r['image_id'])
        point=score(descriptor)
        for mode in ('raw','calibrated'):
            expected=data[mode+'_metrics']
            if expected is None:continue
            for k,v in {**{k:v for k,v in expected['canonical'].items() if k.startswith('fracture_') and k not in ('fracture_yes_threshold','fracture_prevalence')},**{k:expected['VQA'][k] for k in ('overall_acc','closed_acc','open_acc')}}.items():
                assert abs(point[mode+'.'+k]-v)<1e-12,(fold,mode,k)
        per_fold[fold]=point;rows.extend(descriptor)
    assert len(images)==4013
    result=bootstrap(rows,score)
    return {'per_fold_points':per_fold,'pooled_points':score(rows),'bootstrap':result,'images':len(images),'QA':len(rows),'failed_calibration_folds':failed,'fit_during_bootstrap':False}

def run():
    assert not (OUT/'progress.json').exists();start=time.perf_counter();dump(OUT/'progress.json',{'status':'running'})
    try:
        dump(OUT/'results.json',analyze())
        with (OUT/'replay.log').open('x') as log:subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        dump(OUT/'report.json',{'passed_exact_replay':True,'elapsed_seconds':time.perf_counter()-start,'independent_bootstrap_audit_and_final_reports_pending':True,'step3_complete':False,'production_released':False})
        dump(OUT/'progress.json',{'status':'replayed_pending_independent_bootstrap_audit'})
    except Exception as e:dump(OUT/'failure.json',{'error':repr(e),'preserved_no_automatic_retry':True});raise

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='replay':assert analyze()==read(OUT/'results.json')
    else:raise ValueError(sys.argv[1])
