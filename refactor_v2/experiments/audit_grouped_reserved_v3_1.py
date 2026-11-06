"""Independent saved outer artifact/fit/metric audit; no image inference or search."""
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from src.utils import compute_metrics,normalize_answer

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_reserved_evaluation_v3_1'
CANONICAL='Is there a fracture in this X-ray?'
PARAMETERS=dict(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42,fit_intercept=True,class_weight=None)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b,tolerance=1e-12):assert abs(a-b)<tolerance,(a,b)
def identity(r):return r['image_id'],r['question'],r['gt'],r['answer_type']

def audit(fold):
    assert fold in ('0','1','2','3','4');output=OUT/f'fold_{fold}_independent_audit.json';assert not output.exists()
    release=read(OUT/'execution_release.json')
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    replay=OUT/f'fold_{fold}_replay.json';assert read(replay)['all_records_scores_calibration_metrics_exact']
    path=OUT/f'fold_{fold}.json';data=read(path)
    binding=read(REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json');b=binding['folds'][fold]
    assert data['checkpoint_sha256']==b['checkpoint_sha256'] and data['weights_unchanged'] and data['optimizer_updates']==0
    calibration_expected=[r for r in read(b['roles']['calibration']['path']) if r['question']==CANONICAL]
    outer_expected=read(b['roles']['test']['path'])
    cal=data['calibration_records'];raw=data['raw_records'];adjusted=data['calibrated_records']
    assert [identity(r) for r in cal]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in calibration_expected]
    assert [identity(r) for r in raw]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in outer_expected]
    assert len(cal)==len({r['image_id'] for r in cal})==b['roles']['calibration']['images']
    assert not {r['image_id'] for r in cal}&{r['image_id'] for r in raw}
    p=np.array([r['p_yes'] for r in cal]);y=np.array([normalize_answer(r['gt'])=='yes' for r in cal],dtype=int)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    saved=data['calibration'];assert saved['parameters']==PARAMETERS
    if len(np.unique(y))<2:
        assert not saved['passed'] and saved['failure']=='single_class' and adjusted is None
    else:
        clipped=np.clip(p,1e-6,1-1e-6);x=np.log(clipped/(1-clipped)).reshape(-1,1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');independent=LogisticRegression(**PARAMETERS).fit(x,y)
        failed=any(issubclass(w.category,ConvergenceWarning) for w in caught) or not np.isfinite([*independent.coef_.ravel(),*independent.intercept_]).all()
        assert saved['passed']==(not failed)
        if saved['passed']:
            close(saved['slope'],float(independent.coef_[0,0]));close(saved['intercept'],float(independent.intercept_[0]))
            assert adjusted is not None
            for a,c in zip(raw,adjusted):
                assert identity(a)==identity(c)
                if a['question']!=CANONICAL:assert a==c
                else:
                    clipped=float(np.clip(a['p_yes'],1e-6,1-1e-6));z=saved['slope']*np.log(clipped/(1-clipped))+saved['intercept']
                    probability=float(np.exp(-np.logaddexp(0,-z)))
                    close(c['p_yes'],probability)
                    assert c['pred']==('yes' if probability>=.5 else 'no')
        else:assert adjusted is None
    summaries={}
    for name,rows in [('raw',raw),('calibrated',adjusted)]:
        if rows is None:continue
        metrics=data[name+'_metrics'];assert compute_metrics(rows,fracture_threshold=.5)==metrics['VQA']
        clinical=[r for r in rows if r['question']==CANONICAL]
        assert len(clinical)==len({r['image_id'] for r in clinical})==b['roles']['test']['images']
        assert compute_metrics(clinical,fracture_threshold=.5)==metrics['canonical']
        y=np.array([normalize_answer(r['gt'])=='yes' for r in clinical]);p=np.array([r['p_yes'] for r in clinical]);d=p>=.5
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert all(normalize_answer(r['pred'])==('yes' if decision else 'no') for r,decision in zip(clinical,d))
        counts={'TP':int((y&d).sum()),'FN':int((y&~d).sum()),'TN':int((~y&~d).sum()),'FP':int((~y&d).sum())}
        sensitivity=counts['TP']/int(y.sum());specificity=counts['TN']/int((~y).sum())
        manual={'fracture_sensitivity':sensitivity,'fracture_specificity':specificity,'fracture_balanced_accuracy':(sensitivity+specificity)/2,'fracture_auroc':float(roc_auc_score(y,p)),'fracture_auprc':float(average_precision_score(y,p)),'fracture_brier':float(((p-y)**2).mean())}
        for k,v in manual.items():close(v,metrics['canonical'][k])
        for key,subset in [('overall',rows),('closed',[r for r in rows if r['answer_type']=='CLOSED']),('open',[r for r in rows if r['answer_type']=='OPEN'])]:
            close(sum(normalize_answer(r['gt'])==normalize_answer(r['pred']) for r in subset)/len(subset),metrics['VQA'][key+'_acc'])
        summaries[name]={'counts':counts,'manual_clinical':manual,'VQA':metrics['VQA']}
    report={'passed':True,'fold':fold,'prediction_sha256':sha(path),'replay_sha256':sha(replay),'auditor_sha256':sha(__file__),'calibration':saved,'summary':summaries,'scope':'Development-informed component-grouped outer results, one fixed training seed; fold0 historical reuse; not untouched clinical confirmation','step3_complete':False,'production_released':False}
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'fold':fold,'counts':{k:v['counts'] for k,v in summaries.items()},'balanced_accuracy':{k:v['manual_clinical']['fracture_balanced_accuracy'] for k,v in summaries.items()}}),flush=True)

if __name__=='__main__':
    for fold in sys.argv[1:]:audit(fold)
