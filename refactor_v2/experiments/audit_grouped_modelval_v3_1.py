"""Audit saved public modelval predictions, independent clinical counts/ranking metrics."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils import compute_metrics, normalize_answer

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_modelval_evaluation_v3_1'
CANONICAL='Is there a fracture in this X-ray?'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert abs(a-b)<1e-12,(a,b)

def audit(fold):
    assert fold in ('1','2','3','4')
    output=OUT/f'fold_{fold}_independent_metric_audit.json'
    assert not output.exists(),'Preserve prior audit'
    plan=read(OUT/'plan.json')
    for p,h in plan['dependencies'].items():assert sha(p)==h,p
    replay=OUT/f'fold_{fold}_replay.json'
    assert read(replay)['all_records_scores_metrics_exact']
    path=OUT/f'fold_{fold}.json';data=read(path);rows=data['records']
    cfg=read(plan['targets'][fold]['config'])
    splits=cfg['data']['loader_kwargs']['sealed_records_by_split']
    assert splits['calibration'] is None and splits['test'] is None
    expected=read(splits['model_val'])
    assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
    assert compute_metrics(rows,fracture_threshold=.5)==data['metrics']
    for name,subset in [('overall',rows),('closed',[r for r in rows if r['answer_type']=='CLOSED']),('open',[r for r in rows if r['answer_type']=='OPEN'])]:
        correct=sum(normalize_answer(r['gt'])==normalize_answer(r['pred']) for r in subset)
        close(correct/len(subset),data['metrics'][name+'_acc'])
    clinical=[r for r in rows if r['question']==CANONICAL]
    assert len(clinical)==len({r['image_id'] for r in clinical})==321
    assert all(normalize_answer(r['gt']) in ('yes','no') for r in clinical)
    y=np.array([normalize_answer(r['gt'])=='yes' for r in clinical]);p=np.array([r['p_yes'] for r in clinical])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    d=p>=.5
    assert all(normalize_answer(r['pred'])==('yes' if decision else 'no') for r,decision in zip(clinical,d))
    counts={'TP':int((y&d).sum()),'FN':int((y&~d).sum()),'TN':int((~y&~d).sum()),'FP':int((~y&d).sum())}
    sens=counts['TP']/(counts['TP']+counts['FN']);spec=counts['TN']/(counts['TN']+counts['FP'])
    manual={'fracture_sensitivity':sens,'fracture_specificity':spec,'fracture_balanced_accuracy':(sens+spec)/2,'fracture_auroc':float(roc_auc_score(y,p)),'fracture_auprc':float(average_precision_score(y,p)),'fracture_brier':float(((p-y)**2).mean())}
    assert compute_metrics(clinical,fracture_threshold=.5)==data['canonical_metrics']
    for key,value in manual.items():close(value,data['canonical_metrics'][key])
    result={'passed':True,'fold':fold,'prediction_sha256':sha(path),'replay_sha256':sha(replay),'auditor_sha256':sha(__file__),'counts':counts,'independent_clinical_metrics':manual,'QA':len(rows),'scope':'Modelval development diagnostics, existing known-component folds, one training seed; no independent confirmation','reserved_role_inference':False,'step3_complete':False}
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    for fold in sys.argv[1:]:audit(fold)
