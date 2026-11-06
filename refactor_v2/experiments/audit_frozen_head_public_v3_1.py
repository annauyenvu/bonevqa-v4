"""Saved prediction audit and paired known-component analysis; no inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.evaluate_fullfold0_fixed_LAST_v3_1 import bindings,bootstrap,read,sha,ROOT,CANONICAL
from src.utils import compute_metrics

out=ROOT/'refactor_v2/reports/frozen_head_public_evaluation_v3_1'
assert not (out/'paired_artifact_audit.json').exists()
plan=read(out/'plan.json');report=read(out/'report.json')
for p,h in plan['dependencies'].items():assert sha(p)==h
assert report['candidate_sha256']==sha(out/'candidate.json') and report['plan_sha256']==sha(out/'plan.json')
assert read(out/'replay.json')['passed']
old=read(ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1/token_mean.json')
new=read(out/'candidate.json')
summary={}
for name,data in [('original',old),('head_LAST410',new)]:
    rows=data['records'];assert len(rows)==1432
    assert compute_metrics(rows,fracture_threshold=.5)==data['metrics']
    clinical=[r for r in rows if r['question']==CANONICAL]
    assert len(clinical)==321
    y=np.array([r['gt']=='yes' for r in clinical]);p=np.array([r['p_yes'] for r in clinical]);d=p>=.5
    counts={'TP':int((y&d).sum()),'FN':int((y&~d).sum()),'TN':int((~y&~d).sum()),'FP':int((~y&d).sum())}
    metrics=compute_metrics(clinical,fracture_threshold=.5)
    assert abs(metrics['fracture_auroc']-roc_auc_score(y,p))<1e-12
    assert abs(metrics['fracture_auprc']-average_precision_score(y,p))<1e-12
    assert abs(metrics['fracture_balanced_accuracy']-(counts['TP']/57+counts['TN']/264)/2)<1e-12
    summary[name]={'counts':counts,'clinical':metrics,'VQA':data['metrics']}
for a,b in zip(old['records'],new['records']):
    assert all(a[k]==b[k] for k in ['image_id','question','gt','answer_type'])
    if a['question']!=CANONICAL:assert a==b
_,configs=bindings()
paired=bootstrap({'token_mean':old,'sequence_uniform':new},configs['token_mean'])
paired['direction']='head_LAST410 minus original; Brier/ECE lower is better'
result={'passed':True,'auditor_sha256':sha(__file__),'source_report_sha256':sha(out/'report.json'),
    'replay_sha256':sha(out/'replay.json'),'results':summary,'paired':paired,
    'primary':'fracture_balanced_accuracy','bootstrap_timing':'Post-inference analysis,500 resamples seed20261021 retained from prior method; no model fitting/selection',
    'limitations':'One training seed, inspected development fold; intervals conditional on fixed models/known components, not patient independence or independent confirmation',
    'step3_complete':False,'production_released':False}
(out/'paired_artifact_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'counts':{k:v['counts'] for k,v in summary.items()},'BA':{k:v['clinical']['fracture_balanced_accuracy'] for k,v in summary.items()},'BA_interval':paired['paired_intervals']['fracture_balanced_accuracy'],'sensitivity_interval':paired['paired_intervals']['fracture_sensitivity']},indent=2))
