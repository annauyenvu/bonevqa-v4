"""Read-only prediction audit after the bounded V run; exclusive new outputs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'refactor_v2/reports/vnext_local_tokens_v1'
P=ROOT/'refactor_v2/reports/step4_localization_v3_1'
ARMS=('B','real_aux','placebo_aux','grid_aux','real_noaux')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def audit():
    assert read(O/'execution_completed.json')['full_replays_passed']
    for p,h in read(O/'training_release.json')['dependencies'].items():assert sha(Path(p))==h,p
    expected=read(P/'model_val.json');scores={};curves={};prob={};sources={};ids=None;y=None
    for arm in ARMS:
        folder=O/('run_'+arm);c=read(folder/'completion.json');r=read(folder/'fresh_replay.json')
        assert c['updates']==292 and len(c['history'])==4
        assert r['passed'] and r['records_exact']==1432 and r['checkpoint_sha256']==sha(folder/'epoch4.pt')
        curves[arm]=[]
        for epoch in range(1,5):
            path=folder/f'epoch{epoch}_evaluation.json';d=read(path);rows=d['records'];sources[str(path)]=sha(path)
            assert len(rows)==len(expected)==1432
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
            canonical=[r for r in rows if r['question']=='Is there a fracture in this X-ray?']
            current=[r['image_id'] for r in canonical];labels=np.array([r['gt']=='yes' for r in canonical]);p=np.array([r['p_yes'] for r in canonical])
            assert len(current)==len(set(current))==321 and labels.sum()==57 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            if ids is None:ids=current;y=labels
            assert ids==current and np.array_equal(y,labels)
            tp=int(((p>=.5)&y).sum());tn=int(((p<.5)&~y).sum())
            m={'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'TP':tp,'FN':int(y.sum())-tp,'TN':tn,'FP':int((~y).sum())-tn,'sensitivity':tp/int(y.sum()),'specificity':tn/int((~y).sum())}
            m['BA']=(m['sensitivity']+m['specificity'])/2
            # Independently verify the canonical clinical endpoint against runner metrics.
            for key,new in [('fracture_sensitivity','sensitivity'),('fracture_specificity','specificity'),('fracture_balanced_accuracy','BA')]:assert abs(d['metrics'][key]-m[new])<1e-10
            m['runner_VQA_metrics']=d['metrics'];m['timing']=c['history'][epoch-1];curves[arm].append(m)
            if epoch==4:scores[arm]=m;prob[arm]=p
        for name in ('completion.json','fresh_replay.json'):sources[str(folder/name)]=sha(folder/name)
    lookup=read(P/'component_lookup.json');groups=sorted({lookup[i] for i in ids});index=[np.array([j for j,i in enumerate(ids) if lookup[i]==g]) for g in groups]
    controls=('B','placebo_aux','grid_aux','real_noaux');draws={a:[] for a in controls};rng=np.random.default_rng(20261025)
    for _ in range(2000):
        ix=np.concatenate([index[j] for j in rng.integers(len(groups),size=len(groups))])
        if len(set(y[ix]))!=2:continue
        aps={a:average_precision_score(y[ix],p[ix]) for a,p in prob.items()}
        for a in controls:draws[a].append(float(aps['real_aux']-aps[a]))
    comparisons={a:{'real_minus_control_AP':scores['real_aux']['AP']-scores[a]['AP'],'CI95':np.quantile(draws[a],[.025,.975]).tolist(),'valid_draws':len(draws[a])} for a in controls}
    provisional=all(comparisons[a]['real_minus_control_AP']>=.005 for a in ('B','placebo_aux','grid_aux'))
    write(O/'independent_analysis.json',{'scores':scores,'curves':curves,'comparisons':comparisons,'provisional_V_signal':provisional,'bootstrap_seed':20261025,'components':len(groups),'dependencies':sources,'analysis_source_sha256':sha(Path(__file__)),'scope':'Inspected model-validation cohort; fixed seed and LAST4; exploratory intervals without multiplicity correction; not independent clinical confirmation.'})
    lines=['# V local-token pilot evidence','', '| Arm | AP | Sensitivity | Specificity |','|---|---:|---:|---:|']
    lines += [f"| {a} | {scores[a]['AP']:.6f} | {scores[a]['sensitivity']:.4f} | {scores[a]['specificity']:.4f} |" for a in ARMS]
    lines += ['',f'Provisional predeclared V signal: {provisional}. No automatic new seed, combination, threshold or epoch search.','', 'All arms: seed42, 256 TRAIN images / 1164 QA, 321 model-validation images / 1432 QA, 4 epochs / 292 updates, full fresh-process replay.','', 'Fixed-grid uses automatic-proposal count, not independent detector-free grids. Local arms use additional views/compute; the auxiliary target is an image-level label, not a crop diagnosis. Existing data are development-informed; component identities are not verified patients.','', 'Next authorized priority: L spatial supervision with original-image-only inference. PF source verification belongs to the subsequent P study.']
    with (O/'V_PILOT_EVIDENCE.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    write(O/'analysis_closure.json',{'artifacts':{str(O/n):sha(O/n) for n in ('independent_analysis.json','V_PILOT_EVIDENCE.md','execution_completed.json','training_release.json')},'V_analysis_complete':True,'architecture_study_complete':False,'production_certified':False})

if __name__=='__main__':
    try:
        while not (O/'execution_completed.json').exists():
            failures=list(O.glob('failure_*.json'))
            if failures:raise RuntimeError('Runner failure: '+str(failures))
            time.sleep(30)
        audit()
    except Exception as e:
        if not (O/'analysis_failure.json').exists():write(O/'analysis_failure.json',{'error':repr(e),'preserve_all_outputs':True})
        raise
