"""Post-run artifact/metric checks and paired development bootstrap; no fitting."""
import json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from experiments.prepare_step4_localization_v3_1 import OUT,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
Q='Is there a fracture in this X-ray?'
def main():
    release=read(OUT/'execution_release.json')
    for p,h in release['dependencies'].items(): assert sha(p)==h,p
    expected=read(OUT/'model_val.json'); data={}; histories={}; deps={}
    for arm in ('B','V_real','V_placebo'):
        folder=OUT/('run_'+arm); done=read(folder/'completion.json')
        assert done['updates']==292 and not done['disposable'] and len(done['history'])==5
        histories[arm]=done['history']; deps[str(folder/'completion.json')]=sha(folder/'completion.json')
        for epoch in range(5):
            rows=read(folder/f'epoch_{epoch}_evaluation.json')['records']
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
            canonical=[r for r in rows if r['question']==Q]
            assert len(canonical)==len({r['image_id'] for r in canonical})==321
            y=np.array([r['gt']=='yes' for r in canonical]); p=np.array([r['p_yes'] for r in canonical]); d=p>=.5
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            manual={'fracture_auprc':float(average_precision_score(y,p)),'fracture_auroc':float(roc_auc_score(y,p)),'fracture_sensitivity':float(d[y].mean()),'fracture_specificity':float((~d[~y]).mean())}
            for k,v in manual.items(): assert abs(v-done['history'][epoch]['metrics'][k])<1e-12
            if epoch==4: data[arm]=(y,p,[r['image_id'] for r in canonical])
    lookup=read(OUT/'component_lookup.json'); groups=sorted({lookup[i] for i in data['B'][2]})
    indices=[np.array([j for j,i in enumerate(data['B'][2]) if lookup[i]==g]) for g in groups]
    rng=np.random.default_rng(20261022); draws=[]
    for _ in range(2000):
        pick=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))]); y=data['B'][0][pick]
        if len(set(y))<2: continue
        draws.append(average_precision_score(y,data['V_real'][1][pick])-average_precision_score(y,data['V_placebo'][1][pick]))
    delta=average_precision_score(data['V_real'][0],data['V_real'][1])-average_precision_score(data['V_placebo'][0],data['V_placebo'][1])
    dump(OUT/'quality_artifact_metric_audit.json',{'passed':True,'history':histories,'dependencies':deps,'primary_AP_real_minus_placebo':float(delta),'component_bootstrap2000_CI95':np.quantile(draws,[.025,.975]).tolist(),'valid_draws':len(draws),'scope':'seed42 inspected development cohort; no independent confirmation','fresh_full_quality_output_replay_pending':True,'historical_LP_evidence':'historical_LP_evidence.json','step4_complete':False,'next':'Fresh-process full quality output replay and B/V/L/P synthesis/exit review, no automatic seed expansion'})
    print('Post-run metrics and paired AP bootstrap audited; fresh quality replay still required')
if __name__=='__main__': main()
