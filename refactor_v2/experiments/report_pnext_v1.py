"""Six-cell saved-output P factorial analysis; no model selection or fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from experiments.report_lnext_v1 import metrics,question_strata

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';O=R/'pnext_consistency_bank_runs_v1';Q=R/'pnext_consistency_bank_v1_prepare2';P=R/'step4_localization_v3_1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def main():
    assert read(O/'execution_completed.json')['full_replays_passed']
    for p,h in read(O/'training_release.json')['dependencies'].items():assert sha(Path(p))==h,p
    plan=read(Q/'plan.json');expected=read(P/'model_val.json');scores={};curves={};prob={};strata={};sources={};ids=None
    for arm,cell in plan['cells'].items():
        if cell['reused']:
            last=Path(cell['evaluation']);assert sha(last)==cell['evaluation_sha256'];folder=last.parent;replay_path=Path(cell['replay'])
            assert sha(replay_path)==cell['replay_sha256'];replay=read(replay_path);assert replay['passed'] and replay['records']==1432
        else:
            folder=O/('run_'+arm);replay_path=folder/'fresh_replay.json';replay=read(replay_path);assert replay['passed'] and replay['records_exact']==1432
        assert replay['checkpoint_sha256']==sha(folder/'epoch4.pt');completion=read(folder/'completion.json');assert completion['updates']==292
        curves[arm]=[]
        for epoch in range(1,5):
            path=folder/f'epoch{epoch}_evaluation.json';d=read(path);rows=d['records'];sources[str(path)]=sha(path)
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
            m,y,p,canonical=metrics(rows);current=[r['image_id'] for r in canonical];assert len(current)==len(set(current))==321 and y.sum()==57
            if ids is None:ids=current
            assert ids==current
            for old,new in [('fracture_auprc','AP'),('fracture_auroc','AUROC'),('fracture_sensitivity','sensitivity'),('fracture_specificity','specificity'),('fracture_balanced_accuracy','BA'),('fracture_brier','Brier'),('fracture_ece10','ECE10')]:assert abs(d['metrics'][old]-m[new])<1e-7,(arm,epoch,old)
            curves[arm].append({'epoch':epoch,**m,'worker_history':completion['history'][epoch-1]})
            if epoch==4:scores[arm]=m;prob[arm]=p;strata[arm]=question_strata(rows)
        sources[str(folder/'completion.json')]=sha(folder/'completion.json');sources[str(replay_path)]=sha(replay_path)
    pairs=[(f'P_{b}_cs1',f'P_{b}_cs0') for b in ('train','random','empty')]
    pairs += [(f'P_train_cs{c}',f'P_{b}_cs{c}') for c in (0,1) for b in ('random','empty')]
    lookup=read(P/'component_lookup.json');groups=sorted({lookup[i] for i in ids});index=[np.array([j for j,i in enumerate(ids) if lookup[i]==g]) for g in groups]
    draws={a+'_minus_'+b:[] for a,b in pairs};interactions={'consistency_train_minus_random':[],'consistency_train_minus_empty':[]};rng=np.random.default_rng(20261027)
    for _ in range(2000):
        ix=np.concatenate([index[j] for j in rng.integers(len(groups),size=len(groups))])
        if len(set(y[ix]))<2:continue
        ap={a:float(average_precision_score(y[ix],p[ix])) for a,p in prob.items()}
        for a,b in pairs:draws[a+'_minus_'+b].append(ap[a]-ap[b])
        for control in ('random','empty'):interactions['consistency_train_minus_'+control].append((ap['P_train_cs1']-ap['P_train_cs0'])-(ap[f'P_{control}_cs1']-ap[f'P_{control}_cs0']))
    comparisons={a+'_minus_'+b:{'AP_difference':scores[a]['AP']-scores[b]['AP'],'CI95':np.quantile(draws[a+'_minus_'+b],[.025,.975]).tolist(),'valid_draws':len(draws[a+'_minus_'+b])} for a,b in pairs}
    interaction_results={k:{'bootstrap_mean':float(np.mean(v)),'CI95':np.quantile(v,[.025,.975]).tolist(),'valid_draws':len(v)} for k,v in interactions.items()}
    write(O/'independent_analysis.json',{'scores':scores,'curves':curves,'comparisons':comparisons,'AP_interactions':interaction_results,'reused_cells':[a for a,c in plan['cells'].items() if c['reused']],'dependencies':sources,'bootstrap_seed':20261027,'components':len(groups),'scope':'Inspected development cohort; two previously inspected cells reused; empty bank changes active capacity/dropout path; intervals exploratory without multiplicity adjustment.','source_sha256':sha(Path(__file__))})
    write(O/'question_strata.json',strata)
    lines=['# P consistency/bank diagnostic','', '| Cell | AP | Sensitivity | Specificity | Reused |','|---|---:|---:|---:|---|']+[f"| {a} | {scores[a]['AP']:.6f} | {scores[a]['sensitivity']:.4f} | {scores[a]['specificity']:.4f} | {plan['cells'][a]['reused']} |" for a in plan['cells']]
    lines += ['', 'Fixed seed42 LAST4,256TRAIN/321modelval,292updates per cell. Four new cells passed full1432 fresh-process replay; the two existing cells were hash-verified and reused.','', 'Train-versus-random comparisons address bank semantics. Empty-bank comparisons also change active GM capacity and executed dropout paths. Consistency effects and interactions are reported with fixed-model component bootstrap; no threshold or epoch was selected from these results.','', 'Next authorized work is an input-conditioned P design, then PF with a verified source/graph protocol. No T or combination is promoted by this report.']
    with (O/'P_DIAGNOSTIC_EVIDENCE.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm in plan['cells']:
        axes[0].plot(range(1,5),[r['AP'] for r in curves[arm]],marker='o',label=arm)
        axes[1].plot(range(1,5),[r['BA'] for r in curves[arm]],marker='o',label=arm)
    for ax,title in zip(axes,('Canonical fracture AP','Balanced accuracy at fixed0.5')):
        ax.set(title=title,xlabel='Epoch (73 updates)',ylabel='Model-validation metric');ax.set_xticks(range(1,5));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('P consistency/bank study: inspected development cohort, seed42');fig.savefig(O/'P_learning_curves.png',dpi=180);plt.close(fig)
    names=('independent_analysis.json','question_strata.json','P_DIAGNOSTIC_EVIDENCE.md','P_learning_curves.png','training_release.json','execution_completed.json')
    write(O/'analysis_closure.json',{'artifacts':{str(O/n):sha(O/n) for n in names},'P_diagnostic_complete':True,'architecture_study_complete':False,'production_certified':False})
if __name__=='__main__':
    try:
        while not (O/'execution_completed.json').exists():
            if list(O.glob('failure_*.json')):raise RuntimeError('P runner failed; inspect preserved logs')
            time.sleep(30)
        main()
    except Exception as e:
        if O.exists() and not (O/'analysis_failure.json').exists():write(O/'analysis_failure.json',{'error':repr(e),'preserve_all_outputs':True})
        raise
