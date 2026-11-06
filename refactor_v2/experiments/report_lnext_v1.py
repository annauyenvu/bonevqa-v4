"""Independent saved-output L metrics and V/L question strata. No fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score,roc_auc_score
from src.preprocessing import canonicalize_boxes,exif_orientation,canonicalize_raster
from src.utils import normalize_answer
from experiments.lnext_spatial_v1 import spatial_targets

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';O=R/'lnext_spatial_runs_v1';P=R/'step4_localization_v3_1';V=R/'vnext_local_tokens_v1'
ARMS=('L_real','L_placebo','L_uniform','L_off')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def check(d):
    for p,h in d.items():assert sha(Path(p))==h,p
def metrics(rows):
    canonical=[r for r in rows if r['question']=='Is there a fracture in this X-ray?'];y=np.array([r['gt']=='yes' for r in canonical]);p=np.array([r['p_yes'] for r in canonical]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    tp=int(((p>=.5)&y).sum());tn=int(((p<.5)&~y).sum())
    out={'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'sensitivity':tp/int(y.sum()),'specificity':tn/int((~y).sum()),'Brier':float(np.mean((p-y)**2)),'TP':tp,'FN':int(y.sum())-tp,'TN':tn,'FP':int((~y).sum())-tn}
    out['BA']=(out['sensitivity']+out['specificity'])/2
    ece=0
    for i in range(10):
        ix=(p>=i/10)&(p<(i+1)/10) if i<9 else (p>=.9)&(p<=1)
        if ix.any():ece+=ix.mean()*abs(p[ix].mean()-y[ix].mean())
    out['ECE10']=float(ece)
    correct=np.array([normalize_answer(r['gt'])==normalize_answer(r['pred']) for r in rows]);out['overall_exact_accuracy']=float(correct.mean())
    for kind in ('CLOSED','OPEN'):
        ix=np.array([r['answer_type']==kind for r in rows]);out[kind+'_exact_accuracy']=float(correct[ix].mean())
    return out,y,p,canonical
def question_strata(rows):
    result={}
    for q in sorted({r['question'] for r in rows}):
        rs=[r for r in rows if r['question']==q];result[q]={'n':len(rs),'correct':sum(normalize_answer(r['gt'])==normalize_answer(r['pred']) for r in rs)}
    return result
def eval_targets():
    out={}
    for row in read(P/'model_val.json'):
        if row['question']!='Is there a fracture in this X-ray?':continue
        path=ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'/row['image']
        with Image.open(path) as image:
            raw_size=image.size;orientation=exif_orientation(image);size=canonicalize_raster(image).size;ann=(row['width'],row['height'])
            if orientation==1:assert ann==raw_size;boxes=row['boxes']
            elif ann==size and ann!=raw_size:boxes=row['boxes']
            elif ann==raw_size and ann!=size:boxes=canonicalize_boxes(row['boxes'],raw_size,orientation)
            elif not row['boxes']:boxes=None
            else:raise AssertionError('Ambiguous spatial eval canvas: '+row['image_id'])
        t,w,_=spatial_targets(size,boxes);out[row['image_id']]=(t,w)
    assert len(out)==321;return out
def main():
    assert read(O/'execution_completed.json')['full_replays_passed'];check(read(O/'training_release.json')['dependencies'])
    expected=read(P/'model_val.json');sources={};scores={};curves={};prob={};strata={};spatial={};reference_ids=None;target=eval_targets()
    for arm in ARMS:
        folder=O/('run_'+arm);c=read(folder/'completion.json');replay=read(folder/'fresh_replay.json')
        assert c['updates']==292 and replay['passed'] and replay['records_exact']==1432 and sha(folder/'epoch4.pt')==replay['checkpoint_sha256']
        curves[arm]=[]
        for epoch in range(1,5):
            path=folder/f'epoch{epoch}_evaluation.json';d=read(path);rows=d['records'];sources[str(path)]=sha(path)
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
            m,y,p,canonical=metrics(rows);ids=[r['image_id'] for r in canonical]
            assert len(ids)==len(set(ids))==321 and y.sum()==57
            if reference_ids is None:reference_ids=ids
            assert ids==reference_ids
            for old,new in [('fracture_auprc','AP'),('fracture_auroc','AUROC'),('fracture_sensitivity','sensitivity'),('fracture_specificity','specificity'),('fracture_balanced_accuracy','BA'),('fracture_brier','Brier'),('fracture_ece10','ECE10')]:assert abs(d['metrics'][old]-m[new])<1e-7,(arm,epoch,old)
            curves[arm].append({'epoch':epoch,**m,'worker_timing':c['history'][epoch-1]})
            if epoch==4:
                scores[arm]=m;prob[arm]=p;strata[arm]=question_strata(rows);spatial[arm]=[]
                for r in canonical:
                    t,w=target[r['image_id']];s=np.array(r['spatial_probability']);assert s.shape==t.shape==(196,) and np.isfinite(s).all()
                    ix=w>0;entry={'image_id':r['image_id'],'fractional_coverage_Brier':float(((s-t)**2*w).sum()/w.sum())}
                    if t.max()>0:entry['box_overlap_patch_AP']=float(average_precision_score((t[ix]>0).astype(int),s[ix],sample_weight=w[ix]))
                    spatial[arm].append(entry)
        for n in ('completion.json','fresh_replay.json'):sources[str(folder/n)]=sha(folder/n)
    # Existing V results are re-counted by exact question; no new inference.
    for arm in ('B','real_aux','placebo_aux','grid_aux','real_noaux'):
        path=V/f'run_{arm}/epoch4_evaluation.json';rows=read(path)['records'];sources[str(path)]=sha(path);strata['Vstudy_'+arm]=question_strata(rows)
    lookup=read(P/'component_lookup.json');groups=sorted({lookup[i] for i in reference_ids});ixs=[np.array([j for j,i in enumerate(reference_ids) if lookup[i]==g]) for g in groups]
    controls=('L_off','L_placebo','L_uniform');draws={a:[] for a in controls};rng=np.random.default_rng(20261026)
    for _ in range(2000):
        ix=np.concatenate([ixs[j] for j in rng.integers(len(groups),size=len(groups))])
        if len(set(y[ix]))<2:continue
        ap={a:float(average_precision_score(y[ix],p[ix])) for a,p in prob.items()}
        for a in controls:draws[a].append(ap['L_real']-ap[a])
    comparisons={a:{'real_minus_control_AP':scores['L_real']['AP']-scores[a]['AP'],'CI95':np.quantile(draws[a],[.025,.975]).tolist(),'valid_draws':len(draws[a])} for a in controls}
    # Off/B agreement is checked, not used to overwrite either historical file.
    off=read(O/'run_L_off/epoch4_evaluation.json')['records'];b=read(V/'run_B/epoch4_evaluation.json')['records'];shared_fields=('image_id','question','gt','answer_type','pred','p_yes')
    agreement=all({k:r[k] for k in shared_fields if k in r}=={k:s[k] for k in shared_fields if k in s} for r,s in zip(off,b)) and len(off)==len(b)
    provisional=all(v['real_minus_control_AP']>=.005 for v in comparisons.values())
    write(O/'independent_analysis.json',{'scores':scores,'curves':curves,'comparisons':comparisons,'L_off_vs_new_B_all_records_exact':agreement,'provisional_L_signal':provisional,'bootstrap_seed':20261026,'components':len(groups),'dependencies':sources,'source_sha256':sha(Path(__file__)),'development_informed':True})
    write(O/'question_strata.json',strata);write(O/'spatial_head_analysis.json',{'records':spatial,'scope':'Box-derived weak patch targets, evaluation-only; head is not in answer path; no threshold fitted.'})
    lines=['# L original-image spatial pilot','', '| Arm | AP | Sensitivity | Specificity |','|---|---:|---:|---:|']+[f"| {a} | {scores[a]['AP']:.6f} | {scores[a]['sensitivity']:.4f} | {scores[a]['specificity']:.4f} |" for a in ARMS]
    lines+=['',f'Predeclared provisional L signal: {provisional}. L_off vs current new B outputs exact: {agreement}.','', 'Four stage3 QA epochs / 292 updates per arm, seed42, common256TRAIN/321modelval. All full1432-record fresh-process replays passed. Public inference uses original images/questions only. Spatial auxiliary targets are TRAIN box occupancy; placebo/uniform controls share foreground mass.','', 'These development results do not certify an independent cohort or production readiness. Next priority is P consistency/bank ablations; PF source verification remains required before graph integration.']
    with (O/'L_PILOT_EVIDENCE.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm in ARMS:
        axes[0].plot(range(1,5),[r['AP'] for r in curves[arm]],marker='o',label=arm)
        axes[1].plot(range(1,5),[r['BA'] for r in curves[arm]],marker='o',label=arm)
    for ax,title in zip(axes,('Fracture average precision','Fracture balanced accuracy, fixed0.5')):
        ax.set(title=title,xlabel='Epoch (73 updates each)',ylabel='Model-validation metric');ax.set_xticks(range(1,5));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Original-image L pilot: inspected development cohort, seed42')
    fig.savefig(O/'L_learning_curves.png',dpi=180);plt.close(fig)
    names=('independent_analysis.json','question_strata.json','spatial_head_analysis.json','L_PILOT_EVIDENCE.md','L_learning_curves.png','training_release.json','execution_completed.json')
    write(O/'analysis_closure.json',{'artifacts':{str(O/n):sha(O/n) for n in names},'L_analysis_complete':True,'architecture_study_complete':False,'production_certified':False})

if __name__=='__main__':
    try:
        while not (O/'execution_completed.json').exists():
            if list(O.glob('failure_*.json')):raise RuntimeError('L runner failure; inspect preserved logs')
            time.sleep(30)
        main()
    except Exception as e:
        if O.exists() and not (O/'analysis_failure.json').exists():write(O/'analysis_failure.json',{'error':repr(e),'preserve_all_outputs':True})
        raise
