"""Supplementary conditional-positive spatial metrics after the full L audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from experiments.report_lnext_v1 import eval_targets

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';O=R/'lnext_spatial_runs_v1';P=R/'step4_localization_v3_1'
ARMS=('L_real','L_placebo','L_uniform','L_off')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(O/'analysis_closure.json')['L_analysis_complete']
    for p,h in read(O/'analysis_closure.json')['artifacts'].items():assert sha(Path(p))==h,p
    target=eval_targets();lookup=read(P/'component_lookup.json');values={};sources={};ids=None
    for arm in ARMS:
        path=O/f'run_{arm}/epoch4_evaluation.json';sources[str(path)]=sha(path)
        rows=[r for r in read(path)['records'] if r['question']=='Is there a fracture in this X-ray?' and r['gt']=='yes']
        current=[r['image_id'] for r in rows]
        if ids is None:ids=current
        assert ids==current and len(ids)==57
        aps=[]
        for r in rows:
            t,w=target[r['image_id']];p=np.array(r['spatial_probability']);valid=w>0
            assert t.max()>0 and p.shape==t.shape==(196,)
            aps.append(float(average_precision_score(t[valid]>0,p[valid],sample_weight=w[valid])))
        values[arm]=np.array(aps)
    groups=sorted({lookup[i] for i in ids});indices=[np.array([j for j,i in enumerate(ids) if lookup[i]==g]) for g in groups]
    controls=ARMS[1:];draws={a:[] for a in controls};rng=np.random.default_rng(20261030)
    for _ in range(2000):
        pick=np.concatenate([indices[j] for j in rng.integers(len(groups),size=len(groups))])
        for a in controls:draws[a].append(float((values['L_real'][pick]-values[a][pick]).mean()))
    baseline_path=R/'vnext_local_tokens_v1/run_B/epoch4_evaluation.json';sources[str(baseline_path)]=sha(baseline_path)
    baseline={r['image_id']:r['p_yes']>=.5 for r in read(baseline_path)['records'] if r['question']=='Is there a fracture in this X-ray?'}
    subsets={'B_false_negative':[j for j,i in enumerate(ids) if not baseline[i]],'B_true_positive':[j for j,i in enumerate(ids) if baseline[i]]}
    result={'scope':'Descriptive source-box-overlap patch AP, conditional on57positive development images. Spatial head is not in QA inference path; box targets are not pixel-level fracture masks. Exploratory bootstrap, not independent clinical evidence.','positive_images':len(ids),'positive_components':len(groups),'macro_box_patch_AP':{a:float(v.mean()) for a,v in values.items()},'real_minus_control_AP':{a:{'point':float((values['L_real']-values[a]).mean()),'CI95':np.quantile(draws[a],[.025,.975]).tolist(),'draws':2000} for a in controls},'baseline_subsets':{name:{'images':len(ix),'macro_box_patch_AP':{a:float(v[ix].mean()) if ix else None for a,v in values.items()}} for name,ix in subsets.items()},'bootstrap_seed':20261030,'dependencies':sources,'source_sha256':sha(Path(__file__)),'clinical_threshold_or_training_changes':False}
    with (O/'spatial_summary.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('dependencies','scope')}))
if __name__=='__main__':main()
