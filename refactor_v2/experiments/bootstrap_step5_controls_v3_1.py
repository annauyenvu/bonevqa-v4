"""Descriptive paired screening uncertainty, never revises frozen selection."""
import numpy as np
from sklearn.metrics import average_precision_score
from experiments.prepare_step4_localization_v3_1 import OUT as PARENT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step56_screening_v3_1'
def main():
    assert not (OUT/'paired_AP_bootstrap.json').exists();s=read(OUT/'screening_selection.json')
    for p,h in s['dependencies'].items():assert sha(p)==h,p
    sources={a:PARENT/('run_'+a)/'epoch_4_evaluation.json' for a in ('B','V_real','V_placebo')}
    sources.update({a:OUT/('run_'+a)/'epoch4_evaluation.json' for a in ('P_real','P_random','P_unconditioned','L_real','L_original','L_placebo')})
    ids=None;data={};y=None
    for a,p in sources.items():
        rows=[r for r in read(p)['records'] if r['question']=='Is there a fracture in this X-ray?']
        current=[r['image_id'] for r in rows]
        if ids is None:ids=current;y=np.array([r['gt']=='yes' for r in rows])
        assert current==ids and [r['gt']=='yes' for r in rows]==y.tolist()
        data[a]=np.array([r['p_yes'] for r in rows])
    lookup=read(PARENT/'component_lookup.json');groups=sorted({lookup[i] for i in ids});indices=[np.array([j for j,i in enumerate(ids) if lookup[i]==g]) for g in groups]
    pairs=[('V_real','V_placebo'),('V_real','B'),('P_real','P_random'),('P_real','P_unconditioned'),('P_real','B'),('L_real','L_original'),('L_real','L_placebo'),('L_real','B')]
    draws={f'{a}_minus_{b}':[] for a,b in pairs};rng=np.random.default_rng(20261024)
    for _ in range(2000):
        pick=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])
        if len(set(y[pick]))<2:continue
        scores={a:float(average_precision_score(y[pick],p[pick])) for a,p in data.items()}
        for a,b in pairs:draws[f'{a}_minus_{b}'].append(scores[a]-scores[b])
    summary={f'{a}_minus_{b}':{'point':s['AP'][a]-s['AP'][b],'CI95':np.quantile(draws[f'{a}_minus_{b}'],[.025,.975]).tolist(),'valid_draws':len(draws[f'{a}_minus_{b}'])} for a,b in pairs}
    dump(OUT/'paired_AP_bootstrap.json',{'passed':True,'seed':20261024,'draws':2000,'comparisons':summary,'scope':'Exploratory inspected development screening; fixed models; no multiplicity adjustment; L-vs-B phase-unmatched; no selection-rule change','selection_sha256':sha(OUT/'screening_selection.json'),'source_sha256':sha(__file__)})
    print(summary)
if __name__=='__main__':main()
