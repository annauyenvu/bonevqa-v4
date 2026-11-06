"""Descriptive component bootstrap of saved localization; no tuning."""
import json,random,hashlib
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'reports';O=R/'step7_synthesis_v3_1'
def main():
    target=O/'localization_component_intervals.json';assert not target.exists()
    paths=[O/'localization_saved.json',R/'step4_localization_v3_1/model_val.json',R/'step4_localization_v3_1/component_lookup.json']
    data,meta,lookup=[json.loads(p.read_text()) for p in paths]
    meta={r['image_id']:r for r in meta};rows=data['rows'];groups=defaultdict(list)
    for r in rows:groups[lookup[r['image_id']]].append(r)
    def metrics(rs):
        pos=[r for r in rs if r['positive']];fn=[r for r in pos if not r['positive_predictions']['B']]
        result={}
        if pos:
            for arm in ('B','V_real','V_placebo'):result[arm+'_sensitivity']=sum(r['positive_predictions'][arm] for r in pos)/len(pos)
            result['real_minus_placebo_sensitivity']=result['V_real_sensitivity']-result['V_placebo_sensitivity']
            result['IoU05_positive_hit']=sum(r['best_IoU'] is not None and r['best_IoU']>=.5 for r in pos)/len(pos)
        if fn:
            result['real_FN_rescue']=sum(r['positive_predictions']['V_real'] for r in fn)/len(fn)
            result['FN_IoU05_hit']=sum(r['best_IoU'] is not None and r['best_IoU']>=.5 for r in fn)/len(fn)
        return result
    rng=random.Random(20261024); keys=sorted(groups);draws=defaultdict(list)
    for _ in range(2000):
        sampled=[r for _ in keys for r in groups[rng.choice(keys)]]
        for k,v in metrics(sampled).items():draws[k].append(v)
    def quantile(v,p):
        v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
    subgroup=[]
    for hardware in (0,1):
        for category in ('one','multiple'):
            rs=[r for r in rows if r['positive'] and meta[r['image_id']]['hardware']==hardware and (meta[r['image_id']]['fracture_count']==1 if category=='one' else meta[r['image_id']]['fracture_count']>=2)]
            subgroup.append({'hardware':hardware,'fracture_count':category,'positive_images':len(rs),'TP':{a:sum(r['positive_predictions'][a] for r in rs) for a in ('B','V_real','V_placebo')},'metrics':metrics(rs)})
    result={'scope':'Exploratory inspected modelval, fixed seed42 models/threshold0.5; components not verified patients; no multiplicity correction','bootstrap_seed':20261024,'draws':2000,'components':len(keys),'point':metrics(rows),'intervals':{k:{'CI95':[quantile(v,.025),quantile(v,.975)],'valid':len(v)} for k,v in draws.items()},'hardware_count_subgroups':subgroup,'warning':'Zero observed rescue yields a degenerate empirical bootstrap interval; this does not establish a true rescue probability of zero. Tiny subgroup counts cannot support causal claims.','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
