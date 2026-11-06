"""Independent pooled/primary bootstrap confusion-count and percentile audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.utils import normalize_answer

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_outer_bootstrap_v3_1';RUN=REPORTS/'grouped_reserved_evaluation_v3_1'
CANONICAL='Is there a fracture in this X-ray?'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert abs(a-b)<1e-12,(a,b)

def stats(counts):
    tp,fn,tn,fp=counts
    s=tp/(tp+fn) if tp+fn else None;t=tn/(tn+fp) if tn+fp else None
    return {'fracture_sensitivity':s,'fracture_specificity':t,'fracture_balanced_accuracy':(s+t)/2 if s is not None and t is not None else None}

def percentile(values,q):
    values=sorted(values);position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(position-lo)*(values[hi]-values[lo])

def main():
    output=OUT/'independent_audit.json';assert not output.exists()
    assert read(OUT/'report.json')['passed_exact_replay']
    plan=read(OUT/'plan.json')
    for p,h in plan['dependencies'].items():assert sha(p)==h,p
    result=read(OUT/'results.json');draws=result['bootstrap']['draw_values']
    assert result['bootstrap']['resamples']==2000 and result['bootstrap']['seed']==20261021
    binding=read(REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json');owner=binding['image_to_known_component']
    matrices=[]
    for fold in ('0','1','2','3','4'):
        data=read(RUN/f'fold_{fold}.json');assert read(RUN/f'fold_{fold}_independent_audit.json')['passed']
        assert data['calibration']['passed'],'Audit requires explicit failure handling for missing calibrated arms'
        components={}
        for r,c in zip(data['raw_records'],data['calibrated_records']):
            if r['question']!=CANONICAL:continue
            counts=components.setdefault(owner[r['image_id']],np.zeros(8,dtype=np.int64))
            y=normalize_answer(r['gt'])=='yes'
            for offset,row in ((0,r),(4,c)):
                d=row['p_yes']>=.5;category=0 if y and d else 1 if y else 2 if not d else 3
                counts[offset+category]+=1
        matrices.append(np.stack([components[k] for k in sorted(components)]))
    pooled=sum((m.sum(axis=0) for m in matrices),start=np.zeros(8,dtype=np.int64))
    for arm,counts in [('raw',pooled[:4]),('calibrated',pooled[4:])]:
        for key,value in stats(counts).items():close(float(value),result['pooled_points'][arm+'.'+key])
    rng=np.random.default_rng(20261021)
    for draw in range(2000):
        aggregate=np.zeros(8,dtype=np.int64)
        for matrix in matrices:
            selected=rng.choice(np.arange(len(matrix)),size=len(matrix),replace=True)
            aggregate+=matrix[selected].sum(axis=0)
        a=stats(aggregate[:4]);b=stats(aggregate[4:])
        for key,value in a.items():
            for arm,expected in [('raw',value),('calibrated',b[key]),('calibrated_minus_raw',b[key]-value if value is not None and b[key] is not None else None)]:
                saved=draws[arm+'.'+key][draw]
                if expected is None:assert saved is None
                else:close(float(expected),saved)
    for key,values in draws.items():
        assert len(values)==2000
        finite=[v for v in values if v is not None and np.isfinite(v)]
        interval=result['bootstrap']['intervals'][key]
        assert interval['valid_draws']==len(finite) and interval['undefined_draws']==2000-len(finite)
        if finite:
            for saved,expected in zip(interval['percentile_2_5_97_5'],[percentile(finite,.025),percentile(finite,.975)]):close(saved,expected)
        else:assert interval['percentile_2_5_97_5'] is None
    report={'passed':True,'auditor_sha256':sha(__file__),'results_sha256':sha(OUT/'results.json'),'pooled_raw_TP_FN_TN_FP':pooled[:4].tolist(),'pooled_calibrated_TP_FN_TN_FP':pooled[4:].tolist(),
            'checks':'Independent canonical component confusion-count matrices; pooled primary points; all2000 paired sensitivity/specificity/BA draws reconstructed independently; all stored intervals/undefined counts independently computed by sorted linear percentile. Ranking/Brier/ECE/VQA points verified in fold audits; full analysis exact fresh-process replay verified separately.',
            'scope':'Conditional fixed-model/single-seed known-component uncertainty; development-informed, not independent patient/clinical confirmation','step3_complete':False,'production_released':False}
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))

if __name__=='__main__':main()
