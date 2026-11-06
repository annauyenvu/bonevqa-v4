"""Synthetic-only component bootstrap and paired-index/replay verification."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.grouped_bootstrap_v3_1 import groups,index_draws,bootstrap,summarize,SEED,RESAMPLES

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_bootstrap_technical_gate_v3_1.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fixture():
    rows=[{'fold':f,'component':c,'raw':v,'calibrated':1-v} for f,c,v in [('0','A',0.),('0','A',1.),('0','B',1.),('1','C',0.),('1','D',1.)]]
    draws=list(index_draws(rows));assert len(draws)==2000
    # Independent explicit fixture ownership checks preserve every row of a selected group.
    rng=np.random.default_rng(SEED)
    for indices in draws:
        a=rng.choice(['A','B'],2,replace=True).tolist();b=rng.choice(['C','D'],2,replace=True).tolist()
        expected=[]
        for c in a+b:expected += [i for i,r in enumerate(rows) if r['component']==c]
        assert indices==expected
        assert sum(rows[i]['fold']=='1' for i in indices)==2
        assert indices.count(0)==indices.count(1)
    def score(sample):
        raw=sum(r['raw'] for r in sample)/len(sample)
        cal=sum(r['calibrated'] for r in sample)/len(sample)
        return {'raw':raw,'calibrated':cal,'difference':cal-raw,'undefined':None}
    result=bootstrap(rows,score)
    for a,b,d in zip(result['draw_values']['raw'],result['draw_values']['calibrated'],result['draw_values']['difference']):
        assert abs(a+b-1)<1e-15 and abs(d-(b-a))<1e-15
    # All possible component choices give an independent exact support for fixture means.
    support=set()
    for choices in itertools.product(('A','B'),('A','B'),('C','D'),('C','D')):
        sample=[r for c in choices for r in rows if r['component']==c]
        support.add(sum(r['raw'] for r in sample)/len(sample))
    assert all(v in support for v in result['draw_values']['raw'])
    assert result['intervals']['undefined']=={'valid_draws':0,'undefined_draws':2000,'percentile_2_5_97_5':None}
    assert summarize([None,float('nan'),1,2])['valid_draws']==2
    for invalid in ([],[{'fold':'0','component':'A'},{'fold':'1','component':'A'}]):
        try:groups(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid partition accepted')
    return {'intervals':result['intervals'],'draw_sha256':hashlib.sha256(json.dumps(result['draw_values'],sort_keys=True).encode()).hexdigest(),'resamples':RESAMPLES,'seed':SEED,'negative_cases':2}

def run():
    assert not OUT.exists(),'Preserve prior gate'
    started=time.perf_counter();result=fixture()
    replay=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'fixture'],text=True))
    assert result==replay
    paths=[Path(__file__).resolve(),ROOT/'refactor_v2/experiments/grouped_bootstrap_v3_1.py',ROOT/'refactor_v2/reports/GROUPED_EVALUATION_RULES_V3_1.json']
    report={'passed':True,'scope':'Synthetic known-component resampling, paired indices, undefined metrics and exact fresh-process replay; no actual cohort data or inference',
            'fixture':result,'dependencies':{str(p):sha(p) for p in paths},'elapsed_seconds':time.perf_counter()-started,'reserved_role_inference_authorized':False}
    with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':print(json.dumps(fixture()))
    else:run()
