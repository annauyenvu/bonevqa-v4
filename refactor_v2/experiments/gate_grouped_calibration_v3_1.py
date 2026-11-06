"""Synthetic-only calibration checks and fresh-process deterministic replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.grouped_calibration_v3_1 import features,fit,apply,PARAMETERS

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_calibration_technical_gate_v3_1.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fixture():
    p=np.array([.02,.1,.2,.35,.4,.55,.6,.75,.85,.97])
    y=np.array([0,0,1,0,0,1,0,1,1,1]);x=features(p).ravel()
    calibration=fit(p,y);assert calibration['passed']
    q=apply(p,calibration)
    # Independently check the regularized binary logistic objective's stationarity.
    z=calibration['slope']*x+calibration['intercept']
    manual=1/(1+np.exp(-z));assert np.max(abs(manual-q))<1e-15
    residual=manual-y
    derivative=[float((x*residual).sum()+calibration['slope']/PARAMETERS['C']),float(residual.sum())]
    assert max(abs(v) for v in derivative)<1e-4,derivative
    assert np.isfinite(features([0,1])).all()
    assert not fit([.2,.8],[1,1])['passed']
    negatives=0
    for pbad,ybad in [([float('nan'),.5],[0,1]),([-.1,.5],[0,1]),([.2,.8],[0,2]),([.2,.8],[0])]:
        try:fit(pbad,ybad)
        except ValueError:negatives+=1
    assert negatives==4
    assert np.array_equal(q>=.5,manual>=.5)
    return {'calibration':calibration,'probabilities':q.tolist(),'objective_derivatives':derivative,'negative_cases':5}

def run():
    assert not OUT.exists(),'Preserve earlier gate'
    start=time.perf_counter();result=fixture()
    replay=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'fixture'],text=True))
    assert result==replay
    dependencies=[Path(__file__).resolve(),ROOT/'refactor_v2/experiments/grouped_calibration_v3_1.py',ROOT/'refactor_v2/reports/GROUPED_EVALUATION_RULES_V3_1.json']
    report={'passed':True,'scope':'Synthetic calibration arithmetic/optimality/negative-input checks and exact fresh-process replay only; no actual dataset, model inference or fit on reserved roles',
            'dependencies':{str(p):sha(p) for p in dependencies},'fixture':result,'elapsed_seconds':time.perf_counter()-start,'reserved_role_inference_authorized':False,'remaining_gates':'Role bindings, checkpoint freeze, public calibration/evaluation runner, bootstrap implementation and replay gates'}
    with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':print(json.dumps(fixture()))
    else:run()
