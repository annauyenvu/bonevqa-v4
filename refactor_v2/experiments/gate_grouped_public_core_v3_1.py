"""Synthetic public input routing, metadata/order and calibration application gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from experiments.grouped_public_evaluation_core_v3_1 import CANONICAL,predict_records,fit_from_canonical,calibrated_records

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_public_core_gate_v3_1.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fixture():
    class Predictor:
        def __init__(self):self.calls=[]
        def predict_batch(self,b):
            assert set(b)=={'images','questions'}
            self.calls.append((list(b['images']),list(b['questions'])))
            return [({'answer':'yes' if p>=.5 else 'no','p_yes':p} if q==CANONICAL else {'answer':'fixture'}) for p,q in zip(b['images'],b['questions'])]
    expected=[{'image_id':str(i),'question':CANONICAL if i%2==0 else 'Synthetic question','answer':'yes' if i>=2 else 'no','answer_type':'CLOSED' if i%2==0 else 'OPEN'} for i in range(6)]
    dataset=[dict(e,image=p,region='GT_SENTINEL',boxes='GT_BOX_SENTINEL',masks='GT_MASK_SENTINEL') for e,p in zip(expected,[.1,.2,.8,.4,.7,.6])]
    model=Predictor();raw=predict_records(model,dataset,expected)
    assert [len(c[0]) for c in model.calls]==[4,2]
    calibration_rows=predict_records(Predictor(),dataset,expected,canonical_only=True)
    assert [r['image_id'] for r in calibration_rows]==['0','2','4']
    cal=fit_from_canonical(calibration_rows);assert cal['passed']
    adjusted=calibrated_records(raw,cal)
    assert all(a==b for a,b in zip(raw,adjusted) if a['question']!=CANONICAL)
    assert calibrated_records(raw,{'passed':False}) is None
    # Alter all metadata together: predictions must depend only on image/question.
    changed=[dict(e,answer='metadata_changed',answer_type='OPEN') for e in expected]
    changed_ds=[dict(s,answer='metadata_changed',answer_type='OPEN',region='CHANGED',boxes='CHANGED',masks='CHANGED') for s in dataset]
    rerun=predict_records(Predictor(),changed_ds,changed)
    assert [(r['pred'],r.get('p_yes')) for r in raw]==[(r['pred'],r.get('p_yes')) for r in rerun]
    negatives=0
    for call in (lambda:predict_records(Predictor(),dataset,list(reversed(expected))),lambda:fit_from_canonical(calibration_rows+calibration_rows[:1]),lambda:fit_from_canonical(raw)):
        try:call()
        except ValueError:negatives+=1
    assert negatives==3
    return {'raw':raw,'calibration':cal,'adjusted':adjusted,'negative_cases':negatives,'public_keys':['images','questions']}

def run():
    assert not OUT.exists()
    result=fixture();replay=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'fixture'],text=True));assert result==replay
    paths=[Path(__file__).resolve(),ROOT/'refactor_v2/experiments/grouped_public_evaluation_core_v3_1.py',ROOT/'refactor_v2/experiments/grouped_calibration_v3_1.py']
    report={'passed':True,'scope':'Synthetic input routing/order/metadata perturbation/calibration application and exact fresh-process replay; not real-model integration or reserved-role release',
            'fixture':result,'dependencies':{str(p):sha(p) for p in paths},'reserved_role_inference_authorized':False}
    with OUT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'reserved_role_inference':False,'negative_cases':3}))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':print(json.dumps(fixture()))
    else:run()
