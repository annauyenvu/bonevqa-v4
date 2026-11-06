"""Synthetic driver role orchestration check; no actual reserved cohort access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_reserved_driver_gate_v3_1.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fixture():
    import torch
    import src.data
    import src.train
    from src.models.bonevqa import BoneVQAModel
    import experiments.run_vl_bounded_pilot_v3_1 as environment
    import experiments.grouped_public_evaluation_core_v3_1 as core
    import experiments.run_grouped_reserved_evaluation_v3_1 as driver
    canonical=core.CANONICAL
    flags={k:False for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt')}
    class Model:
        cfg=dict(flags,fracture_yes_threshold=.5)
        def eval(self):return self
        def trainable_state_dict(self):return {'fixture_weight':torch.tensor([1.])}
        def predict_batch(self,b):
            assert set(b)=={'images','questions'}
            return [{'answer':'yes' if p>=.5 else 'no','p_yes':p} if q==canonical else {'answer':'fixture'} for p,q in zip(b['images'],b['questions'])]
    def rows(prefix):return [{'image_id':prefix+str(i),'question':canonical if i<2 else 'Synthetic OPEN question','answer':['no','yes','fixture'][i],'answer_type':'CLOSED' if i<2 else 'OPEN'} for i in range(3)]
    b={'checkpoint':'synthetic_checkpoint','checkpoint_sha256':'fixture','config':'synthetic_config','roles':{role:{'path':role,'sha256':'fixture','images':2} for role in ('calibration','test')}}
    cfg={'data':{'dataset':'synthetic','loader_kwargs':{}}}
    base={'calibration':rows('cal_'),'test':rows('outer_'),'synthetic_config':cfg}
    calls=[]
    def build(_,name,role,max_samples,real,loader):
        assert role in ('calibration','test') and real and max_samples is None
        assert loader['sealed_records_by_split']=={'calibration':'calibration','test':'test'}
        calls.append(role)
        return [dict(r,image=[.1,.8,.4][i],region='GT_REGION',boxes='GT_ROI',masks='GT_MASK') for i,r in enumerate(base[role])]
    fitted=[];original_fit=core.fit_from_canonical
    def fit_checked(records):
        assert all(r['image_id'].startswith('cal_') for r in records)
        fitted.append([r['gt'] for r in records]);return original_fit(records)
    def invoke():
        with patch.object(driver,'ready',return_value=({'folds':['0']},{'folds':{'0':b}})),patch.object(driver,'read',side_effect=lambda p:base[str(p)]),patch.object(driver,'sha',return_value='fixture'),patch.object(environment,'check_environment'),patch.object(src.train,'build_split',side_effect=build),patch.object(BoneVQAModel,'load',return_value=Model()),patch.object(core,'fit_from_canonical',side_effect=fit_checked):
            return driver.measure('0',90210,False)
    first=invoke();assert calls==['calibration','test'] and len(fitted)==1
    # Outer-label perturbation must not enter fitting or change any predictions.
    for r in base['test'][:2]:r['answer']='yes' if r['answer']=='no' else 'no'
    second=invoke();assert first['calibration']==second['calibration']
    for kind in ('raw_records','calibrated_records'):
        assert [(r['pred'],r.get('p_yes')) for r in first[kind]]==[(r['pred'],r.get('p_yes')) for r in second[kind]]
    assert first['weights_unchanged'] and first['optimizer_updates']==0
    assert fitted==[['no','yes'],['no','yes']]
    # A failed fit preserves raw inference, without a fallback calibrated result.
    with patch.object(core,'fit_from_canonical',return_value={'passed':False,'failure':'synthetic_single_class'}):
        # invoke() installs its fit spy; test the exact downstream failure primitive directly.
        assert core.calibrated_records(first['raw_records'],{'passed':False}) is None
    return {'calibration':first['calibration'],'raw_predictions':[(r['pred'],r.get('p_yes')) for r in first['raw_records']],
            'calls':calls,'fitting_labels':fitted,'outer_label_perturbation_invariant':True,'public_input_keys_checked':True,'zero_updates':True}

def run():
    assert not OUT.exists();start=time.perf_counter();result=fixture()
    # JSON normalizes tuple/list representation for exact fresh-process comparison.
    result=json.loads(json.dumps(result))
    replay=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'fixture'],text=True));assert result==replay
    paths=[Path(__file__).resolve(),ROOT/'refactor_v2/experiments/run_grouped_reserved_evaluation_v3_1.py',ROOT/'refactor_v2/experiments/grouped_public_evaluation_core_v3_1.py',ROOT/'refactor_v2/experiments/grouped_calibration_v3_1.py']
    report={'passed':True,'scope':'Synthetic driver/calibration/test role orchestration and label-perturbation check, exact fresh-process replay; hash binding is separately audited on real metadata; zero actual cohort inference','fixture':result,'dependencies':{str(p):sha(p) for p in paths},'elapsed_seconds':time.perf_counter()-start,'reserved_role_inference':False}
    with OUT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'elapsed_seconds':report['elapsed_seconds'],'reserved_role_inference':False}))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':print(json.dumps(fixture()))
    else:run()
